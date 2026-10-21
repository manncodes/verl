# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrated importance sampling (CIS, arXiv 2609.32444): operator, config validation and trainer wiring."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.core_algos import compute_policy_loss_bypass_mode, compute_policy_loss_vanilla
from verl.trainer.ppo.rollout_corr_helper import (
    apply_bypass_mode,
    compute_cis_weights,
    compute_rollout_correction_and_add_to_batch,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.config import omega_conf_to_dataclass, validate_config

LAM, KAPPA = 2.3, 5e-3
CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config")
CIS = ("algorithm.rollout_correction.rollout_is=token", f"algorithm.rollout_correction.rollout_is_cis_lambda={LAM}")
MICRO_BATCHES = (
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
)


def reference_cis_weight(logp_train, logp_infer, lam=LAM, kappa=KAPPA):
    """``cis_weight`` of the reference implementation, kzhao5/CIS-RL@a36a387 cis/operator.py."""
    log_k = logp_train.float() - logp_infer.float()
    log_k = torch.where(torch.isfinite(log_k), log_k, torch.zeros_like(log_k))
    phi = torch.clamp(1.0 - torch.exp(logp_train.float()), min=kappa)
    return torch.minimum(torch.exp(log_k), 1.0 + lam * phi)


def random_logps(shape=(64, 128), seed=0):
    """Confident and uncertain tokens with a 0.3 nat engine mismatch, as in the reference tests."""
    g = torch.Generator().manual_seed(seed)
    old = -(torch.rand(shape, generator=g) ** 4) * 8.0
    rollout = torch.clamp(old - torch.randn(shape, generator=g) * 0.3, max=0.0)
    return old, rollout


def padding_mask(shape, seed=0):
    lengths = torch.randint(1, shape[1] + 1, (shape[0],), generator=torch.Generator().manual_seed(seed))
    return (torch.arange(shape[1]) < lengths[:, None]).float()


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def cis(old, rollout, mask=None, lam=LAM, kappa=KAPPA):
    return compute_cis_weights(old, rollout, torch.ones_like(old) if mask is None else mask, lam, kappa)[0]


def compose_config(*overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="ppo_trainer", overrides=[*MICRO_BATCHES, *overrides])


@pytest.mark.parametrize("lam, kappa", [(LAM, KAPPA), (0.0, KAPPA), (1.6, 0.0), (9.2, 0.1), (LAM, 1.0)])
def test_matches_reference_operator(lam, kappa):
    old, rollout = random_logps()
    torch.testing.assert_close(cis(old, rollout, lam=lam, kappa=kappa), reference_cis_weight(old, rollout, lam, kappa))


def test_cap_tightens_with_confidence_down_to_the_floor():
    p = torch.tensor([[0.2, 0.9, 0.999, 1.0]])
    weights, metrics = compute_cis_weights(p.log(), torch.full_like(p, -20.0), torch.ones_like(p), LAM, KAPPA)
    torch.testing.assert_close(weights, 1 + LAM * torch.tensor([[0.8, 0.1, KAPPA, KAPPA]]))
    assert metrics["rollout_is_cis_truncated_fraction"] == 1.0
    assert metrics["rollout_is_cis_floor_fraction"] == 0.5


def test_sub_cap_weights_are_the_exact_untruncated_ratio():
    old, rollout = random_logps()
    ratio = torch.exp(old - rollout)
    below = ratio < 1 + LAM * (1 - old.exp()).clamp(min=KAPPA)
    weights = cis(old, rollout)
    # no lower truncation: ratios below one, down to the smallest, pass through bit for bit
    assert (ratio[below] < 0.5).any()
    assert torch.equal(weights[below], ratio[below])
    assert (weights[~below] < ratio[~below]).all()


def test_extreme_and_non_finite_log_ratios():
    inf = math.inf
    old = torch.tensor([[0.0, -1e-9, -50.0, -1e4, -inf, -inf, 0.0]])
    rollout = torch.tensor([[-1e4, -200.0, -1e-6, -1e-6, -inf, -1.0, -inf]])
    weights, metrics = compute_cis_weights(old, rollout, torch.ones_like(old), LAM, KAPPA)
    torch.testing.assert_close(weights, reference_cis_weight(old, rollout))
    assert weights[0, :2].tolist() == pytest.approx([1 + LAM * KAPPA] * 2)
    assert weights[0, 4:].tolist() == [1.0, 1.0, 1.0]
    assert all(math.isfinite(v) for v in metrics.values())


def test_padding_is_zero_and_masked_garbage_does_not_leak():
    old, rollout = random_logps((16, 32))
    mask = padding_mask(old.shape)
    pad = mask == 0
    garbage = torch.tensor([math.nan, math.inf, -math.inf, 1e4]).repeat(pad.sum().item() // 4 + 1)[: pad.sum()]
    dirty_old, dirty_rollout = old.clone(), rollout.clone()
    dirty_old[pad], dirty_rollout[pad] = garbage, garbage.flip(0)

    weights, metrics = compute_cis_weights(dirty_old, dirty_rollout, mask, LAM, KAPPA)
    clean_weights, clean_metrics = compute_cis_weights(old, rollout, mask, LAM, KAPPA)
    assert torch.equal(weights, clean_weights)
    assert torch.equal(weights[pad], torch.zeros(pad.sum()))
    assert metrics == clean_metrics


def test_weights_are_detached():
    old, rollout = (t.requires_grad_() for t in random_logps((4, 8)))
    assert not cis(old, rollout).requires_grad


def test_policy_loss_keeps_ppo_clipping_and_scales_token_gradients_by_the_weights():
    actor = omega_conf_to_dataclass(compose_config(*CIS).actor_rollout_ref.actor)
    old, rollout = random_logps((8, 64))
    mask = padding_mask(old.shape)
    advantages = torch.randn(old.shape, generator=torch.Generator().manual_seed(1))
    log_prob = old + 0.3 * torch.randn(old.shape, generator=torch.Generator().manual_seed(2))
    weights = cis(old, rollout, mask)

    def loss_and_grad(rollout_is_weights):
        current = log_prob.clone().requires_grad_()
        loss, metrics = compute_policy_loss_vanilla(
            old, current, advantages, mask, "token-mean", actor, rollout_is_weights=rollout_is_weights
        )
        return torch.autograd.grad(loss, current)[0], metrics

    grad, metrics = loss_and_grad(weights)
    plain_grad, plain_metrics = loss_and_grad(None)
    assert metrics == plain_metrics and 0 < metrics["actor/pg_clipfrac"] < 1
    torch.testing.assert_close(grad, weights * plain_grad)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bypass_mode=True),
        dict(bypass_mode=True, loss_type="reinforce", score_centering=True),
        dict(rollout_is=None),
        dict(rollout_is="sequence"),
        dict(rollout_is_batch_normalize=True),
        dict(rollout_is_threshold="0.5_5.0"),
        dict(rollout_is_cis_lambda=-0.1),
        dict(rollout_is_cis_lambda=math.inf),
        dict(rollout_is_cis_lambda=math.nan),
        dict(rollout_is_cis_kappa=-1e-3),
        dict(rollout_is_cis_kappa=1.5),
    ],
)
def test_invalid_cis_config_is_rejected(kwargs):
    with pytest.raises(ValueError, match="CIS"):
        RolloutCorrectionConfig(**{"rollout_is": "token", "rollout_is_cis_lambda": LAM, **kwargs})


def test_preset_and_paper_ablations_are_valid():
    preset = RolloutCorrectionConfig.decoupled_token_cis()
    assert (preset.rollout_is, preset.rollout_is_cis_lambda, preset.rollout_is_cis_kappa) == ("token", LAM, KAPPA)
    assert not preset.bypass_mode and not preset.rollout_is_batch_normalize
    for lam, kappa in [(0.0, KAPPA), (LAM, 0.0), (LAM, 1.0)]:
        RolloutCorrectionConfig(rollout_is="token", rollout_is_cis_lambda=lam, rollout_is_cis_kappa=kappa)


@pytest.mark.parametrize(
    "overrides, error",
    [
        ((), None),
        (CIS, None),
        ((*CIS, "algorithm.rollout_correction.bypass_mode=true"), "bypass_mode=False"),
        ((*CIS, "algorithm.rollout_correction.rollout_is_batch_normalize=true"), "rollout_is_batch_normalize"),
        (
            (
                "algorithm.rollout_correction.rollout_is=sequence",
                f"algorithm.rollout_correction.rollout_is_cis_lambda={LAM}",
            ),
            "rollout_is='token'",
        ),
        ((*CIS, "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode"), "loss_mode"),
        (
            (
                "+actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=token",
                f"+actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_cis_lambda={LAM}",
            ),
            "set algorithm.rollout_correction.rollout_is_cis_lambda",
        ),
    ],
)
def test_validate_config_on_composed_hydra_config(overrides, error):
    config = compose_config(*overrides)
    if error is None:
        validate_config(config, use_reference_policy=False, use_critic=False)
    else:
        with pytest.raises(ValueError, match=error):
            validate_config(config, use_reference_policy=False, use_critic=False)


def test_default_config_keeps_cis_disabled():
    rollout_correction = compose_config().algorithm.rollout_correction
    assert rollout_correction.rollout_is is None
    assert rollout_correction.rollout_is_cis_lambda is None
    assert rollout_correction.rollout_is_cis_kappa == KAPPA


# Token TIS, token IcePop and sequence TIS of the importance ratio k, derived independently of the helper.
LEGACY_WEIGHTS = {
    ("token", 2.0): lambda k, mask: k.clamp(max=2.0) * mask,
    ("token", "0.5_5.0"): lambda k, mask: torch.where((k >= 0.5) & (k <= 5.0), k, 0.0) * mask,
    ("sequence", 2.0): lambda k, mask: torch.where(mask.bool(), k, 1.0).prod(-1, keepdim=True).clamp(max=2.0) * mask,
}


@pytest.mark.parametrize("rollout_is, threshold", list(LEGACY_WEIGHTS))
def test_legacy_weights_are_unchanged_for_configs_without_cis_keys(rollout_is, threshold):
    old, rollout = random_logps((8, 16))
    mask = padding_mask(old.shape)
    batch = DataProto.from_dict(tensors={"old_log_probs": old, "rollout_log_probs": rollout, "response_mask": mask})
    batch, metrics = compute_rollout_correction_and_add_to_batch(
        batch, {"rollout_is": rollout_is, "rollout_is_threshold": threshold}
    )
    expected = LEGACY_WEIGHTS[rollout_is, threshold]((old - rollout).exp(), mask)
    torch.testing.assert_close(batch.batch["rollout_is_weights"], expected)
    assert not any("cis" in key for key in metrics)


def test_decoupled_batch_correction_with_composed_config():
    """The call the legacy and separation trainers make once old_log_probs are recomputed."""
    old, rollout = random_logps((8, 64))
    mask = padding_mask(old.shape)
    batch = DataProto.from_dict(tensors={"old_log_probs": old, "rollout_log_probs": rollout, "response_mask": mask})
    batch, metrics = compute_rollout_correction_and_add_to_batch(
        batch, compose_config(*CIS).algorithm.rollout_correction
    )
    expected = torch.where(mask.bool(), reference_cis_weight(old, rollout), 0.0)
    torch.testing.assert_close(batch.batch["rollout_is_weights"], expected)
    assert torch.equal(batch.batch["response_mask"], mask)
    assert 0 < metrics["rollout_corr/rollout_is_cis_truncated_fraction"] < 1


def test_bypass_paths_reject_cis_instead_of_substituting_rollout_log_probs():
    old, rollout = random_logps((2, 4))
    rollout_correction = OmegaConf.create({"rollout_is": "token", "rollout_is_cis_lambda": LAM, "bypass_mode": True})
    batch = DataProto.from_dict(tensors={"rollout_log_probs": rollout, "response_mask": torch.ones_like(rollout)})
    with pytest.raises(ValueError, match="bypass_mode=False"):
        apply_bypass_mode(batch, rollout_correction, OmegaConf.create({}))
    assert "old_log_probs" not in batch.batch

    actor = SimpleNamespace(policy_loss={"rollout_correction": rollout_correction})
    with pytest.raises(ValueError, match="bypass_mode"):
        compute_policy_loss_bypass_mode(rollout, old, torch.ones_like(old), torch.ones_like(old), config=actor)


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ((), None),
        (("algorithm.rollout_correction.rollout_is=token",), lambda old, rollout: (old - rollout).exp().clamp(max=2.0)),
        (CIS, reference_cis_weight),
    ],
)
def test_v1_trainer_writes_weights_from_recomputed_old_log_probs(overrides, expected):
    lengths = [3, 7, 1, 5]
    old, rollout = random_logps((len(lengths), max(lengths)))
    store = {
        "uid": torch.arange(len(lengths)),
        "response_mask": nested([torch.ones(n, dtype=torch.long) for n in lengths]),
        "rm_scores": nested([torch.zeros(n) for n in lengths]),
        "old_log_probs": nested([row[:n] for row, n in zip(old, lengths, strict=True)]),
        "rollout_log_probs": nested([row[:n] for row, n in zip(rollout, lengths, strict=True)]),
    }
    meta = MagicMock(keys=[str(i) for i in range(len(lengths))], partition_id="train")
    meta.__len__.return_value = len(lengths)

    def kv_batch_get(keys, partition_id, select_fields):
        return TensorDict({field: store[field] for field in select_fields if field in store}, batch_size=len(keys))

    def kv_batch_put(keys, partition_id, fields):
        store.update(fields.items())
        return meta

    # The advantage estimator is irrelevant here, and its global registry is cleared by other CPU tests.
    def zero_advantages(data, **kwargs):
        data.batch["advantages"] = data.batch["returns"] = torch.zeros_like(data.batch["response_mask"]).float()
        return data

    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = compose_config(*overrides)
    with (
        patch("verl.trainer.ppo.v1.trainer_base.tq") as tq,
        patch("verl.trainer.ppo.v1.trainer_base.compute_advantage_for_multi_trajectories", zero_advantages),
    ):
        tq.kv_batch_get.side_effect = kv_batch_get
        tq.kv_batch_put.side_effect = kv_batch_put
        trainer._compute_advantage(meta, metrics={})

    if expected is None:
        assert "rollout_is_weights" not in store
        return
    for row, (n, written) in enumerate(zip(lengths, store["rollout_is_weights"].unbind(), strict=True)):
        torch.testing.assert_close(written, expected(old[row, :n], rollout[row, :n]))
