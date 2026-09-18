# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Unit tests for score centering (arXiv:2609.20807): the control variate in
`verl.utils.torch_functional.score_centering_correction[_from_logits]` and the
threshold-driven `weight_fn` builder in `verl.trainer.ppo.rollout_corr_helper`.
"""

import pytest
import torch

from verl.trainer.ppo.rollout_corr_helper import score_centering_weight_fn
from verl.utils.torch_functional import score_centering_correction, score_centering_correction_from_logits
from verl.workers.utils.losses import score_centering_loss


def _random_head_logp(*shape: int, generator: torch.Generator) -> torch.Tensor:
    """A (..., k) head with the k+1'th "tail" bucket implicit, all log-probs < 0."""
    logits = torch.randn(*shape, generator=generator, dtype=torch.float64)
    # Reserve probability mass for the modeled tail so log-softmax over k+1 buckets,
    # then dropping the tail bucket, leaves a valid (non-exhaustive) head.
    logits = torch.cat([logits, torch.full(shape[:-1] + (1,), 3.0, dtype=torch.float64)], dim=-1)
    return torch.log_softmax(logits, dim=-1)[..., :-1]


@pytest.mark.parametrize("weight_fn", [torch.ones_like, lambda r: r.clamp(max=2.0)])
def test_score_centering_correction_vanishes_on_policy(weight_fn):
    """The control variate must be identically zero when trainer == rollout."""
    gen = torch.Generator().manual_seed(0)
    head_logp = _random_head_logp(4, 7, 5, generator=gen).requires_grad_()

    correction = score_centering_correction(head_logp, head_logp.detach(), weight_fn=weight_fn)

    assert correction.shape == (4, 7)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=1e-10, rtol=0)

    # The on-policy residual is the zero vector, so the *gradient* also vanishes there,
    # not just the value at this particular sample.
    correction.sum().backward()
    torch.testing.assert_close(head_logp.grad, torch.zeros_like(head_logp.grad), atol=1e-10, rtol=0)


def test_score_centering_correction_nonzero_and_finite_off_policy():
    gen = torch.Generator().manual_seed(1)
    train_logp = _random_head_logp(3, 6, generator=gen)
    rollout_logp = _random_head_logp(3, 6, generator=torch.Generator().manual_seed(2))

    correction = score_centering_correction(train_logp, rollout_logp)

    assert torch.isfinite(correction).all()
    assert (correction != 0).any()


def test_score_centering_correction_from_logits_matches_manual_gather():
    gen = torch.Generator().manual_seed(3)
    batch, seq, vocab, k = 2, 5, 32, 8
    logits = torch.randn(batch, seq, vocab, generator=gen)
    topk_ids = torch.randint(0, vocab, (batch, seq, k), generator=gen)
    rollout_logp = _random_head_logp(batch, seq, k, generator=torch.Generator().manual_seed(4))

    actual = score_centering_correction_from_logits(logits, topk_ids, rollout_logp)

    expected_head_logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, topk_ids)
    expected = score_centering_correction(expected_head_logp, rollout_logp)
    torch.testing.assert_close(actual, expected)


def test_score_centering_loss_matches_direct_formula_and_preserves_gradient():
    correction = torch.tensor([[1.0, 2.0, 0.0]], requires_grad=True)
    advantages = torch.tensor([[3.0, -1.0, 5.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])

    loss = score_centering_loss(correction, advantages, response_mask, "token-mean", {})

    expected = (advantages * correction * response_mask).sum() / response_mask.sum()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    # advantages are detached: d(loss)/d(correction) is just advantages * mask / count.
    torch.testing.assert_close(correction.grad, advantages * response_mask / response_mask.sum())


@pytest.mark.parametrize(
    ("threshold", "ratio", "expected"),
    [
        (None, torch.tensor([0.1, 1.0, 10.0]), torch.tensor([1.0, 1.0, 1.0])),  # no IS reweighting
        (2.0, torch.tensor([0.1, 1.0, 10.0]), torch.tensor([0.1, 1.0, 2.0])),  # TIS: clamp upper
        ("0.5_5.0", torch.tensor([0.1, 1.0, 10.0]), torch.tensor([0.0, 1.0, 0.0])),  # IcePop/MIS band
    ],
)
def test_score_centering_weight_fn_matches_rollout_is_threshold_semantics(threshold, ratio, expected):
    weight_fn = score_centering_weight_fn(threshold)
    torch.testing.assert_close(weight_fn(ratio), expected)
