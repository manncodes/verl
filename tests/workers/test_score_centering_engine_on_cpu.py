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
"""Exercises FSDPEngineWithLMHead.prepare_model_outputs' score-centering hook
(calculate_score_centering=True) directly on CPU: the dense, non-fused,
non-Ulysses-SP branch that gathers the trainer's head log-probs from full-vocab
logits and narrows them down to each sample's true (prompt+response) length,
mirroring the existing `calculate_sum_pi_squared` plumbing. Also covers the
guards that reject the not-yet-implemented use_remove_padding/use_fused_kernels
combinations.

See `verl.utils.torch_functional.score_centering_correction_from_logits` for the
underlying math (arXiv:2609.20807), tested in isolation in
`tests/utils/test_score_centering_on_cpu.py`; this file only tests the engine's
reshaping/narrowing wiring for it.
"""

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import score_centering_correction_from_logits
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

_VOCAB_SIZE = 8
_TOPK = 3


def _make_engine_stub():
    """Bypass FSDPEngineWithLMHead.__init__; set only what prepare_model_outputs
    touches on this path (no SP, no fused kernels, no entropy checkpointing)."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False

    class _EngineCfg:
        entropy_checkpointing = False

    eng.engine_config = _EngineCfg()
    return eng


def _nested_from_lengths(values: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def _build_micro_batch(seq_lengths: list[int], *, calculate_score_centering: bool = True) -> tuple[TensorDict, dict]:
    total_nnz = sum(seq_lengths)
    flat_input_ids = torch.randint(0, _VOCAB_SIZE, (total_nnz,))
    input_ids_nested = _nested_from_lengths(flat_input_ids, seq_lengths)

    bsz, max_seqlen = len(seq_lengths), max(seq_lengths)
    output_args = {
        "input_ids_rmpad_rolled": torch.randint(0, _VOCAB_SIZE, (total_nnz,)),
        "temperature": torch.ones(bsz),
    }

    micro_batch = TensorDict({"input_ids": input_ids_nested}, batch_size=[])
    if calculate_score_centering:
        topk_ids = _nested_from_lengths(torch.randint(0, _VOCAB_SIZE, (total_nnz, _TOPK)), seq_lengths)
        topk_logprobs = _nested_from_lengths(-torch.rand(total_nnz, _TOPK), seq_lengths)
        micro_batch.set("rollout_topk_ids", topk_ids)
        micro_batch.set("rollout_topk_logprobs", topk_logprobs)
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=False,
        distillation_only=True,  # skips log_probs so logprobs_from_logits need not be patched
        calculate_score_centering=calculate_score_centering,
        max_response_length=max_seqlen,
    )
    return micro_batch, output_args


def test_score_centering_correction_matches_direct_computation_after_narrowing():
    seq_lengths = [3, 2]
    micro_batch, output_args = _build_micro_batch(seq_lengths)
    bsz, max_seqlen = len(seq_lengths), max(seq_lengths)

    class _Output:
        pass

    output = _Output()
    output.logits = torch.randn(bsz, max_seqlen, _VOCAB_SIZE)

    model_output = FSDPEngineWithLMHead.prepare_model_outputs(
        _make_engine_stub(),
        output=output,
        output_args=output_args,
        micro_batch=micro_batch,
        logits_processor_func=None,
    )

    assert "score_centering_correction" in model_output
    actual = model_output["score_centering_correction"]
    assert actual.is_nested

    # Ground truth: gather at each sample's true length directly from the dense
    # logits/topk tensors, bypassing the engine's narrow/repack machinery entirely.
    topk_ids_dense = torch.nested.to_padded_tensor(
        micro_batch["rollout_topk_ids"], padding=0, output_size=(bsz, max_seqlen, _TOPK)
    )
    topk_logprobs_dense = torch.nested.to_padded_tensor(
        micro_batch["rollout_topk_logprobs"], padding=0.0, output_size=(bsz, max_seqlen, _TOPK)
    )
    full_correction = score_centering_correction_from_logits(output.logits, topk_ids_dense, topk_logprobs_dense)
    expected = torch.cat([full_correction[i, :length] for i, length in enumerate(seq_lengths)])

    torch.testing.assert_close(actual.values(), expected)


def test_score_centering_omitted_when_disabled():
    seq_lengths = [3, 2]
    micro_batch, output_args = _build_micro_batch(seq_lengths, calculate_score_centering=False)
    bsz, max_seqlen = len(seq_lengths), max(seq_lengths)

    class _Output:
        pass

    output = _Output()
    output.logits = torch.randn(bsz, max_seqlen, _VOCAB_SIZE)

    model_output = FSDPEngineWithLMHead.prepare_model_outputs(
        _make_engine_stub(),
        output=output,
        output_args=output_args,
        micro_batch=micro_batch,
        logits_processor_func=None,
    )

    assert "score_centering_correction" not in model_output


@pytest.mark.parametrize(
    ("use_remove_padding", "use_fused_kernels"),
    [(True, False), (False, True)],
)
def test_score_centering_rejects_unsupported_engine_configs(use_remove_padding, use_fused_kernels):
    micro_batch, output_args = _build_micro_batch([3, 2])
    tu.assign_non_tensor(micro_batch, use_remove_padding=use_remove_padding, use_fused_kernels=use_fused_kernels)

    class _Output:
        pass

    with pytest.raises(NotImplementedError, match="calculate_score_centering"):
        FSDPEngineWithLMHead.prepare_model_outputs(
            _make_engine_stub(),
            output=_Output(),
            output_args=output_args,
            micro_batch=micro_batch,
            logits_processor_func=None,
        )
