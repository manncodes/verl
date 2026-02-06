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
"""
Core algorithms for TRACE (Truncated Reasoning AUC Evaluation).

This module implements the core computation of TRACE scores by:
1. Truncating chain-of-thought (CoT) responses at various fractions
2. Generating completions from truncated prefixes
3. Computing expected rewards at each truncation level
4. Computing the AUC (TRACE score) across truncation fractions

Reference:
    Wang et al. (2025). "Is It Thinking or Cheating? Detecting Implicit
    Reward Hacking by Measuring Reasoning Effort." arXiv:2510.01367
"""

from typing import Callable, Optional

import numpy as np
import torch

__all__ = [
    "truncate_response",
    "compute_expected_reward_at_truncation",
    "compute_trace_auc",
    "compute_trace_scores_batch",
]


def truncate_response(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Truncate a response at a given fraction of its valid length.

    Args:
        response_ids: Token IDs for the response, shape (response_length,).
        response_mask: Binary mask indicating valid tokens, shape (response_length,).
        fraction: Fraction of valid tokens to keep, in (0, 1].

    Returns:
        truncated_ids: Token IDs truncated at the given fraction.
        truncated_mask: Updated mask for the truncated sequence.
        truncation_length: Number of valid tokens kept.
    """
    valid_length = int(response_mask.sum().item())
    truncation_length = max(1, int(valid_length * fraction))

    truncated_ids = response_ids.clone()
    truncated_mask = response_mask.clone()

    # Zero out tokens beyond the truncation point
    truncated_ids[truncation_length:] = 0
    truncated_mask[truncation_length:] = 0

    return truncated_ids, truncated_mask, truncation_length


def truncate_response_batch(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncate a batch of responses at a given fraction of their valid lengths.

    Args:
        response_ids: Token IDs, shape (batch_size, response_length).
        response_mask: Binary mask, shape (batch_size, response_length).
        fraction: Fraction of valid tokens to keep, in (0, 1].

    Returns:
        truncated_ids: Truncated token IDs, shape (batch_size, response_length).
        truncated_mask: Updated masks, shape (batch_size, response_length).
        truncation_lengths: Per-sample truncation lengths, shape (batch_size,).
    """
    valid_lengths = response_mask.sum(dim=-1)  # (batch_size,)
    truncation_lengths = torch.clamp((valid_lengths * fraction).long(), min=1)

    truncated_ids = response_ids.clone()
    truncated_mask = response_mask.clone()

    # Create position indices: (batch_size, response_length)
    positions = torch.arange(response_ids.shape[-1], device=response_ids.device).unsqueeze(0)
    # Mask out positions beyond truncation length
    beyond_trunc = positions >= truncation_lengths.unsqueeze(-1)
    truncated_ids[beyond_trunc] = 0
    truncated_mask[beyond_trunc] = 0

    return truncated_ids, truncated_mask, truncation_lengths


def compute_expected_reward_at_truncation(
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Compute the expected reward from multiple completions at a truncation point.

    Given multiple sampled completions from a truncated prefix, this estimates
    E[R̂] by averaging the rewards.

    Args:
        rewards: Reward scores for each completion, shape (num_completions,)
            or (batch_size, num_completions).

    Returns:
        Expected reward, shape () or (batch_size,).
    """
    return rewards.float().mean(dim=-1)


def compute_trace_auc(
    expected_rewards: torch.Tensor,
    truncation_fractions: list[float],
) -> torch.Tensor:
    """Compute the TRACE score (AUC) from expected rewards at each truncation level.

    The TRACE score is the area under the curve of E[R̂] vs truncation fraction,
    computed using the trapezoidal rule. A higher AUC indicates the model can
    achieve high reward with less reasoning -- a signal of reward hacking.

    The AUC is normalized to [0, 1] by dividing by the maximum possible area
    (which is 1.0 when the x-axis goes from 0 to 1 with max reward of 1).

    Args:
        expected_rewards: Expected rewards at each truncation fraction.
            Shape (num_fractions,) for a single sample, or
            (batch_size, num_fractions) for a batch.
        truncation_fractions: List of truncation fractions corresponding to
            the expected_rewards. Must be sorted in ascending order.

    Returns:
        TRACE score(s) (AUC value), shape () or (batch_size,).
    """
    fractions = torch.tensor(truncation_fractions, dtype=torch.float32, device=expected_rewards.device)

    if expected_rewards.dim() == 1:
        # Single sample: use numpy-style trapezoidal rule
        auc = torch.trapezoid(expected_rewards, fractions)
        # Normalize by the x-range so AUC is in [0, max_reward]
        x_range = fractions[-1] - fractions[0]
        if x_range > 0:
            auc = auc / x_range
        return auc
    else:
        # Batch: (batch_size, num_fractions)
        auc = torch.trapezoid(expected_rewards, fractions, dim=-1)
        x_range = fractions[-1] - fractions[0]
        if x_range > 0:
            auc = auc / x_range
        return auc


def compute_trace_scores_batch(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    truncation_fractions: list[float],
    generate_fn: Callable,
    reward_fn: Callable,
    tokenizer,
    num_completions: int = 4,
    temperature: float = 0.7,
    max_completion_tokens: int = 128,
    answer_prompt: str = "\n\nTherefore, the answer is",
    ground_truths: Optional[list] = None,
    data_sources: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """Compute TRACE scores for a batch of samples.

    For each sample, this:
    1. Truncates the CoT at each fraction
    2. Appends an answer prompt to the truncated prefix
    3. Generates multiple completions from the truncated prefix
    4. Computes the reward for each completion
    5. Averages rewards to get E[R̂] at each truncation level
    6. Computes the AUC (TRACE score)

    Args:
        prompt_ids: Prompt token IDs, shape (batch_size, prompt_length).
        response_ids: Response token IDs, shape (batch_size, response_length).
        response_mask: Response masks, shape (batch_size, response_length).
        attention_mask: Full attention mask, shape (batch_size, seq_length).
        truncation_fractions: List of truncation fractions.
        generate_fn: Function to generate completions. Signature:
            generate_fn(input_ids, attention_mask, num_completions, temperature,
                       max_new_tokens) -> generated_ids
        reward_fn: Function to compute rewards. Signature:
            reward_fn(response_str, ground_truth, data_source) -> float
        tokenizer: Tokenizer for encoding/decoding.
        num_completions: Number of completions per truncation point.
        temperature: Sampling temperature.
        max_completion_tokens: Max new tokens for completion.
        answer_prompt: Prompt appended after truncation to force answer.
        ground_truths: Ground truth answers for reward computation.
        data_sources: Data source identifiers for reward computation.

    Returns:
        Dictionary with:
            "trace_scores": TRACE AUC scores, shape (batch_size,).
            "expected_rewards": E[R̂] at each truncation, shape (batch_size, num_fractions).
            "truncation_fractions": The truncation fractions used.
    """
    batch_size = response_ids.shape[0]
    num_fractions = len(truncation_fractions)
    device = response_ids.device

    # Store expected rewards at each truncation fraction
    all_expected_rewards = torch.zeros(batch_size, num_fractions, device=device)
    answer_prompt_ids = tokenizer.encode(answer_prompt, add_special_tokens=False)
    answer_prompt_tensor = torch.tensor(answer_prompt_ids, device=device)

    for frac_idx, fraction in enumerate(truncation_fractions):
        # Truncate the responses
        trunc_ids, trunc_mask, trunc_lengths = truncate_response_batch(
            response_ids, response_mask, fraction
        )

        # Build input for generation: prompt + truncated response + answer prompt
        # For each sample, concatenate and generate completions
        fraction_rewards = torch.zeros(batch_size, num_completions, device=device)

        for sample_idx in range(batch_size):
            trunc_len = trunc_lengths[sample_idx].item()
            sample_prompt = prompt_ids[sample_idx]

            # Get valid prompt tokens
            prompt_length = sample_prompt.shape[0]
            valid_prompt_mask = attention_mask[sample_idx][:prompt_length]
            valid_prompt_start = (valid_prompt_mask == 0).sum().item()

            # Build input: valid_prompt + truncated_response + answer_prompt
            valid_prompt = sample_prompt[valid_prompt_start:]
            truncated_response = trunc_ids[sample_idx][:trunc_len]
            input_ids = torch.cat([valid_prompt, truncated_response, answer_prompt_tensor])
            input_mask = torch.ones(input_ids.shape[0], device=device)

            # Generate multiple completions
            input_ids_batched = input_ids.unsqueeze(0).expand(num_completions, -1)
            input_mask_batched = input_mask.unsqueeze(0).expand(num_completions, -1)

            with torch.no_grad():
                generated = generate_fn(
                    input_ids=input_ids_batched,
                    attention_mask=input_mask_batched,
                    num_return_sequences=1,
                    temperature=temperature,
                    max_new_tokens=max_completion_tokens,
                    do_sample=True,
                )

            # Score each completion
            for comp_idx in range(num_completions):
                if generated is not None and comp_idx < generated.shape[0]:
                    # Extract the generated completion (after the input)
                    full_output = generated[comp_idx]
                    completion = full_output[input_ids.shape[0]:]
                    completion_str = tokenizer.decode(completion, skip_special_tokens=True)

                    # Also decode the truncated response for context
                    trunc_response_str = tokenizer.decode(
                        truncated_response, skip_special_tokens=True
                    )
                    full_response = trunc_response_str + answer_prompt + completion_str

                    # Compute reward
                    gt = ground_truths[sample_idx] if ground_truths is not None else None
                    ds = data_sources[sample_idx] if data_sources is not None else "default"

                    try:
                        score = reward_fn(
                            data_source=ds,
                            solution_str=full_response,
                            ground_truth=gt,
                            extra_info={},
                        )
                        if isinstance(score, dict):
                            score = score.get("score", 0.0)
                        fraction_rewards[sample_idx, comp_idx] = float(score)
                    except Exception:
                        fraction_rewards[sample_idx, comp_idx] = 0.0

        # Compute E[R̂] at this truncation fraction
        all_expected_rewards[:, frac_idx] = compute_expected_reward_at_truncation(
            fraction_rewards
        )

    # Compute TRACE AUC scores
    trace_scores = compute_trace_auc(all_expected_rewards, truncation_fractions)

    return {
        "trace_scores": trace_scores,
        "expected_rewards": all_expected_rewards,
        "truncation_fractions": torch.tensor(truncation_fractions, device=device),
    }
