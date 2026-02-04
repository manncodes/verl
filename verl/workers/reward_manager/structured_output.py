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
Structured output reward manager for JSON schema validation.

Supports:
- Fine-grained JSON schema validation rewards (SRL-style)
- CRANE-style hybrid decoding reward: separate scoring for reasoning
  and structured output sections
- Binary schema compliance scoring
- Configurable reward weight distribution

References:
- Schema RL (SRL): arxiv:2502.18878
- CRANE: arxiv:2502.09061
- RL-Struct: arxiv:2512.00319
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.structured_output import compute_score as structured_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


def _extract_reasoning_and_answer(
    text: str,
    reasoning_delimiter: str = "<answer>",
    reasoning_end_delimiter: str = "</answer>",
) -> tuple[str, str]:
    """Extract reasoning and answer sections from CRANE-style output.

    The model output is expected to have the format:
        <reasoning text> <answer> <structured output> </answer>

    Args:
        text: Full model output text.
        reasoning_delimiter: Start delimiter for structured section.
        reasoning_end_delimiter: End delimiter for structured section.

    Returns:
        Tuple of (reasoning_text, answer_text). If delimiters not found,
        reasoning_text is empty and answer_text is the full text.
    """
    start_idx = text.find(reasoning_delimiter)
    if start_idx == -1:
        return "", text

    reasoning = text[:start_idx].strip()
    rest = text[start_idx + len(reasoning_delimiter) :]

    end_idx = rest.find(reasoning_end_delimiter)
    if end_idx == -1:
        answer = rest.strip()
    else:
        answer = rest[:end_idx].strip()

    return reasoning, answer


@register("structured_output")
class StructuredOutputRewardManager(AbstractRewardManager):
    """Reward manager for structured output generation with fine-grained schema validation.

    This manager computes rewards based on:
    1. JSON parsability
    2. Schema validation (structural compliance)
    3. Field coverage (required fields present with correct types)
    4. Content correctness (if ground truth answers available)

    Supports CRANE-style hybrid output where the model first reasons freely
    then produces constrained structured output within delimiters.

    Args:
        tokenizer: Tokenizer for decoding token IDs.
        num_examine: Number of examples to print for debugging.
        compute_score: Optional custom scoring function.
        reward_fn_key: Key for data source in non_tensor_batch.
        reward_mode: "fine_grained" (default), "binary", or "crane".
        reasoning_delimiter: Start delimiter for CRANE-style answer section.
        reasoning_end_delimiter: End delimiter for CRANE-style answer section.
        reasoning_bonus: Bonus reward for having reasoning before answer (CRANE mode).
        reward_weights: Dict of weights for score components.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        reward_mode: str = "fine_grained",
        reasoning_delimiter: str = "<answer>",
        reasoning_end_delimiter: str = "</answer>",
        reasoning_bonus: float = 0.1,
        reward_weights: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or structured_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_mode = reward_mode
        self.reasoning_delimiter = reasoning_delimiter
        self.reasoning_end_delimiter = reasoning_end_delimiter
        self.reasoning_bonus = reasoning_bonus
        self.reward_weights = reward_weights

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute structured output rewards for a batch of data."""

        # Fast path: pre-computed rewards
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            # Handle CRANE-style output: extract reasoning and answer sections
            if self.reward_mode == "crane":
                reasoning_text, answer_text = _extract_reasoning_and_answer(
                    response_str,
                    self.reasoning_delimiter,
                    self.reasoning_end_delimiter,
                )
                eval_text = answer_text
                has_reasoning = len(reasoning_text) > 0
            else:
                eval_text = response_str
                reasoning_text = ""
                has_reasoning = False

            # Compute structured output score
            if self.reward_mode == "binary":
                from verl.utils.reward_score.structured_output import compute_score_binary

                score = compute_score_binary(eval_text, ground_truth, extra_info)
                reward = float(score)
                score_result = {"score": reward, "json_parse_score": reward, "schema_valid_score": reward}
            else:
                score_result = self.compute_score(
                    solution_str=eval_text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    reward_weights=self.reward_weights,
                )
                if isinstance(score_result, dict):
                    reward = score_result["score"]
                else:
                    reward = float(score_result)
                    score_result = {"score": reward}

            # CRANE bonus: reward for having reasoning before the answer
            if self.reward_mode == "crane" and has_reasoning:
                reward += self.reasoning_bonus

            # Store component scores in extra info for logging
            if isinstance(score_result, dict):
                for key, value in score_result.items():
                    reward_extra_info[key].append(value)

            reward_extra_info["has_reasoning"].append(1.0 if has_reasoning else 0.0)
            reward_extra_info["reward_mode"].append(self.reward_mode)

            reward_tensor[i, valid_response_length - 1] = reward

            # Debug printing
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str[:200])
                print("[response]", response_str[:500])
                if reasoning_text:
                    print("[reasoning]", reasoning_text[:200])
                print("[ground_truth]", str(ground_truth)[:200])
                if isinstance(score_result, dict):
                    for key, value in score_result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
