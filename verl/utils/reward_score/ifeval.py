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
IFEval (Instruction Following Evaluation) reward function.

Uses the IFEvalG library from open-instruct for instruction verification.

Reference: Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models."
arXiv preprint arXiv:2311.07911 (2023).
"""

import re
from typing import Any

from .ifeval_util import INSTRUCTION_DICT


def compute_score(
    solution_str: str,
    ground_truth: dict | list,
    extra_info: dict[str, Any] | None = None,
    strict: bool = True,
    use_ifeval_library: bool = True,
) -> dict[str, Any]:
    """Compute IFEval reward score for instruction following.

    Args:
        solution_str: The model's generated response to evaluate
        ground_truth: Dictionary or list containing instruction details:
            - If dict: should contain 'instruction_id_list' and 'kwargs' for each instruction
            - If list: list of instruction verification functions/rules
        extra_info: Optional dict containing additional context like 'prompt', 'instruction_id_list', etc.
        strict: If True, use strict verification. If False, apply preprocessing before verification
        use_ifeval_library: If True, attempt to use the official ifeval library if installed.
                           If False or if library is not available, use IFEvalG from ifeval_util.

    Returns:
        Dictionary containing:
            - score: Overall instruction following rate (0.0 to 1.0)
            - prompt_strict_acc: Prompt-level strict accuracy (all instructions followed)
            - inst_strict_acc: Instruction-level strict accuracy (average across instructions)
            - num_instructions: Total number of instructions
            - num_followed: Number of instructions successfully followed
    """
    # Try to use the official ifeval library if available and requested
    if use_ifeval_library:
        try:
            from instruction_following_eval import instructions_registry

            return _compute_score_with_library(solution_str, ground_truth, extra_info, strict)
        except ImportError:
            # Fall back to IFEvalG from ifeval_util
            pass

    # Use IFEvalG from ifeval_util (copied from open-instruct)
    return _compute_score_with_ifeval_g(solution_str, ground_truth, extra_info, strict)


def _compute_score_with_library(
    solution_str: str, ground_truth: dict | list, extra_info: dict[str, Any] | None, strict: bool
) -> dict[str, Any]:
    """Compute score using the official ifeval library."""
    from instruction_following_eval import instructions_registry

    # IMPORTANT: Remove thinking section first (before any other processing)
    # This ensures that reasoning model think tags don't affect instruction checks
    response = _remove_thinking_section(solution_str)

    # Preprocess response for loose accuracy
    if not strict:
        response = _preprocess_response_loose(response)

    # Extract instruction information
    instruction_list, kwargs_list = _extract_instruction_data(ground_truth, extra_info)

    if not instruction_list:
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0}

    # Verify each instruction
    num_instructions = len(instruction_list)
    num_followed = 0

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
            if instruction_cls is None:
                continue

            instruction = instruction_cls(instruction_id)
            is_followed = instruction.check_following(response, **kwargs)
            if is_followed:
                num_followed += 1
        except Exception:
            continue

    # Calculate metrics
    inst_level_acc = num_followed / num_instructions if num_instructions > 0 else 0.0
    prompt_level_acc = 1.0 if num_followed == num_instructions else 0.0

    return {
        "score": inst_level_acc,
        "prompt_strict_acc": prompt_level_acc,
        "inst_strict_acc": inst_level_acc,
        "num_instructions": num_instructions,
        "num_followed": num_followed,
    }


def _compute_score_with_ifeval_g(
    solution_str: str, ground_truth: dict | list, extra_info: dict[str, Any] | None, strict: bool
) -> dict[str, Any]:
    """Compute score using IFEvalG from ifeval_util (open-instruct implementation)."""
    # IMPORTANT: Remove thinking section first (before any other processing)
    # This ensures that reasoning model think tags don't affect instruction checks
    response = _remove_thinking_section(solution_str)

    # Preprocess response for loose accuracy
    if not strict:
        response = _preprocess_response_loose(response)

    # Extract instruction information
    instruction_list, kwargs_list = _extract_instruction_data(ground_truth, extra_info)

    if not instruction_list:
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0}

    # Verify each instruction using IFEvalG
    num_instructions = len(instruction_list)
    num_followed = 0

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            instruction_cls = INSTRUCTION_DICT.get(instruction_id)
            if instruction_cls is None:
                continue

            # Create instruction instance
            instruction = instruction_cls(instruction_id)

            # Clean kwargs - remove None values
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

            # Build description (required by IFEvalG)
            if hasattr(instruction, "build_description"):
                instruction.build_description(**clean_kwargs)

            # Check if instruction is followed
            is_followed = instruction.check_following(response)
            if is_followed:
                num_followed += 1
        except Exception as e:
            # If verification fails, count as not followed
            continue

    # Calculate metrics
    inst_level_acc = num_followed / num_instructions if num_instructions > 0 else 0.0
    prompt_level_acc = 1.0 if num_followed == num_instructions else 0.0

    return {
        "score": inst_level_acc,
        "prompt_strict_acc": prompt_level_acc,
        "inst_strict_acc": inst_level_acc,
        "num_instructions": num_instructions,
        "num_followed": num_followed,
    }


def _remove_thinking_section(prediction: str) -> str:
    """Remove thinking section from prediction.

    This is critical for evaluating reasoning models that output their thinking process
    before the final answer. The thinking section should not be considered when checking
    instructions like word count, forbidden words, commas, etc.

    Based on open-instruct implementation.

    Args:
        prediction: The raw model output which may contain think tags

    Returns:
        The prediction with thinking section, answer tags, and assistant prefix removed

    Examples:
        >>> _remove_thinking_section("<think>Reasoning here</think>Final answer")
        'Final answer'
        >>> _remove_thinking_section("<|assistant|><think>Think</think><answer>Answer</answer>")
        'Answer'
    """
    # Remove assistant prefix (common in chat models)
    prediction = prediction.replace("<|assistant|>", "").strip()

    # Remove thinking section (everything before and including </think>)
    if "</think>" in prediction:
        prediction = prediction.split("</think>")[-1]

    # Remove answer tags (some models wrap final answer in these)
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")

    return prediction.strip()


def _extract_instruction_data(ground_truth: dict | list, extra_info: dict[str, Any] | None):
    """Extract instruction_id_list and kwargs from ground_truth or extra_info."""
    instruction_list = []
    kwargs_list = []

    # Try ground_truth first
    if isinstance(ground_truth, dict):
        instruction_list = ground_truth.get("instruction_id_list", [])
        kwargs_list = ground_truth.get("kwargs", [])

    # Fall back to extra_info if needed
    if not instruction_list and extra_info is not None:
        instruction_list = extra_info.get("instruction_id_list", [])
        kwargs_list = extra_info.get("kwargs", [])

    # Ensure kwargs_list matches instruction_list length
    if len(kwargs_list) < len(instruction_list):
        kwargs_list.extend([{}] * (len(instruction_list) - len(kwargs_list)))

    return instruction_list, kwargs_list


def _preprocess_response_loose(response: str) -> str:
    """Preprocess response for loose accuracy evaluation.

    Applies transformations:
    - Remove first line (to skip intros like "Sure, here it is:")
    - Remove last line (to skip outros)
    - Remove markdown formatting
    """
    lines = response.split("\n")

    # Remove first and last lines if response has more than 2 lines
    if len(lines) > 2:
        lines = lines[1:-1]

    response = "\n".join(lines)

    # Remove common markdown formatting
    response = re.sub(r"\*\*(.+?)\*\*", r"\1", response)  # Bold
    response = re.sub(r"\*(.+?)\*", r"\1", response)  # Italic
    response = re.sub(r"__(.+?)__", r"\1", response)  # Bold
    response = re.sub(r"_(.+?)_", r"\1", response)  # Italic

    return response
