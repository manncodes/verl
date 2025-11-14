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
Batched IFEval (Instruction Following Evaluation) reward function.

Uses the IFEvalG library from open-instruct for instruction verification,
with batched processing for efficient LLM judge evaluation.

Reference: Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models."
arXiv preprint arXiv:2311.07911 (2023).
"""

import re
from typing import Any, Optional

from verl.utils.reward_score.ifeval_util import instructions_registry
from verl.utils.reward_score.ifeval_util.instructions_registry import INSTRUCTION_DICT
from verl.utils.reward_score.judge import StructuredJudge

# format score weightage for the final calculation of score
FORMAT_SCORE_WEIGHT = 0.1

# Module-level private variable for the judge instance
_judge_instance: Optional[StructuredJudge] = None


def _get_judge() -> StructuredJudge:
    """Get or create the singleton StructuredJudge instance.

    This function implements lazy initialization to avoid creating
    the judge until it's actually needed.

    The judge is configured with default settings:
    - max_workers: 128 (for concurrent batch evaluation)
    - timeout: 2.0 seconds per evaluation
    - max_retries: 0 (no retries by default)

    To customize these settings, modify the StructuredJudge initialization below.
    """
    global _judge_instance

    if _judge_instance is None:
        _judge_instance = StructuredJudge(
            base_url="http://qpn744-vllm-gptoss120b-svc.llm-pretraining.svc.cluster.local:8000/v1",
            api_key="dummy",
            max_workers=128,  # High concurrency for batch processing
            timeout=2.0,       # Timeout per evaluation
            max_retries=0      # Retries on timeout/failure
        )

    return _judge_instance


def format_chat_prompt(prompt_input: list | str) -> str:
    """Format chat prompt from various input formats."""
    if isinstance(prompt_input, str):
        return prompt_input

    if not isinstance(prompt_input, list):
        return str(prompt_input)

    formatted_parts = []
    for message in prompt_input:
        role = message.get('role', 'user')
        content = message.get('content', '')
        formatted_parts.append(f"<|{role}|>\n{content}")
    result = '\n'.join(formatted_parts)

    if prompt_input and prompt_input[-1].get('role', '').lower() != 'assistant':
        result += '\n<|assistant|>'

    return result


def remove_thinking_section(prediction: str) -> str:
    """Remove thinking tags and extract the final answer from prediction."""
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove evaluation
    prediction = prediction.split("</evaluation>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def format_reward(predict_str: str) -> float:
    """Compute format reward based on presence of required tags."""
    # Pattern ensures:
    # 1. <think> tag with content followed by </think>
    # 2. Any content (including nothing) in between
    # 3. <answer> tag with content followed by </answer>
    # 4. Potentially more content after
    pattern = re.compile(
        r"<think>.+?</think>.*?<answer>.+?</answer>",
        re.DOTALL
    )

    match_result = re.search(pattern, predict_str)
    # @bill suggestion -1.0 if no match ; prev 0.0
    return 1.0 if match_result else -1.0


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


def _compute_verifiable_reward(
    solution_str: str,
    ground_truth: dict | list,
    extra_info: dict[str, Any] | None,
    strict: bool
) -> tuple[float, float, float, int, int]:
    """Compute verifiable reward (V_i) using IFEval instruction verification.

    Returns:
        tuple: (inst_level_acc, prompt_level_acc, format_score, num_instructions, num_followed)
    """
    # Preprocess response for loose accuracy
    response = solution_str

    if not strict:
        response = _preprocess_response_loose(response)

    format_score = format_reward(solution_str)
    response = remove_thinking_section(response)

    # Extract instruction information
    instruction_list, kwargs_list = _extract_instruction_data(ground_truth, extra_info)

    if not instruction_list:
        return 0.0, 0.0, format_score, 0, 0

    # Verify each instruction using IFEvalG
    num_instructions = len(instruction_list)
    num_followed = 0

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            instruction_cls = INSTRUCTION_DICT.get(instruction_id)
            if instruction_cls is None:
                continue

            instruction = instruction_cls(instruction_id)
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if hasattr(instruction, "build_description"):
                instruction.build_description(**clean_kwargs)
            is_followed = instruction.check_following(response)
            if is_followed:
                num_followed += 1
        except Exception as e:
            # If verification fails, count as not followed
            continue

    # Calculate metrics
    inst_level_acc = num_followed / num_instructions if num_instructions > 0 else 0.0
    prompt_level_acc = 1.0 if num_followed == num_instructions else 0.0

    return inst_level_acc, prompt_level_acc, format_score, num_instructions, num_followed


def _batch_evaluate_with_judge(
    prompts: list[str],
    responses: list[str],
    indices_to_eval: list[int],
    show_progress: bool = False
) -> dict[int, float]:
    """Batch evaluate responses using LLM judge's built-in batching.

    Leverages StructuredJudge.evaluate_batch() which handles concurrent
    processing, timeout, retry, and progress tracking.

    Args:
        prompts: Full list of prompts
        responses: Full list of responses
        indices_to_eval: Indices of items that need judge evaluation (V_i > 0)
        show_progress: Whether to show progress bar during evaluation

    Returns:
        Dictionary mapping index to S_i (preference score)
    """
    if not indices_to_eval:
        return {}

    judge = _get_judge()

    # Extract only the prompts and responses that need evaluation
    prompts_to_eval = [format_chat_prompt(prompts[idx]) for idx in indices_to_eval]
    responses_to_eval = [responses[idx] for idx in indices_to_eval]

    # Use judge's built-in batch evaluation
    # This handles concurrency, timeout, retry, and progress tracking
    evaluations = judge.evaluate_batch(
        prompts=prompts_to_eval,
        responses=responses_to_eval,
        temperature=0.0,
        show_progress=show_progress,
        desc="Judge evaluation"
    )

    # Map results back to original indices
    results = {}
    for i, idx in enumerate(indices_to_eval):
        eval_result = evaluations[i]
        if eval_result is not None:
            # Compute S_i as normalized aggregate score
            S_i = eval_result.overall.aggregate_score / 10.0
            results[idx] = S_i
        else:
            # Failed evaluation - default to low score (conservative approach)
            print(f"Warning: Judge evaluation failed for index {idx}, using S_i=0.0")
            results[idx] = 0.0

    return results


def compute_score_batched(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict | list],
    extra_infos: list[dict[str, Any]],
    strict: bool = True,
    alpha_threshold: float = 0.5,
    show_progress: bool = False,
    **kwargs
) -> list[dict[str, Any]]:
    """Compute batched IFEval reward scores for instruction following.

    This is the main entry point for batched reward computation, optimized for
    use with BatchRewardManager. Uses StructuredJudge's built-in batching
    capabilities for efficient concurrent judge evaluations.

    Args:
        data_sources: List of data source identifiers
        solution_strs: List of model generated responses to evaluate
        ground_truths: List of dictionaries containing instruction details:
            - Each should contain 'instruction_id_list' and 'kwargs' for each instruction
        extra_infos: List of dicts containing additional context like 'prompt', etc.
        strict: If True, use strict verification. If False, apply preprocessing before verification
        alpha_threshold: Threshold for preference score (S_i > alpha gives bonus, else penalty)
        show_progress: Whether to show progress bar during judge evaluation
        **kwargs: Additional keyword arguments (unused, for compatibility)

    Returns:
        List of dictionaries, one per input, each containing:
            - score: Overall reward F_i (verifiable + preference components)
            - V_i: Verifiable reward component
            - S_i: Preference reward component (or None if not computed)
            - alpha_threshold: The threshold used
            - reward_case: Description of which reward formula was applied
            - prompt_strict_acc: Prompt-level strict accuracy
            - inst_strict_acc: Instruction-level strict accuracy
            - num_instructions: Total number of instructions
            - num_followed: Number of instructions successfully followed
            - format_score: Format compliance score

    Note:
        Judge concurrency is configured via StructuredJudge's max_workers parameter
        (default: 128). Timeout and retry settings are also configured at judge
        initialization.
    """
    batch_size = len(solution_strs)

    # Phase 1: Compute verifiable rewards (V_i) for all items
    verifiable_results = []
    indices_needing_judge = []

    for i in range(batch_size):
        inst_acc, prompt_acc, fmt_score, n_inst, n_followed = _compute_verifiable_reward(
            solution_strs[i],
            ground_truths[i],
            extra_infos[i],
            strict
        )

        V_i = inst_acc
        verifiable_results.append({
            'V_i': V_i,
            'inst_level_acc': inst_acc,
            'prompt_level_acc': prompt_acc,
            'format_score': fmt_score,
            'num_instructions': n_inst,
            'num_followed': n_followed
        })

        # Only need judge evaluation if V_i > 0
        if V_i > 0:
            indices_needing_judge.append(i)

    # Phase 2: Batch evaluate with LLM judge (only for items with V_i > 0)
    # Uses judge's built-in evaluate_batch() with its configured max_workers
    prompts = [extra_info.get('prompt', '') for extra_info in extra_infos]
    judge_results = _batch_evaluate_with_judge(
        prompts,
        solution_strs,
        indices_needing_judge,
        show_progress=show_progress
    )

    # Phase 3: Combine results and compute final rewards
    final_results = []

    for i in range(batch_size):
        result = verifiable_results[i].copy()
        V_i = result['V_i']
        S_i = None

        # Apply reward formula based on V_i and S_i
        if V_i > 0:
            # Get judge score
            S_i = judge_results.get(i, 0.0)  # Default to 0 if judge failed

            if S_i > alpha_threshold:
                # Good preference score: bonus reward
                F_i = V_i + 1
                reward_case = "V_i > 0 and S_i > α (bonus)"
            else:
                # Poor preference score: penalty
                F_i = V_i - 0.5
                reward_case = "V_i > 0 and S_i ≤ α (penalty)"
        else:
            # V_i ≤ 0: No verifiable reward or failed all constraints
            F_i = V_i
            reward_case = "V_i ≤ 0 (no bonus/penalty)"

        # Compile final result
        result.update({
            'score': F_i,
            'S_i': S_i,
            'alpha_threshold': alpha_threshold,
            'reward_case': reward_case,
            'prompt_strict_acc': result['prompt_level_acc'],
            'inst_strict_acc': result['inst_level_acc'],
        })

        final_results.append(result)

    return final_results
