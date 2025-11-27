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
Reward scoring for allenai/Dolci-Think-RL dataset.

Routes to appropriate existing reward functions based on dataset_source:
- math: Uses math_dapo (verifiable)
- IF: Uses ifeval (verifiable)
- code: Uses LLM-as-a-judge via StructuredJudge
- chat: Uses basic verifiable matching

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Singleton instance for LLM judge
_judge_instance: Optional[Any] = None


def compute_score(
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute the score for a Dolci-Think-RL solution.

    Routes to appropriate existing reward function based on dataset_source:
    - math: math_dapo (verifiable)
    - IF: ifeval (verifiable)
    - code: LLM-as-a-judge
    - chat/default: basic string matching

    Args:
        solution_str: The model's generated solution.
        ground_truth: The expected ground truth answer.
        extra_info: Dict with 'dataset_source' to determine scoring method.
        **kwargs: Additional arguments passed to reward functions.

    Returns:
        Score as a float.
    """
    if solution_str is None:
        return 0.0

    # Handle ground_truth being a list (as in the dataset)
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else None

    if ground_truth is None:
        return 0.0

    # Determine task type from extra_info
    dataset_source = ""
    if extra_info and isinstance(extra_info, dict):
        dataset_source = extra_info.get("dataset_source", "").lower()

    # Route to appropriate existing reward function
    if dataset_source == "math":
        from . import math_dapo
        return math_dapo.compute_score(solution_str, ground_truth)

    elif dataset_source == "if":
        from . import ifeval
        return ifeval.compute_score(solution_str, ground_truth, extra_info)

    elif dataset_source == "code":
        # Use LLM-as-a-judge for code via StructuredJudge
        return _compute_code_score_llm_judge(solution_str, ground_truth, extra_info, **kwargs)

    else:
        # Default: try math first, then basic matching
        from . import math_dapo
        try:
            score = math_dapo.compute_score(solution_str, ground_truth)
            if score > 0:
                return score
        except Exception:
            pass

        # Fallback to basic string matching
        return _basic_string_match(solution_str, ground_truth)


def _get_judge(**kwargs) -> Any:
    """Get or create the singleton StructuredJudge instance.

    Implements lazy initialization to avoid creating the judge until needed.

    Args:
        **kwargs: Arguments for create_verl_judge (base_url, model, max_concurrent).

    Returns:
        StructuredJudge instance or None if unavailable.
    """
    global _judge_instance

    if _judge_instance is None:
        try:
            from verl.utils.reward_score.llm_judge import create_verl_judge
            _judge_instance = create_verl_judge(
                base_url=kwargs.get("base_url", "http://0.0.0.0:8000/v1"),
                model=kwargs.get("model"),
                max_concurrent=kwargs.get("max_concurrent", 128),
            )
            logger.info("Initialized singleton StructuredJudge instance")
        except ImportError:
            logger.warning("StructuredJudge not available (llm_judge module not found)")
            return None

    return _judge_instance


def _compute_code_score_llm_judge(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute code reward using LLM-as-a-judge via StructuredJudge.

    Uses singleton pattern to reuse judge instance across calls.

    Args:
        solution_str: The model's generated code solution.
        ground_truth: The expected ground truth or problem description.
        extra_info: Optional dict with problem context.
        **kwargs: Additional arguments (base_url, model, etc. for judge).

    Returns:
        Score as a float (0.0 or 1.0).
    """
    judge = _get_judge(**kwargs)

    if judge is None:
        return _basic_string_match(solution_str, str(ground_truth))

    try:
        # Get problem context
        prompt = ""
        if extra_info and isinstance(extra_info, dict):
            prompt = extra_info.get("original_prompt", "") or extra_info.get("problem", "")

        # Use judge to compute reward
        rewards = judge.compute_rewards(
            prompts=[prompt],
            responses=[solution_str],
            reference_answers=[str(ground_truth)],
        )
        return float(rewards[0]) if rewards else 0.0

    except Exception as e:
        logger.warning(f"LLM judge failed: {e}, falling back to basic comparison")
        return _basic_string_match(solution_str, str(ground_truth))


def _basic_string_match(solution_str: str, ground_truth: str) -> float:
    """Basic string matching for fallback."""
    import re

    # Extract answer after </think> if present
    match = re.search(r"</think>\s*(.*)$", solution_str, re.DOTALL | re.IGNORECASE)
    answer = match.group(1).strip() if match else solution_str.strip()

    # Normalize
    answer = answer.lower().strip()
    ground_truth = str(ground_truth).lower().strip()

    if answer == ground_truth:
        return 1.0
    if ground_truth in answer:
        return 1.0
    return 0.0
