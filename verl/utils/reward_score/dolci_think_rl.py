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
- code: Uses LLM-as-a-judge via StructuredJudge (BATCHED)
- chat: Uses basic verifiable matching

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import logging
from typing import Optional, Any, List

logger = logging.getLogger(__name__)

# Singleton instance for LLM judge
_judge_instance: Optional[Any] = None


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


def compute_score(
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute the score for a single Dolci-Think-RL solution.

    NOTE: For code tasks, prefer using compute_score_batch() for efficiency.

    Routes to appropriate existing reward function based on dataset_source:
    - math: math_dapo (verifiable)
    - IF: ifeval (verifiable)
    - code: LLM-as-a-judge (inefficient for single calls, use batch)
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
        # Single call - still use judge but with batch of 1
        # For better performance, use compute_score_batch
        scores = _compute_code_scores_batched(
            [solution_str], [ground_truth], [extra_info], **kwargs
        )
        return scores[0]

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


def compute_score_batch(
    solution_strs: List[str],
    ground_truths: List,
    extra_infos: Optional[List[dict]] = None,
    **kwargs,
) -> List[float]:
    """Compute scores for a batch of Dolci-Think-RL solutions.

    Groups samples by dataset_source and processes efficiently:
    - math/IF/chat: processed individually (already fast)
    - code: batched together for single LLM judge call

    Args:
        solution_strs: List of model solutions.
        ground_truths: List of ground truth answers.
        extra_infos: List of extra_info dicts with 'dataset_source'.
        **kwargs: Additional arguments (base_url, model for judge).

    Returns:
        List of scores.
    """
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [None] * n

    # Initialize results
    scores = [0.0] * n

    # Group indices by dataset_source
    math_indices = []
    if_indices = []
    code_indices = []
    other_indices = []

    for i in range(n):
        extra_info = extra_infos[i] or {}
        dataset_source = extra_info.get("dataset_source", "").lower()

        if dataset_source == "math":
            math_indices.append(i)
        elif dataset_source == "if":
            if_indices.append(i)
        elif dataset_source == "code":
            code_indices.append(i)
        else:
            other_indices.append(i)

    # Process math (verifiable, fast)
    if math_indices:
        from . import math_dapo
        for i in math_indices:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else None
            if solution_strs[i] is not None and gt is not None:
                scores[i] = math_dapo.compute_score(solution_strs[i], gt)

    # Process IF (verifiable, fast)
    if if_indices:
        from . import ifeval
        for i in if_indices:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else None
            if solution_strs[i] is not None and gt is not None:
                scores[i] = ifeval.compute_score(solution_strs[i], gt, extra_infos[i])

    # Process code in BATCH (LLM judge)
    if code_indices:
        code_solutions = [solution_strs[i] for i in code_indices]
        code_gts = []
        code_extras = []
        for i in code_indices:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else ""
            code_gts.append(gt)
            code_extras.append(extra_infos[i])

        code_scores = _compute_code_scores_batched(
            code_solutions, code_gts, code_extras, **kwargs
        )
        for idx, i in enumerate(code_indices):
            scores[i] = code_scores[idx]

    # Process other (fallback)
    if other_indices:
        from . import math_dapo
        for i in other_indices:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else None
            if solution_strs[i] is None or gt is None:
                continue
            try:
                score = math_dapo.compute_score(solution_strs[i], gt)
                if score > 0:
                    scores[i] = score
                    continue
            except Exception:
                pass
            scores[i] = _basic_string_match(solution_strs[i], gt)

    return scores


def _compute_code_scores_batched(
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Optional[dict]],
    **kwargs,
) -> List[float]:
    """Compute code rewards using batched LLM-as-a-judge.

    Args:
        solution_strs: List of code solutions.
        ground_truths: List of ground truth answers.
        extra_infos: List of extra_info dicts with problem context.
        **kwargs: Arguments for judge (base_url, model, max_concurrent).

    Returns:
        List of scores.
    """
    n = len(solution_strs)
    if n == 0:
        return []

    judge = _get_judge(**kwargs)

    if judge is None:
        # Fallback to basic matching
        return [
            _basic_string_match(sol, str(gt)) if sol else 0.0
            for sol, gt in zip(solution_strs, ground_truths)
        ]

    try:
        # Extract prompts from extra_info
        prompts = []
        for extra_info in extra_infos:
            prompt = ""
            if extra_info and isinstance(extra_info, dict):
                prompt = extra_info.get("original_prompt", "") or extra_info.get("problem", "")
            prompts.append(prompt)

        # Batch call to judge
        rewards = judge.compute_rewards(
            prompts=prompts,
            responses=solution_strs,
            reference_answers=[str(gt) for gt in ground_truths],
        )

        return [float(r) for r in rewards]

    except Exception as e:
        logger.warning(f"Batched LLM judge failed: {e}, falling back to basic comparison")
        return [
            _basic_string_match(sol, str(gt)) if sol else 0.0
            for sol, gt in zip(solution_strs, ground_truths)
        ]
