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
- math: Uses math_verify.MathVerifier (verifiable)
- instruction_following: Uses ifeval (verifiable)
- code: Uses LLM-as-a-judge via StructuredJudge (BATCHED)
- chat: Uses basic verifiable matching

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Singleton instance for LLM judge
_judge_instance: Optional[Any] = None


def remove_thinking_section(prediction: str) -> str:
    """Remove thinking/reasoning sections from model output before reward computation.

    Strips <think>...</think>, <evaluation>...</evaluation>, and <answer> tags.
    """
    if prediction is None:
        return ""
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove evaluation
    prediction = prediction.split("</evaluation>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def _get_judge(**kwargs) -> Any:
    """Get or create the singleton StructuredJudge instance.

    Implements lazy initialization to avoid creating the judge until needed.

    Args:
        **kwargs: Arguments for create_verl_judge (base_url, model, max_concurrent).

    Returns:
        StructuredJudge instance or None if unavailable.
    """
    global _judge_instance

    if _judge_instance is not None:
        return _judge_instance

    try:
        from verl.utils.reward_score.judgev3 import create_verl_judge

        _judge_instance = create_verl_judge(
            base_url=kwargs.get(
                "base_url",
                "http://qpn744-vllm-gptoss120b-svc.llm-pretraining.svc.cluster.local:8000/v1",
            ),
            model=kwargs.get("model", "openai/gpt-oss-120b"),
            # max_concurrent=kwargs.get("max_concurrent", 128),
        )
        logger.info("Initialized singleton StructuredJudge instance")
    except ImportError:
        logger.warning("StructuredJudge not available (judgev3 module not found)")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize StructuredJudge: {e}")
        return None

    return _judge_instance


def _basic_string_match(solution_str: str, ground_truth: str) -> float:
    """Basic string matching for fallback.

    NOTE: Assumes thinking section already removed via remove_thinking_section().
    """
    # Normalize
    answer = solution_str.lower().strip()
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
    - math: math_verify.MathVerifier (verifiable)
    - instruction_following: ifeval (verifiable)
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

    # Remove thinking section before reward computation
    solution_str = remove_thinking_section(solution_str)

    # Determine task type from extra_info
    dataset_source = ""
    if extra_info and isinstance(extra_info, dict):
        dataset_source = extra_info.get("dataset_source", "").lower()

    # Route to appropriate existing reward function
    if dataset_source == "math":
        from verl.utils.reward_score.math_verify import MathVerifier

        verifier = MathVerifier()
        result = verifier.compute_score(solution_str, ground_truth)
        return result["score"]

    elif dataset_source == "instruction_following" or dataset_source == "ifeval":
        from verl.utils.reward_score import ifeval

        result = ifeval.compute_score(solution_str, ground_truth, extra_info)
        return result["score"]

    elif dataset_source == "code" or dataset_source == "code_stdio":
        # Single call - still use judge but with batch of 1
        # For better performance, use compute_score_batch
        scores = _compute_code_scores_batched([solution_str], [ground_truth], [extra_info], **kwargs)
        return scores[0]

    else:
        # Default: try math first, then basic matching
        try:
            from verl.utils.reward_score.math_verify import MathVerifier

            verifier = MathVerifier()
            result = verifier.compute_score(solution_str, ground_truth)
            if result["score"] > 0:
                return result["score"]
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
    if n == 0:
        return []

    if extra_infos is None:
        extra_infos = [None] * n

    logger.info(f"compute_score_batch called with {n} samples")

    # Remove thinking sections from all solutions upfront
    solution_strs = [remove_thinking_section(s) if s else "" for s in solution_strs]

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
        elif dataset_source == "instruction_following" or dataset_source == "ifeval":
            if_indices.append(i)
        elif dataset_source == "code":
            code_indices.append(i)
        elif dataset_source == "code_stdio":
            code_indices.append(i)  # same for now, soon will have separate executor
        else:
            other_indices.append(i)

    # Log group sizes
    if math_indices:
        logger.info(f"  math: {len(math_indices)} samples")
    if if_indices:
        logger.info(f"  ifeval: {len(if_indices)} samples")
    if code_indices:
        logger.info(f"  code: {len(code_indices)} samples")
    if other_indices:
        logger.info(f"  other: {len(other_indices)} samples")

    # Process math (verifiable, fast)
    if math_indices:
        from verl.utils.reward_score.math_verify import MathVerifier

        verifier = MathVerifier()

        for i in math_indices:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else None
            if solution_strs[i] is not None and gt is not None:
                try:
                    result = verifier.compute_score(solution_strs[i], gt)
                    scores[i] = result["score"]
                except Exception as e:
                    logger.debug(f"Math verification failed for index {i}: {e}")
                    scores[i] = 0.0

    # Process IF (verifiable, fast)
    if if_indices:
        from verl.utils.reward_score import ifeval

        for i in if_indices:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else None
            if solution_strs[i] is not None and gt is not None:
                try:
                    result = ifeval.compute_score(solution_strs[i], gt, extra_infos[i])
                    scores[i] = result["score"]
                except Exception as e:
                    logger.debug(f"IFEval scoring failed for index {i}: {e}")
                    scores[i] = 0.0

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

        code_scores = _compute_code_scores_batched(code_solutions, code_gts, code_extras, **kwargs)
        for idx, i in enumerate(code_indices):
            scores[i] = code_scores[idx]

    # Process other (fallback)
    if other_indices:
        try:
            from verl.utils.reward_score.math_verify import MathVerifier

            verifier = MathVerifier()
        except ImportError:
            verifier = None

        for i in other_indices:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else None
            if solution_strs[i] is None or gt is None:
                continue
            if verifier:
                try:
                    result = verifier.compute_score(solution_strs[i], gt)
                    if result["score"] > 0:
                        scores[i] = result["score"]
                        continue
                except Exception:
                    pass
            scores[i] = _basic_string_match(solution_strs[i], gt)

    return scores


def compute_score_llm_judge_batch(
    responses: List[str],
    ground_truths: List[str],
    prompts: Optional[List[str]] = None,
    extra_infos: Optional[List[dict]] = None,
    remove_thinking: bool = True,
    fallback_to_string_match: bool = True,
    **kwargs,
) -> List[float]:
    """Compute rewards using batched LLM-as-a-judge.

    Provides a direct interface to the LLM judge for arbitrary response/ground-truth
    pairs without routing through dataset_source logic.

    Args:
        responses: List of model-generated responses to evaluate.
        ground_truths: List of reference/ground-truth answers.
        prompts: Optional list of original prompts/problems for context.
            If not provided, attempts to extract from extra_infos.
        extra_infos: Optional list of dicts containing additional context.
            Used to extract prompts if 'prompts' arg not provided.
            Looks for 'original_prompt' or 'problem' keys.
        remove_thinking: If True, strips <think>...</think> sections from
            responses before scoring. Defaults to True.
        fallback_to_string_match: If True, falls back to basic string matching
            when the LLM judge is unavailable or fails. Defaults to True.
        **kwargs: Additional arguments passed to judge initialization
            (base_url, model, max_concurrent).

    Returns:
        List of float scores, one per response. Scores are typically in [0, 1].

    Raises:
        RuntimeError: If judge unavailable and fallback_to_string_match is False.

    Example:
        >>> scores = compute_score_llm_judge_batch(
        ...     responses=["def add(a, b): return a + b", "def add(a, b): return a - b"],
        ...     ground_truths=["Function should add two numbers", "Function should add two numbers"],
        ...     prompts=["Write a function to add two numbers"] * 2,
        ... )
    """
    n = len(responses)
    if n == 0:
        return []

    if len(ground_truths) != n:
        raise ValueError(f"Length mismatch: responses ({n}) vs ground_truths ({len(ground_truths)})")

    # Normalize ground_truths (handle list-wrapped values from datasets)
    normalized_gts = []
    for gt in ground_truths:
        if isinstance(gt, list):
            normalized_gts.append(gt[0] if gt else "")
        else:
            normalized_gts.append(str(gt) if gt is not None else "")
    ground_truths = normalized_gts

    # Optionally remove thinking sections
    if remove_thinking:
        responses = [remove_thinking_section(r) if r else "" for r in responses]
    else:
        responses = [r if r else "" for r in responses]

    # Build prompts list
    if prompts is None:
        prompts = []
        extra_infos = extra_infos or [None] * n
        for extra_info in extra_infos:
            prompt = ""
            if extra_info and isinstance(extra_info, dict):
                prompt = extra_info.get("original_prompt", "") or extra_info.get("problem", "")
            prompts.append(prompt)
    elif len(prompts) != n:
        raise ValueError(f"Length mismatch: responses ({n}) vs prompts ({len(prompts)})")

    # Get or create judge instance
    judge = _get_judge(**kwargs)

    if judge is None:
        if not fallback_to_string_match:
            raise RuntimeError(
                "LLM judge unavailable and fallback_to_string_match is False. "
                "Ensure judgev3 module is installed or enable fallback."
            )
        logger.warning("LLM judge unavailable, falling back to string matching")
        return [_basic_string_match(resp, gt) for resp, gt in zip(responses, ground_truths)]

    try:
        logger.info(f"Computing LLM judge scores for {n} samples")
        rewards = judge.compute_rewards(
            prompts=prompts,
            responses=responses,
            reference_answers=ground_truths,
        )
        return [float(r) for r in rewards]

    except Exception as e:
        logger.warning(f"LLM judge call failed: {e}")
        if not fallback_to_string_match:
            raise RuntimeError(f"LLM judge failed and fallback disabled: {e}") from e
        logger.warning("Falling back to string matching")
        return [_basic_string_match(resp, gt) for resp, gt in zip(responses, ground_truths)]


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
        return [_basic_string_match(sol, str(gt)) if sol else 0.0 for sol, gt in zip(solution_strs, ground_truths)]

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
        return [_basic_string_match(sol, str(gt)) if sol else 0.0 for sol, gt in zip(solution_strs, ground_truths)]
