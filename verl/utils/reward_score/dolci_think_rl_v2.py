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
- instruction_following/ifeval: Uses ifeval (verifiable)
- code/code_stdio: Uses sandbox_fusion for code execution (PARALLEL)
- general-quality/general-quality_ref: Uses StructuredJudge LLM-as-a-judge (PARALLEL)
- other: Falls back to math verification then string matching

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import ast
import threading
from typing import Any, Optional
from tqdm.auto import tqdm
from collections import defaultdict

import requests

logger = logging.getLogger(__name__)

from verl.utils.reward_score.judgev3 import StructuredJudge

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SANDBOX_TIMEOUT = 100
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_MAX_WORKERS = max(32, os.cpu_count() * 4 if os.cpu_count() else 32)

# Kubernetes proxy configuration
if "KUBERNETES_SERVICE_HOST" in os.environ and os.getenv("KUBERNETES_SERVICE_HOST") != "":
    if os.getenv("NO_PROXY"):
        os.environ["NO_PROXY"] += ",.svc.cluster.local"
    if os.getenv("no_proxy"):
        os.environ["no_proxy"] += ",.svc.cluster.local"

# LLM Judge configuration
DEFAULT_LLM_JUDGE_URL = os.environ.get(
    "LLM_JUDGE_URL",
    "http://qpn744-vllm-gptoss120b-svc.llm-pretraining.svc.cluster.local:8000/v1",
)
DEFAULT_LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "openai/gpt-oss-120b")
DEFAULT_LLM_JUDGE_MAX_CONCURRENT = int(os.environ.get("LLM_JUDGE_MAX_CONCURRENT", "1024"))
DEFAULT_LLM_JUDGE_TIMEOUT = float(os.environ.get("LLM_JUDGE_TIMEOUT", "30.0"))
DEFAULT_LLM_JUDGE_BATCH_TIMEOUT = float(os.environ.get("LLM_JUDGE_BATCH_TIMEOUT", "1200.0"))

# Sandbox Fusion configuration
SANDBOX_FUSION_URL = "http://sandbox-fusion-code-rl-service.llm-pretraining.svc.cluster.local:8080/run_code"

_wandb_step = {"reward": 0}


# =============================================================================
# Ability Bucket Mapping (shared between compute_score and compute_score_batch)
# =============================================================================

ABILITY_BUCKETS = {
    "math": "math",
    "instruction_following": "if",
    "ifeval": "if",
    "code": "code",
    "code_stdio": "code",
    "general-quality": "general_quality",
    "general-quality_ref": "general_quality",
}


def get_task_bucket(data_source: Optional[str] = None, extra_info: Optional[dict] = None) -> str:
    """
    Determine the task bucket from data_source or extra_info.

    Uses the same logic as bucket_indices() in compute_score_batch for consistency.

    Args:
        data_source: Direct data_source value (from naive reward manager)
        extra_info: Extra info dict containing ability/dataset_source

    Returns:
        Task bucket string: "math", "if", "code", "general_quality", or "other"
    """
    extra_info = extra_info or {}

    # Priority: data_source arg > ability > dataset_source
    key = data_source or extra_info.get("ability") or extra_info.get("dataset_source", "")
    key = key.lower() if key else ""

    return ABILITY_BUCKETS.get(key, "other")


# =============================================================================
# StructuredJudge Singleton Management
# =============================================================================

_judge_instance: Optional["StructuredJudge"] = None
_judge_lock = threading.Lock()


def _get_or_create_judge(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    max_concurrent: int = DEFAULT_LLM_JUDGE_MAX_CONCURRENT,
    timeout: float = DEFAULT_LLM_JUDGE_TIMEOUT,
    batch_timeout: float = DEFAULT_LLM_JUDGE_BATCH_TIMEOUT,
) -> "StructuredJudge":
    """
    Get or create a singleton StructuredJudge instance.

    Thread-safe lazy initialization ensures we reuse connections and avoid
    repeatedly detecting models from the API.
    """
    global _judge_instance

    if _judge_instance is not None:
        return _judge_instance

    with _judge_lock:
        # Double-check after acquiring lock
        if _judge_instance is not None:
            return _judge_instance

        effective_url = base_url or DEFAULT_LLM_JUDGE_URL
        effective_model = model or DEFAULT_LLM_JUDGE_MODEL

        _judge_instance = StructuredJudge(
            base_url=effective_url,
            api_key="dummy",  # vLLM typically doesn't require auth
            model=effective_model,
            max_concurrent=max_concurrent,
            timeout=timeout,
            max_retries=3,
            retry_base_delay=0.1,
            batch_timeout=batch_timeout,
        )

        logger.info(
            f"Initialized StructuredJudge: url={effective_url}, "
            f"model={_judge_instance.model}, max_concurrent={max_concurrent}"
        )

        return _judge_instance


def reset_judge() -> None:
    """Reset the singleton judge instance."""
    global _judge_instance

    with _judge_lock:
        if _judge_instance is not None:
            _judge_instance.close()
            _judge_instance = None


# =============================================================================
# Text Processing Utilities
# =============================================================================

def recursive_literal_eval(obj: Any, *, _depth: int = 0, max_depth: int = 100) -> Any:
    """Recursively apply ast.literal_eval to string representations of Python literals."""
    if _depth > max_depth:
        raise RecursionError(f"Exceeded max recursion depth: {max_depth}")

    if isinstance(obj, str):
        try:
            evaluated = ast.literal_eval(obj)
            return recursive_literal_eval(evaluated, _depth=_depth + 1, max_depth=max_depth)
        except (ValueError, SyntaxError, TypeError):
            return obj

    if isinstance(obj, dict):
        return {
            recursive_literal_eval(k, _depth=_depth + 1, max_depth=max_depth):
            recursive_literal_eval(v, _depth=_depth + 1, max_depth=max_depth)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [recursive_literal_eval(item, _depth=_depth + 1, max_depth=max_depth) for item in obj]

    if isinstance(obj, tuple):
        return tuple(recursive_literal_eval(item, _depth=_depth + 1, max_depth=max_depth) for item in obj)

    if isinstance(obj, set):
        return {recursive_literal_eval(item, _depth=_depth + 1, max_depth=max_depth) for item in obj}

    return obj


def remove_thinking_section(prediction: str) -> str:
    """
    Remove thinking/reasoning sections from model output before reward computation.

    Strips <think>...</think>, <evaluation>...</evaluation>, and <answer> tags.
    """
    if prediction is None:
        return ""

    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.split("</evaluation>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")

    return prediction.strip()


def _basic_string_match(solution_str: str, ground_truth: str) -> float:
    """Basic string matching for fallback scoring."""
    answer = solution_str.lower().strip()
    ground_truth = str(ground_truth).lower().strip()

    if answer == ground_truth:
        return 1.0
    if ground_truth in answer:
        return 1.0
    return 0.0


# =============================================================================
# Code Scoring with sandbox_fusion
# =============================================================================


def _extract_fn_name_from_asserts(ground_truth: Any) -> Optional[str]:
    """Extract the expected function name from assert-style test cases."""
    if not ground_truth:
        return None
    if not isinstance(ground_truth, list) or len(ground_truth) == 0:
        return None

    first_item = ground_truth[0]
    if not isinstance(first_item, str):
        return None

    first_item = first_item.strip()
    if not first_item.startswith("assert"):
        return None

    # Pattern 1: assert fn_name(...) - most common
    match = re.search(r'assert\s+(\w+)\s*\(', first_item)
    if match:
        fn_name = match.group(1)
        if fn_name not in ('True', 'False', 'None', 'not', 'len', 'str', 'int',
                           'float', 'list', 'dict', 'set', 'tuple', 'sorted',
                           'abs', 'sum', 'min', 'max', 'round', 'type', 'all', 'any'):
            return fn_name

    # Pattern 2: assert Solution().method_name(...) - LeetCode style
    match = re.search(r'assert\s+\w+\(\)\s*\.\s*(\w+)\s*\(', first_item)
    if match:
        return match.group(1)

    # Pattern 3: assert obj.method_name(...) - instance method
    match = re.search(r'assert\s+\w+\s*\.\s*(\w+)\s*\(', first_item)
    if match:
        return match.group(1)

    return None


def _convert_to_sandbox_format(ground_truth: Any) -> Optional[dict]:
    """
    Converts the normalized ground truth into the dict format required by Sandbox Fusion.
    """
    if not ground_truth:
        return None

    if isinstance(ground_truth, str):
        ground_truth = recursive_literal_eval(ground_truth)

    # Case 1: Ground truth is already a dict
    if isinstance(ground_truth, dict):
        return ground_truth

    # Case 2: Ground truth is a list
    if isinstance(ground_truth, list) and len(ground_truth) > 0:
        first_item = ground_truth[0]

        # Sub-case A: List of Assert Strings (Unit Tests)
        if isinstance(first_item, str):
            result = {
                "assert_case": ground_truth,
                "inputs": [""] * len(ground_truth),
                "outputs": [None] * len(ground_truth)
            }
            fn_name = _extract_fn_name_from_asserts(ground_truth)
            if fn_name:
                result["fn_name"] = fn_name
            return result

        # Sub-case B: List of IO Dictionaries (Standard Input/Output)
        elif isinstance(first_item, dict) and 'input' in first_item:
            return {
                "inputs": [item.get('input', '') for item in ground_truth],
                "outputs": [item.get('output', '') for item in ground_truth]
            }

    return None


def _compute_code_score_sandbox(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    sandbox_fusion_url: Optional[str] = SANDBOX_FUSION_URL,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
) -> float:
    """Compute code score using sandbox_fusion execution."""
    test_cases = _convert_to_sandbox_format(ground_truth)

    if not test_cases:
        logger.debug("No valid test cases for code scoring, using string match")
        return _basic_string_match(solution_str, str(ground_truth)) if ground_truth else 0.0

    try:
        from verl.utils.reward_score import sandbox_fusion

        score, err = sandbox_fusion.compute_score(
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
            completion=solution_str,
            test_cases=test_cases,
            continuous=True,
            timeout=timeout,
        )
        return float(score)
    except Exception as e:
        logger.warning(f"Sandbox execution failed: {e}")
        return 0.0


# =============================================================================
# LLM Judge Scoring (using StructuredJudge)
# =============================================================================

def _compute_llm_judge_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    **kwargs,
) -> float:
    """Compute score using StructuredJudge LLM-as-a-judge (Single Item)."""
    if not solution_str or not solution_str.strip():
        return 0.0

    prompt = ""
    if extra_info and isinstance(extra_info, dict):
        prompt = extra_info.get("original_prompt", "") or extra_info.get("problem", "")

    prompt = prompt[:4000] if prompt else ""
    response = solution_str[:4000]
    reference = str(ground_truth)[:4000] if ground_truth else ""

    try:
        judge = _get_or_create_judge(
            base_url=llm_judge_url,
            model=llm_judge_model,
        )

        evaluation = judge.evaluate(
            prompt=prompt,
            response=response,
            temperature=0.0,
            reference_answer=reference,
        )

        if evaluation is None:
            return 0.0

        return evaluation.response_quality.score / 10.0

    except Exception as e:
        logger.warning(f"LLM judge scoring failed: {e}")
        return 0.0


def _compute_llm_judge_scores_batch(
    indices: list[int],
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[Optional[dict]],
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    show_progress: bool = True,
) -> dict[int, float]:
    """Compute LLM judge scores for a batch of samples using StructuredJudge."""
    if not indices:
        return {}

    valid_indices = []
    prompts = []
    responses = []
    references = []

    results: dict[int, float] = {}

    for i in indices:
        solution = solution_strs[i] if solution_strs[i] else ""
        if not solution.strip():
            results[i] = 0.0
            continue

        gt = ground_truths[i]
        if isinstance(gt, list):
            gt = gt[0] if gt else ""

        if gt is None:
            results[i] = 0.0
            continue

        extra_info = extra_infos[i] or {}
        prompt = extra_info.get("original_prompt", "") or extra_info.get("problem", "")

        valid_indices.append(i)
        prompts.append(prompt[:-1] if prompt else "")
        responses.append(solution[:-1])
        references.append(str(gt)[:-1] if gt else "")

    if not valid_indices:
        return results

    try:
        judge = _get_or_create_judge(
            base_url=llm_judge_url,
            model=llm_judge_model,
        )

        rewards = judge.compute_rewards(
            prompts=prompts,
            responses=responses,
            temperature=0.0,
            show_progress=show_progress,
            reference_answers=references,
        )

        for idx, reward in zip(valid_indices, rewards):
            results[idx] = reward

        return results

    except Exception as e:
        logger.error(f"Batch LLM judge scoring failed: {e}")
        for idx in valid_indices:
            results[idx] = 0.0
        return results


# =============================================================================
# Helper: Unwrap ground truth consistently
# =============================================================================

def _unwrap_ground_truth(ground_truth: Any, task_bucket: str) -> Any:
    """
    Unwrap ground_truth consistently for both compute_score and compute_score_batch.

    For code tasks, keep the list as-is (it contains test cases).
    For other tasks, unwrap single-element lists.
    """
    if task_bucket == "code":
        # Code tasks: keep as-is, the list IS the ground truth (test cases)
        return ground_truth

    # Non-code tasks: unwrap single-element lists
    if isinstance(ground_truth, list):
        if len(ground_truth) == 1 and isinstance(ground_truth[0], str):
            return ground_truth[0]
        elif len(ground_truth) == 0:
            return None

    return ground_truth


# =============================================================================
# Main Scoring Functions
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    data_source: Optional[str] = None,  # Added: passed by naive reward manager
    sandbox_fusion_url: Optional[str] = SANDBOX_FUSION_URL,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    **kwargs,
) -> float:
    """
    Compute the score for a single Dolci-Think-RL solution.

    This function has the same behavior as processing a single item in compute_score_batch.

    Args:
        solution_str: The model's response string.
        ground_truth: The expected answer/test cases.
        extra_info: Additional context (contains ability, dataset_source, etc.)
        data_source: Direct data_source value (from naive reward manager)
        sandbox_fusion_url: URL for code sandbox execution.
        concurrent_semaphore: Semaphore for concurrency control.
        memory_limit_mb: Memory limit for sandbox execution.
        timeout: Timeout for sandbox execution.
        llm_judge_url: URL for LLM judge.
        llm_judge_model: Model name for LLM judge.
        **kwargs: Additional arguments (ignored, for compatibility).

    Returns:
        Score as float (0.0 to 1.0).
    """
    if solution_str is None:
        return 0.0

    if ground_truth is None:
        return 0.0

    # Determine task bucket using same logic as compute_score_batch
    task_bucket = get_task_bucket(data_source, extra_info)

    # Unwrap ground_truth consistently
    ground_truth = _unwrap_ground_truth(ground_truth, task_bucket)

    if ground_truth is None:
        return 0.0

    # Remove thinking section before reward computation
    solution_str = remove_thinking_section(solution_str)

    # Get URLs from extra_info if not provided
    extra_info = extra_info or {}
    if sandbox_fusion_url is None:
        sandbox_fusion_url = SANDBOX_FUSION_URL
    if llm_judge_url is None:
        llm_judge_url = extra_info.get("llm_judge_url") or DEFAULT_LLM_JUDGE_URL

    # Route to appropriate reward function based on task bucket
    if task_bucket == "math":
        from verl.utils.reward_score.math_verify import MathVerifier
        verifier = MathVerifier()
        try:
            result = verifier.compute_score(solution_str, ground_truth)
            return result["score"]
        except Exception as e:
            logger.debug(f"Math verification failed: {e}")
            return 0.0

    elif task_bucket == "if":
        from verl.utils.reward_score import ifeval
        try:
            result = ifeval.compute_score(solution_str, ground_truth, extra_info)
            return result["score"]
        except Exception as e:
            logger.debug(f"IFEval scoring failed: {e}")
            return 0.0

    elif task_bucket == "code":
        return _compute_code_score_sandbox(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
            timeout=timeout,
        )

    elif task_bucket == "general_quality":
        return _compute_llm_judge_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

    else:
        # Default fallback (same as "other" in compute_score_batch)
        try:
            from verl.utils.reward_score.math_verify import MathVerifier
            verifier = MathVerifier()
            result = verifier.compute_score(solution_str, ground_truth)
            if result["score"] > 0:
                return result["score"]
        except Exception:
            pass

        return _basic_string_match(solution_str, ground_truth)


def bucket_indices(extra_infos: list[dict | None]) -> dict[str, list[int]]:
    """Partition indices by ability, falling back to dataset_source."""
    buckets = defaultdict(list)
    for i, info in enumerate(extra_infos):
        bucket = get_task_bucket(None, info)
        buckets[bucket].append(i)
    return buckets


def compute_score_batch(
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: Optional[list[dict]] = None,
    sandbox_fusion_url: Optional[str] = SANDBOX_FUSION_URL,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    show_progress: bool = True,
    **kwargs,
) -> list[float]:
    """
    Compute scores for a batch of Dolci-Think-RL solutions.

    Uses parallel execution for code (ThreadPoolExecutor) and LLM judge (async batch).
    """
    n = len(solution_strs)
    if n == 0:
        return []

    if extra_infos is None:
        extra_infos = [None] * n

    logger.info(f"compute_score_batch called with {n} samples")

    # Clean solutions upfront
    solution_strs = [remove_thinking_section(s) if s else "" for s in solution_strs]

    scores = [0.0] * n

    buckets = bucket_indices(extra_infos)
    math_indices = buckets["math"]
    if_indices = buckets["if"]
    code_indices = buckets["code"]
    general_quality_indices = buckets["general_quality"]
    other_indices = buckets["other"]

    # Log group sizes
    batch_info = {
        "math_indices": len(math_indices),
        "if_indices": len(if_indices),
        "code_indices": len(code_indices),
        "general_quality_indices": len(general_quality_indices),
        "other_indices": len(other_indices),
    }
    logger.info(f"Batch distribution: {batch_info}")

    # -------------------------------------------------------------------------
    # 1. Process math (Sequential)
    # -------------------------------------------------------------------------
    if math_indices:
        from verl.utils.reward_score.math_verify import MathVerifier
        verifier = MathVerifier()
        for i in math_indices:
            gt = _unwrap_ground_truth(ground_truths[i], "math")

            if solution_strs[i] and gt is not None:
                try:
                    result = verifier.compute_score(solution_strs[i], gt)
                    scores[i] = result["score"]
                except Exception as e:
                    logger.debug(f"Math verification failed for index {i}: {e}")

    # -------------------------------------------------------------------------
    # 2. Process IF (Sequential)
    # -------------------------------------------------------------------------
    if if_indices:
        from verl.utils.reward_score import ifeval
        for i in if_indices:
            gt = _unwrap_ground_truth(ground_truths[i], "if")

            if solution_strs[i] and gt is not None:
                try:
                    result = ifeval.compute_score(solution_strs[i], gt, extra_infos[i])
                    scores[i] = result["score"]
                except Exception as e:
                    logger.debug(f"IFEval scoring failed for index {i}: {e}")

    # -------------------------------------------------------------------------
    # 3. Process code (Parallel Sandbox)
    # -------------------------------------------------------------------------
    if code_indices:
        def _score_code_task(i: int) -> tuple[int, float]:
            if not solution_strs[i]:
                return i, 0.0

            # Code tasks: keep ground_truth as-is (list of test cases)
            gt = ground_truths[i]

            try:
                score = _compute_code_score_sandbox(
                    solution_str=solution_strs[i],
                    ground_truth=gt,
                    extra_info=extra_infos[i],
                    sandbox_fusion_url=SANDBOX_FUSION_URL,
                    concurrent_semaphore=concurrent_semaphore,
                    memory_limit_mb=memory_limit_mb,
                    timeout=timeout,
                )
                return i, score
            except Exception as e:
                logger.debug(f"Code scoring failed for index {i}: {e}")
                return i, 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_score_code_task, i) for i in code_indices]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(code_indices), desc="Code Sandbox", disable=not show_progress):
                idx, score = future.result()
                scores[idx] = score

    # -------------------------------------------------------------------------
    # 4. Process general-quality (Parallel Judge)
    # -------------------------------------------------------------------------
    if general_quality_indices:
        effective_llm_judge_url = llm_judge_url
        if effective_llm_judge_url is None and general_quality_indices:
            first_idx = general_quality_indices[0]
            if extra_infos[first_idx] and extra_infos[first_idx].get("llm_judge_url"):
                effective_llm_judge_url = extra_infos[first_idx].get("llm_judge_url")

        batch_scores = _compute_llm_judge_scores_batch(
            indices=general_quality_indices,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            llm_judge_url=effective_llm_judge_url,
            llm_judge_model=llm_judge_model,
            show_progress=show_progress,
        )

        for idx, score in batch_scores.items():
            scores[idx] = score

    # -------------------------------------------------------------------------
    # 5. Process other (Fallback)
    # -------------------------------------------------------------------------
    if other_indices:
        try:
            from verl.utils.reward_score.math_verify import MathVerifier
            verifier = MathVerifier()
        except ImportError:
            verifier = None

        for i in other_indices:
            gt = _unwrap_ground_truth(ground_truths[i], "other")

            if not solution_strs[i] or gt is None:
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

    # -------------------------------------------------------------------------
    # 6. Log domain-wise mean rewards
    # -------------------------------------------------------------------------
    domain_stats = {}

    if math_indices:
        math_scores = [scores[i] for i in math_indices]
        domain_stats["math"] = {
            "count": len(math_indices),
            "mean": sum(math_scores) / len(math_scores),
        }

    if if_indices:
        if_scores = [scores[i] for i in if_indices]
        domain_stats["ifeval"] = {
            "count": len(if_indices),
            "mean": sum(if_scores) / len(if_scores),
        }

    if code_indices:
        code_scores = [scores[i] for i in code_indices]
        domain_stats["code"] = {
            "count": len(code_indices),
            "mean": sum(code_scores) / len(code_scores),
        }

    if general_quality_indices:
        gq_scores = [scores[i] for i in general_quality_indices]
        domain_stats["general_quality"] = {
            "count": len(general_quality_indices),
            "mean": sum(gq_scores) / len(gq_scores),
        }

    if other_indices:
        other_scores = [scores[i] for i in other_indices]
        domain_stats["other"] = {
            "count": len(other_indices),
            "mean": sum(other_scores) / len(other_scores),
        }

    # Print domain-wise rewards
    overall_mean = sum(scores) / len(scores) if scores else 0.0
    domain_summary = " | ".join([f"{k}: {v['mean']:.4f} (n={v['count']})" for k, v in domain_stats.items()])
    logger.info(f"Domain rewards: {domain_summary} | overall: {overall_mean:.4f} (n={len(scores)})")

    try:
        import wandb
        if wandb.run is not None:
            is_validation = (
                extra_infos is not None
                and len(extra_infos) > 0
                and any(ei.get("validate", False) for ei in extra_infos if ei)
            )

            prefix = "val" if is_validation else "train"

            wandb_metrics = {
                f"{prefix}/{domain}/reward/mean@1": stats["mean"]
                for domain, stats in domain_stats.items()
            } | {
                f"{prefix}/{domain}/reward/count": stats["count"]
                for domain, stats in domain_stats.items()
            } | {
                f"{prefix}/overall/reward/mean@1": overall_mean
            }

            wandb.log(wandb_metrics, commit=False)
    except Exception as e:
        logger.error(f"Error in wandb logging: {e}")

    return scores
