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
import threading
from typing import Any, Optional
from tqdm.auto import tqdm

import requests

logger = logging.getLogger(__name__)

from verl.utils.reward_score.judgev3 import StructuredJudge

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SANDBOX_TIMEOUT = 10
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
DEFAULT_LLM_JUDGE_MAX_CONCURRENT = int(os.environ.get("LLM_JUDGE_MAX_CONCURRENT", "128"))
DEFAULT_LLM_JUDGE_TIMEOUT = float(os.environ.get("LLM_JUDGE_TIMEOUT", "30.0"))
DEFAULT_LLM_JUDGE_BATCH_TIMEOUT = float(os.environ.get("LLM_JUDGE_BATCH_TIMEOUT", "120.0"))

# Sandbox Fusion configuration
SANDBOX_FUSION_URL = "http://sandbox-fusion-code-rl-service.llm-pretraining.svc.cluster.local:8080/run_code"


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
    
    Args:
        base_url: OpenAI-compatible API endpoint. Defaults to DEFAULT_LLM_JUDGE_URL.
        model: Model name. Auto-detected if None.
        max_concurrent: Maximum concurrent requests.
        timeout: Per-request timeout in seconds.
        batch_timeout: Global batch timeout in seconds.
        
    Returns:
        StructuredJudge instance.
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
            max_retries=1,
            retry_base_delay=0.1,
            batch_timeout=batch_timeout,
        )
        
        logger.info(
            f"Initialized StructuredJudge: url={effective_url}, "
            f"model={_judge_instance.model}, max_concurrent={max_concurrent}"
        )
        
        return _judge_instance


def reset_judge() -> None:
    """
    Reset the singleton judge instance.
    
    Useful for testing or when configuration changes.
    """
    global _judge_instance
    
    with _judge_lock:
        if _judge_instance is not None:
            _judge_instance.close()
            _judge_instance = None


# =============================================================================
# Text Processing Utilities
# =============================================================================

def remove_thinking_section(prediction: str) -> str:
    """
    Remove thinking/reasoning sections from model output before reward computation.

    Strips <think>...</think>, <evaluation>...</evaluation>, and <answer> tags.
    
    Args:
        prediction: Raw model output string.
        
    Returns:
        Cleaned prediction string.
    """
    if prediction is None:
        return ""
    
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.split("</evaluation>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    
    return prediction.strip()


def _basic_string_match(solution_str: str, ground_truth: str) -> float:
    """
    Basic string matching for fallback scoring.

    NOTE: Assumes thinking section already removed via remove_thinking_section().
    
    Args:
        solution_str: Model's cleaned response.
        ground_truth: Expected answer.
        
    Returns:
        1.0 if match found, 0.0 otherwise.
    """
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
    """
    Extract the expected function name from assert-style test cases.

    This helps identify what function the model should implement, which can be
    used for validation or to provide better error messages.

    Args:
        ground_truth: List of assert strings, e.g. ["assert add(1,2)==3", ...]

    Returns:
        Function name if found, None otherwise.
    """
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
        # Filter out built-in functions and keywords
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
    Converts the normalized ground truth (from preprocessor) into
    the specific dict format required by Sandbox Fusion.
    
    The preprocessor now ensures ground_truth is already a clean Python object
    (no compressed blobs), typically:
      - IO: [{'input': 'a', 'output': 'b'}, {'input': 'c', 'output': 'd'}]
      - Assert: ["assert a==b", "assert c==d"]
      
    This function pivots that structure into:
      - IO: {'inputs': ['a', 'c'], 'outputs': ['b', 'd']}
      - Assert: {'assert_case': ["assert a==b", "assert c==d"], 'inputs': ...}
    """
    if not ground_truth:
        return None
        
    # Case 1: Ground truth is already a dict (rare, but possible if passed through directly)
    if isinstance(ground_truth, dict):
        return ground_truth

    # Case 2: Ground truth is a list
    if isinstance(ground_truth, list) and len(ground_truth) > 0:
        first_item = ground_truth[0]
        
        # Sub-case A: List of Assert Strings (Unit Tests)
        if isinstance(first_item, str):
            # Sandbox fusion requires "inputs" array to match length of "assert_case"
            result = {
                "assert_case": ground_truth,
                "inputs": [""] * len(ground_truth),
                "outputs": [None] * len(ground_truth)
            }
            # Extract and include the expected function name
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
    """
    Compute code score using sandbox_fusion execution.

    Args:
        solution_str: The model's code solution.
        ground_truth: Cleaned ground truth object (List or Dict).
        extra_info: Additional context.
        sandbox_fusion_url: URL of the sandbox service.
        concurrent_semaphore: Semaphore for concurrency control.
        memory_limit_mb: Memory limit for execution.
        timeout: Timeout for execution.

    Returns:
        Score as float (0.0 to 1.0).
    """
    # 1. Pivot the data structure for the sandbox
    test_cases = _convert_to_sandbox_format(ground_truth)
    
    if not test_cases:
        logger.debug("No valid test cases for code scoring, using string match")
        # Fallback to string matching if formatting fails
        return _basic_string_match(solution_str, str(ground_truth)) if ground_truth else 0.0

    try:
        from verl.utils.reward_score import sandbox_fusion

        score, _ = sandbox_fusion.compute_score(
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
    """
    Compute score using StructuredJudge LLM-as-a-judge (Single Item).
    """
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
    """
    Compute LLM judge scores for a batch of samples using StructuredJudge.
    Leverages async batch processing for high throughput.
    """
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
        prompts.append(prompt[:4000] if prompt else "")
        responses.append(solution[:4000])
        references.append(str(gt)[:4000] if gt else "")
    
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
# Main Scoring Functions
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
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
    """
    if solution_str is None:
        return 0.0

    # Handle ground_truth being a list (common wrapper)
    if isinstance(ground_truth, list) and not extra_info.get("dataset_source", "") == "code":
        # For non-code tasks, we often peel the list.
        # For code tasks, the list IS the ground truth (e.g. list of IO pairs), so we keep it.
        if isinstance(ground_truth[0], str) and len(ground_truth) == 1:
             ground_truth = ground_truth[0]

    if ground_truth is None:
        return 0.0

    # Remove thinking section before reward computation
    solution_str = remove_thinking_section(solution_str)

    # Determine task type
    dataset_source = ""
    if extra_info and isinstance(extra_info, dict):
        dataset_source = extra_info.get("dataset_source", "").lower()
        if sandbox_fusion_url is None:
            sandbox_fusion_url = SANDBOX_FUSION_URL
        if llm_judge_url is None:
            llm_judge_url = DEFAULT_LLM_JUDGE_URL

    # Route to appropriate reward function
    if dataset_source == "math":
        from verl.utils.reward_score.math_verify import MathVerifier
        verifier = MathVerifier()
        result = verifier.compute_score(solution_str, ground_truth)
        return result["score"]

    elif dataset_source in ("instruction_following", "ifeval"):
        from verl.utils.reward_score import ifeval
        result = ifeval.compute_score(solution_str, ground_truth, extra_info)
        return result["score"]

    elif dataset_source in ("code", "code_stdio"):
        return _compute_code_score_sandbox(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
            timeout=timeout,
        )

    elif dataset_source in ("general-quality", "general-quality_ref"):
        return _compute_llm_judge_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

    else:
        # Default fallback
        try:
            from verl.utils.reward_score.math_verify import MathVerifier
            verifier = MathVerifier()
            result = verifier.compute_score(solution_str, ground_truth)
            if result["score"] > 0:
                return result["score"]
        except Exception:
            pass

        return _basic_string_match(solution_str, ground_truth)


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

    # Group indices
    math_indices = []
    if_indices = []
    code_indices = []
    general_quality_indices = []
    other_indices = []

    for i in range(n):
        extra_info = extra_infos[i] or {}
        dataset_source = extra_info.get("dataset_source", "").lower()

        if dataset_source == "math":
            math_indices.append(i)
        elif dataset_source in ("instruction_following", "ifeval"):
            if_indices.append(i)
        elif dataset_source in ("code", "code_stdio"):
            code_indices.append(i)
        elif dataset_source in ("general-quality", "general-quality_ref"):
            general_quality_indices.append(i)
        else:
            other_indices.append(i)

    # Log group sizes
    batch_info = {
        "math_indices" : len(math_indices),
        "if_indices" : len(if_indices),
        "code_indices" : len(code_indices),
        "general_quality_indices" : len(general_quality_indices),
        "other_indices" : len(other_indices),
    }
    logger.info(f"Batch distribution: {batch_info}")

    # -------------------------------------------------------------------------
    # 1. Process math (Sequential)
    # -------------------------------------------------------------------------
    if math_indices:
        from verl.utils.reward_score.math_verify import MathVerifier
        verifier = MathVerifier()
        for i in math_indices:
            # Handle standard wrapper for Math tasks
            gt = ground_truths[i]
            if isinstance(gt, list): gt = gt[0] if gt else None
            
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
            gt = ground_truths[i]
            if isinstance(gt, list): gt = gt[0] if gt else None
            
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
                
            # NOTE: We assume ground_truths[i] is already clean (list/dict) from preprocessor
            # We do NOT peel [0] here because code tasks often use the whole list as test cases
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
        # If not provided, try to find one in extra_info of first item
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
            gt = ground_truths[i]
            if isinstance(gt, list): gt = gt[0] if gt else None
            
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

    # Lazy wandb logging for domain-wise rewards
    try:
        import wandb
        if wandb.run is not None:
            wandb_metrics = {}
            for domain, stats in domain_stats.items():
                wandb_metrics[f"train/Dolci-Think-RL/{domain}/reward/mean@1"] = stats["mean"]
                wandb_metrics[f"train/Dolci-Think-RL/{domain}/reward/count"] = stats["count"]
            wandb_metrics["train/Dolci-Think-RL/overall/reward/mean@1"] = overall_mean
            wandb.log(wandb_metrics, commit=False)
    except Exception:
        pass  # Silently ignore if wandb not available or not initialized

    return scores
