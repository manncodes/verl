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
- code/code_stdio: Uses sandbox_fusion for code execution (PARALLEL)
- general-quality/general-quality_ref: Uses LLM-as-a-judge (PARALLEL)
- other: Falls back to math verification then string matching

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import concurrent.futures
import json
import logging
import os
import re
import threading
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SANDBOX_TIMEOUT = 10
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_MAX_WORKERS = max(32, os.cpu_count() * 4 if os.cpu_count() else 32)

# LLM Judge configuration
DEFAULT_LLM_JUDGE_URL = os.environ.get(
    "LLM_JUDGE_URL",
    "http://qpn744-vllm-gptoss120b-svc.llm-pretraining.svc.cluster.local:8000/v1",
)
DEFAULT_LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "openai/gpt-oss-120b")


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


# =============================================================================
# LLM-as-a-Judge for General Quality
# =============================================================================

LLM_JUDGE_PROMPT = """You are an expert evaluator. Compare the model's response to the reference answer and rate the quality.

**Problem/Prompt:**
{prompt}

**Reference Answer:**
{reference}

**Model Response:**
{response}

Rate the model's response on a scale of 0.0 to 1.0:
- 1.0: Excellent - response is as good as or better than the reference
- 0.8: Good - response is mostly correct with minor issues
- 0.5: Partial - response has some correct elements but significant gaps
- 0.2: Poor - response is mostly incorrect or unhelpful
- 0.0: Wrong - response is completely incorrect or harmful

Respond with ONLY a single number between 0.0 and 1.0. No explanation."""


def _call_llm_judge(
    prompt: str,
    response: str,
    reference: str,
    llm_judge_url: str,
    llm_judge_model: str,
    timeout: int = 30,
) -> float:
    """Call LLM judge API to score a response.

    Args:
        prompt: The original problem/prompt.
        response: The model's response.
        reference: The reference/ground truth answer.
        llm_judge_url: URL of the LLM judge API.
        llm_judge_model: Model name for the judge.
        timeout: Request timeout in seconds.

    Returns:
        Score as float (0.0 to 1.0).
    """
    import requests

    judge_prompt = LLM_JUDGE_PROMPT.format(
        prompt=prompt[:2000] if prompt else "N/A",
        reference=reference[:2000] if reference else "N/A",
        response=response[:2000] if response else "N/A",
    )

    try:
        payload = {
            "model": llm_judge_model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        api_url = llm_judge_url.rstrip("/") + "/chat/completions"
        resp = requests.post(
            api_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

        result = resp.json()
        content = result["choices"][0]["message"]["content"].strip()

        # Parse score from response
        try:
            score = float(content)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except ValueError:
            # Try to extract first number from response
            match = re.search(r"(\d+\.?\d*)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            logger.warning(f"Could not parse LLM judge response: {content}")
            return 0.5  # Default to middle score

    except Exception as e:
        logger.warning(f"LLM judge call failed: {e}")
        return 0.0  # Return 0 on failure


def _compute_llm_judge_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    **kwargs,
) -> float:
    """Compute score using LLM-as-a-judge.

    Args:
        solution_str: Model's response.
        ground_truth: Reference answer.
        extra_info: Extra info with 'original_prompt' or 'problem'.
        llm_judge_url: LLM judge API URL.
        llm_judge_model: LLM judge model name.

    Returns:
        Score as float.
    """
    llm_judge_url = llm_judge_url or DEFAULT_LLM_JUDGE_URL
    llm_judge_model = llm_judge_model or DEFAULT_LLM_JUDGE_MODEL

    if not llm_judge_url:
        logger.warning("No LLM judge URL provided, falling back to string match")
        return _basic_string_match(solution_str, str(ground_truth))

    # Extract prompt from extra_info
    prompt = ""
    if extra_info and isinstance(extra_info, dict):
        prompt = extra_info.get("original_prompt", "") or extra_info.get("problem", "")

    return _call_llm_judge(
        prompt=prompt,
        response=solution_str,
        reference=str(ground_truth) if ground_truth else "",
        llm_judge_url=llm_judge_url,
        llm_judge_model=llm_judge_model,
    )


# =============================================================================
# Code Scoring with sandbox_fusion
# =============================================================================


def _compute_code_score_sandbox(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
) -> float:
    """Compute code score using sandbox_fusion.

    Args:
        solution_str: The model's code solution.
        ground_truth: Test cases (dict or JSON string with 'inputs'/'outputs').
        extra_info: Additional context.
        sandbox_fusion_url: URL of the sandbox service.
        concurrent_semaphore: Semaphore for concurrency control.
        memory_limit_mb: Memory limit for execution.
        timeout: Timeout for execution.

    Returns:
        Score as float (0.0 to 1.0).
    """
    if not sandbox_fusion_url:
        logger.warning("No sandbox_fusion_url provided for code task, falling back to string match")
        return _basic_string_match(solution_str, str(ground_truth)) if ground_truth else 0.0

    # Parse test cases from ground_truth
    test_cases = ground_truth
    if isinstance(ground_truth, str):
        try:
            test_cases = json.loads(ground_truth)
        except json.JSONDecodeError:
            # Try to construct test cases from extra_info
            test_cases = _build_test_cases_from_extra_info(ground_truth, extra_info)

    if not test_cases or not isinstance(test_cases, dict):
        logger.debug("No valid test cases for code scoring, using string match")
        return _basic_string_match(solution_str, str(ground_truth)) if ground_truth else 0.0

    try:
        from verl.utils.reward_score import sandbox_fusion

        score, metadata = sandbox_fusion.compute_score(
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


def _build_test_cases_from_extra_info(
    ground_truth: str,
    extra_info: Optional[dict],
) -> Optional[dict]:
    """Build test cases dict from ground_truth and extra_info.

    Args:
        ground_truth: The ground truth string (may be expected output).
        extra_info: Extra info that may contain input/output examples.

    Returns:
        Test cases dict with 'inputs' and 'outputs' keys, or None.
    """
    if not extra_info:
        return None

    # Try to extract test cases from common field names
    inputs = extra_info.get("inputs", extra_info.get("input", []))
    outputs = extra_info.get("outputs", extra_info.get("output", [ground_truth] if ground_truth else []))

    if not isinstance(inputs, list):
        inputs = [inputs] if inputs else [""]
    if not isinstance(outputs, list):
        outputs = [outputs] if outputs else []

    # Ensure we have at least one test case
    if not outputs and ground_truth:
        outputs = [ground_truth]
    if not inputs:
        inputs = [""] * len(outputs)

    if outputs:
        return {"inputs": inputs, "outputs": outputs}

    return None


# =============================================================================
# Main Scoring Functions
# =============================================================================


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    **kwargs,
) -> float:
    """Compute the score for a single Dolci-Think-RL solution.

    NOTE: For batch processing, prefer using compute_score_batch() for efficiency.

    Routes to appropriate existing reward function based on dataset_source:
    - math: math_verify.MathVerifier (verifiable)
    - code/code_stdio: sandbox_fusion code execution
    - general-quality/general-quality_ref: LLM-as-a-judge
    - other: Falls back to math then string matching

    Args:
        solution_str: The model's generated solution.
        ground_truth: The expected ground truth answer.
        extra_info: Dict with 'dataset_source' to determine scoring method.
        sandbox_fusion_url: URL of sandbox service for code tasks.
        concurrent_semaphore: Semaphore for sandbox concurrency control.
        memory_limit_mb: Memory limit for code execution.
        timeout: Timeout for code execution.
        llm_judge_url: URL of LLM judge API for general quality.
        llm_judge_model: Model name for LLM judge.
        **kwargs: Additional arguments.

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
        # Check for URLs in extra_info
        if sandbox_fusion_url is None:
            sandbox_fusion_url = extra_info.get("sandbox_fusion_url")
        if llm_judge_url is None:
            llm_judge_url = extra_info.get("llm_judge_url")

    # Route to appropriate existing reward function
    if dataset_source == "math":
        from verl.utils.reward_score.math_verify import MathVerifier

        verifier = MathVerifier()
        result = verifier.compute_score(solution_str, ground_truth)
        return result["score"]

    elif dataset_source == "code" or dataset_source == "code_stdio":
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
    ground_truths: List[Any],
    extra_infos: Optional[List[dict]] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    **kwargs,
) -> List[float]:
    """Compute scores for a batch of Dolci-Think-RL solutions.

    Groups samples by dataset_source and processes EFFICIENTLY IN PARALLEL:
    - math: processed individually (already fast)
    - code/code_stdio: processed in PARALLEL with sandbox_fusion
    - general-quality/general-quality_ref: processed in PARALLEL with LLM judge
    - other: Falls back to math then string matching

    Args:
        solution_strs: List of model solutions.
        ground_truths: List of ground truth answers.
        extra_infos: List of extra_info dicts with 'dataset_source'.
        sandbox_fusion_url: URL of sandbox service for code tasks.
        concurrent_semaphore: Semaphore for sandbox concurrency control.
        memory_limit_mb: Memory limit for code execution.
        timeout: Timeout for code execution.
        llm_judge_url: URL of LLM judge API for general quality.
        llm_judge_model: Model name for LLM judge.
        max_workers: Max parallel workers for code/llm-judge tasks.
        **kwargs: Additional arguments.

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
    code_indices = []
    general_quality_indices = []
    other_indices = []

    for i in range(n):
        extra_info = extra_infos[i] or {}
        dataset_source = extra_info.get("dataset_source", "").lower()

        if dataset_source == "math":
            math_indices.append(i)
        elif dataset_source == "code" or dataset_source == "code_stdio":
            code_indices.append(i)
        elif dataset_source in ("general-quality", "general-quality_ref"):
            general_quality_indices.append(i)
        else:
            other_indices.append(i)

    # Log group sizes
    if math_indices:
        logger.info(f"  math: {len(math_indices)} samples")
    if code_indices:
        logger.info(f"  code: {len(code_indices)} samples (parallel)")
    if general_quality_indices:
        logger.info(f"  general-quality: {len(general_quality_indices)} samples (parallel)")
    if other_indices:
        logger.info(f"  other: {len(other_indices)} samples")

    # Process math (verifiable, fast - sequential is fine)
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

    # Process code with sandbox_fusion - PARALLEL
    if code_indices:
        # Get sandbox URL from first extra_info if not provided
        if sandbox_fusion_url is None:
            for i in code_indices:
                if extra_infos[i] and extra_infos[i].get("sandbox_fusion_url"):
                    sandbox_fusion_url = extra_infos[i].get("sandbox_fusion_url")
                    break

        def _score_code_task(i: int) -> tuple[int, float]:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else ""
            if solution_strs[i] is None:
                return i, 0.0
            try:
                score = _compute_code_score_sandbox(
                    solution_str=solution_strs[i],
                    ground_truth=gt,
                    extra_info=extra_infos[i],
                    sandbox_fusion_url=sandbox_fusion_url,
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
            for future in concurrent.futures.as_completed(futures):
                idx, score = future.result()
                scores[idx] = score

    # Process general-quality with LLM judge - PARALLEL
    if general_quality_indices:
        # Get LLM judge URL from first extra_info if not provided
        if llm_judge_url is None:
            for i in general_quality_indices:
                if extra_infos[i] and extra_infos[i].get("llm_judge_url"):
                    llm_judge_url = extra_infos[i].get("llm_judge_url")
                    break
            if llm_judge_url is None:
                llm_judge_url = DEFAULT_LLM_JUDGE_URL

        if llm_judge_model is None:
            llm_judge_model = DEFAULT_LLM_JUDGE_MODEL

        def _score_llm_judge_task(i: int) -> tuple[int, float]:
            gt = ground_truths[i]
            if isinstance(gt, list):
                gt = gt[0] if gt else ""
            if solution_strs[i] is None:
                return i, 0.0
            try:
                score = _compute_llm_judge_score(
                    solution_str=solution_strs[i],
                    ground_truth=gt,
                    extra_info=extra_infos[i],
                    llm_judge_url=llm_judge_url,
                    llm_judge_model=llm_judge_model,
                )
                return i, score
            except Exception as e:
                logger.debug(f"LLM judge scoring failed for index {i}: {e}")
                return i, 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_score_llm_judge_task, i) for i in general_quality_indices]
            for future in concurrent.futures.as_completed(futures):
                idx, score = future.result()
                scores[idx] = score

    # Process other (fallback) - sequential
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
