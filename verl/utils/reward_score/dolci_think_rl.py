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
- math: Uses math_verify (verifiable)
- ifeval: Uses IFEval constraint checking (verifiable)
- code/code_stdio: Uses sandbox_fusion for code execution (BATCHED)
- general-quality/other: Uses basic verifiable matching

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import concurrent.futures
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SANDBOX_TIMEOUT = 10
DEFAULT_MEMORY_LIMIT_MB = 1024
DEFAULT_MAX_WORKERS = max(32, os.cpu_count() * 4 if os.cpu_count() else 32)


def remove_thinking_section(prediction: str) -> str:
    """Remove thinking/reasoning sections from model output before reward computation.

    Strips <think>...</think>, <evaluation>...</evaluation>, and <answer> tags.

    Args:
        prediction: The model's raw output.

    Returns:
        Cleaned output with thinking sections removed.
    """
    if prediction is None:
        return ""
    prediction = prediction.replace("<|assistant|>", "").strip()
    # Remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # Remove evaluation section
    prediction = prediction.split("</evaluation>")[-1]
    # Remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def _basic_string_match(solution_str: str, ground_truth: str) -> float:
    """Basic string matching for fallback.

    Args:
        solution_str: The model's solution (already cleaned).
        ground_truth: The expected answer.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    # Normalize
    answer = solution_str.lower().strip()
    ground_truth = str(ground_truth).lower().strip()

    if answer == ground_truth:
        return 1.0
    if ground_truth in answer:
        return 1.0
    return 0.0


def _normalize_ground_truth(ground_truth: Any) -> Optional[str]:
    """Normalize ground_truth which may be a list or string.

    Args:
        ground_truth: The ground truth value (may be list or string).

    Returns:
        Normalized string or None.
    """
    if ground_truth is None:
        return None
    if isinstance(ground_truth, list):
        return ground_truth[0] if ground_truth else None
    return str(ground_truth)


# =============================================================================
# Math Scoring
# =============================================================================


def _compute_math_scores(
    indices: List[int],
    solutions: List[str],
    ground_truths: List[str],
) -> Dict[int, float]:
    """Compute math scores using math_verify or math_dapo.

    Args:
        indices: Original indices in the batch.
        solutions: List of solution strings.
        ground_truths: List of ground truth answers.

    Returns:
        Dict mapping index to score.
    """
    results = {}

    # Try math_verify first (more accurate), fallback to math_dapo
    math_compute_score = None
    try:
        from verl.utils.reward_score import math_verify

        math_compute_score = math_verify.compute_score
        logger.debug("Using math_verify for math scoring")
    except ImportError:
        try:
            from verl.utils.reward_score import math_dapo

            math_compute_score = math_dapo.compute_score
            logger.debug("math_verify not available, using math_dapo")
        except ImportError:
            logger.error("Neither math_verify nor math_dapo available")

    if math_compute_score is None:
        for idx in indices:
            results[idx] = 0.0
        return results

    for idx, sol, gt in zip(indices, solutions, ground_truths):
        if sol is None or gt is None:
            results[idx] = 0.0
            continue
        try:
            result = math_compute_score(sol, gt)
            if isinstance(result, dict):
                # math_dapo returns {"score": ..., "acc": ..., "pred": ...}
                results[idx] = result.get("score", 0.0)
            else:
                results[idx] = float(result)
        except Exception as e:
            logger.debug(f"Math verification failed for index {idx}: {e}")
            results[idx] = 0.0

    return results


# =============================================================================
# IFEval Scoring (Instruction Following)
# =============================================================================


class IFEvalChecker:
    """Checker for IFEval instruction-following constraints.

    Supports common constraint types from the IFEval benchmark.
    """

    def check_constraint(
        self,
        response: str,
        constraint_type: Optional[str],
        constraint: Optional[str],
        extra_info: Optional[Dict],
    ) -> float:
        """Check if response satisfies the given constraint.

        Args:
            response: The model's response.
            constraint_type: Type of constraint (e.g., "keywords", "length", "format").
            constraint: The constraint specification.
            extra_info: Additional context that may contain constraint details.

        Returns:
            1.0 if constraint satisfied, 0.0 otherwise.
        """
        if not constraint_type or not constraint:
            # No constraint specified, check basic presence
            return 1.0 if response.strip() else 0.0

        constraint_type = constraint_type.lower()
        try:
            # Parse constraint if it's JSON
            if isinstance(constraint, str):
                try:
                    constraint_data = json.loads(constraint)
                except json.JSONDecodeError:
                    constraint_data = constraint
            else:
                constraint_data = constraint

            # Route to appropriate checker
            if "keyword" in constraint_type:
                return self._check_keywords(response, constraint_data)
            elif "length" in constraint_type or "word" in constraint_type:
                return self._check_length(response, constraint_data)
            elif "format" in constraint_type:
                return self._check_format(response, constraint_data)
            elif "start" in constraint_type:
                return self._check_start(response, constraint_data)
            elif "end" in constraint_type:
                return self._check_end(response, constraint_data)
            elif "section" in constraint_type or "bullet" in constraint_type:
                return self._check_sections(response, constraint_data)
            elif "json" in constraint_type:
                return self._check_json(response, constraint_data)
            elif "language" in constraint_type:
                return self._check_language(response, constraint_data)
            else:
                # Unknown constraint type - be lenient
                logger.debug(f"Unknown constraint type: {constraint_type}")
                return 1.0 if response.strip() else 0.0

        except Exception as e:
            logger.debug(f"Constraint check failed: {e}")
            return 0.0

    def _check_keywords(self, response: str, constraint: Any) -> float:
        """Check keyword inclusion/exclusion constraints."""
        response_lower = response.lower()

        if isinstance(constraint, dict):
            include = constraint.get("include", constraint.get("keywords", []))
            exclude = constraint.get("exclude", [])

            # Check all required keywords are present
            if include:
                if isinstance(include, str):
                    include = [include]
                for keyword in include:
                    if keyword.lower() not in response_lower:
                        return 0.0

            # Check no excluded keywords are present
            if exclude:
                if isinstance(exclude, str):
                    exclude = [exclude]
                for keyword in exclude:
                    if keyword.lower() in response_lower:
                        return 0.0

            return 1.0

        elif isinstance(constraint, list):
            # List of keywords to include
            for keyword in constraint:
                if str(keyword).lower() not in response_lower:
                    return 0.0
            return 1.0

        elif isinstance(constraint, str):
            return 1.0 if constraint.lower() in response_lower else 0.0

        return 1.0

    def _check_length(self, response: str, constraint: Any) -> float:
        """Check word/character length constraints."""
        word_count = len(response.split())

        if isinstance(constraint, dict):
            min_words = constraint.get("min_words", constraint.get("min", 0))
            max_words = constraint.get("max_words", constraint.get("max", float("inf")))

            if word_count < min_words or word_count > max_words:
                return 0.0
            return 1.0

        elif isinstance(constraint, int):
            # Assume it's exact word count
            return 1.0 if word_count == constraint else 0.0

        return 1.0

    def _check_format(self, response: str, constraint: Any) -> float:
        """Check format constraints (markdown, lists, etc.)."""
        if isinstance(constraint, dict):
            fmt = constraint.get("format", "").lower()
        else:
            fmt = str(constraint).lower()

        if "markdown" in fmt or "md" in fmt:
            # Check for markdown elements
            has_headers = bool(re.search(r"^#{1,6}\s", response, re.MULTILINE))
            has_lists = bool(re.search(r"^[-*]\s|\d+\.\s", response, re.MULTILINE))
            return 1.0 if (has_headers or has_lists) else 0.0

        elif "bullet" in fmt or "list" in fmt:
            has_bullets = bool(re.search(r"^[-*•]\s", response, re.MULTILINE))
            has_numbers = bool(re.search(r"^\d+[.)]\s", response, re.MULTILINE))
            return 1.0 if (has_bullets or has_numbers) else 0.0

        return 1.0

    def _check_start(self, response: str, constraint: Any) -> float:
        """Check if response starts with specific text."""
        if isinstance(constraint, dict):
            prefix = constraint.get("prefix", constraint.get("start", ""))
        else:
            prefix = str(constraint)

        if prefix:
            return 1.0 if response.strip().lower().startswith(prefix.lower()) else 0.0
        return 1.0

    def _check_end(self, response: str, constraint: Any) -> float:
        """Check if response ends with specific text."""
        if isinstance(constraint, dict):
            suffix = constraint.get("suffix", constraint.get("end", ""))
        else:
            suffix = str(constraint)

        if suffix:
            return 1.0 if response.strip().lower().endswith(suffix.lower()) else 0.0
        return 1.0

    def _check_sections(self, response: str, constraint: Any) -> float:
        """Check section/paragraph count constraints."""
        # Count sections by double newlines or headers
        sections = re.split(r"\n\n+|^#{1,6}\s", response, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]
        section_count = len(sections)

        if isinstance(constraint, dict):
            min_sections = constraint.get("min", 1)
            max_sections = constraint.get("max", float("inf"))
            return 1.0 if min_sections <= section_count <= max_sections else 0.0

        elif isinstance(constraint, int):
            return 1.0 if section_count >= constraint else 0.0

        return 1.0

    def _check_json(self, response: str, constraint: Any) -> float:
        """Check if response contains valid JSON."""
        # Try to extract JSON from response
        json_pattern = r"\{[^{}]*\}|\[[^\[\]]*\]"
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                json.loads(match)
                return 1.0
            except json.JSONDecodeError:
                continue

        # Try parsing entire response as JSON
        try:
            json.loads(response.strip())
            return 1.0
        except json.JSONDecodeError:
            pass

        return 0.0

    def _check_language(self, response: str, constraint: Any) -> float:
        """Check language constraints (basic heuristic)."""
        # This is a simplified check - full language detection would need a library
        if isinstance(constraint, dict):
            lang = constraint.get("language", "").lower()
        else:
            lang = str(constraint).lower()

        if lang in ["english", "en"]:
            # Check for non-ASCII characters (crude check)
            ascii_ratio = sum(1 for c in response if ord(c) < 128) / max(len(response), 1)
            return 1.0 if ascii_ratio > 0.8 else 0.0

        # For other languages, be lenient
        return 1.0


# Global IFEval checker instance
_ifeval_checker = IFEvalChecker()


def _compute_ifeval_scores(
    indices: List[int],
    solutions: List[str],
    ground_truths: List[str],
    extra_infos: List[Optional[Dict]],
) -> Dict[int, float]:
    """Compute IFEval instruction-following scores.

    Args:
        indices: Original indices in the batch.
        solutions: List of solution strings.
        ground_truths: List of ground truth answers (may be None for IF tasks).
        extra_infos: List of extra_info dicts with constraint info.

    Returns:
        Dict mapping index to score.
    """
    results = {}

    for idx, sol, gt, extra_info in zip(indices, solutions, ground_truths, extra_infos):
        if sol is None:
            results[idx] = 0.0
            continue

        # Extract constraint info from extra_info
        constraint_type = None
        constraint = None
        if extra_info and isinstance(extra_info, dict):
            constraint_type = extra_info.get("constraint_type")
            constraint = extra_info.get("constraint")

        # Check the constraint
        score = _ifeval_checker.check_constraint(sol, constraint_type, constraint, extra_info)
        results[idx] = score

    return results


# =============================================================================
# Code Scoring (sandbox_fusion)
# =============================================================================


def _extract_code_from_response(completion: str) -> Optional[str]:
    """Extract code from a model completion.

    Args:
        completion: The model's response potentially containing code blocks.

    Returns:
        Extracted code or None if no valid code found.
    """
    if completion is None:
        return None

    # Try to extract Python code block
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
        return solution.strip()

    # Try generic code block
    if "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():
                    solution = rest
            return solution.strip()

    # Return raw completion if no code blocks found
    return completion.strip()


def _compute_code_scores_sandbox(
    indices: List[int],
    solutions: List[str],
    ground_truths: List[str],
    extra_infos: List[Optional[Dict]],
    sandbox_fusion_url: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
) -> Dict[int, float]:
    """Compute code scores using sandbox_fusion.

    Args:
        indices: Original indices in the batch.
        solutions: List of code solutions.
        ground_truths: List of ground truths (test cases as JSON string).
        extra_infos: List of extra_info dicts with problem context.
        sandbox_fusion_url: URL of the sandbox_fusion service.
        concurrent_semaphore: Semaphore for concurrency control.
        memory_limit_mb: Memory limit for code execution.
        timeout: Timeout for code execution.

    Returns:
        Dict mapping index to score.
    """
    from verl.utils.reward_score.sandbox_fusion import compute_score as sandbox_compute_score

    results = {}

    for idx, sol, gt, extra_info in zip(indices, solutions, ground_truths, extra_infos):
        if sol is None:
            results[idx] = 0.0
            continue

        # Extract code from solution
        code = _extract_code_from_response(sol)
        if not code:
            results[idx] = 0.0
            continue

        # Parse test cases from ground_truth
        test_cases = gt
        if isinstance(gt, str):
            try:
                test_cases = json.loads(gt)
            except json.JSONDecodeError:
                # Try to construct test cases from extra_info
                test_cases = _build_test_cases_from_extra_info(gt, extra_info)

        if not test_cases or not isinstance(test_cases, dict):
            # No valid test cases, fallback to string comparison
            logger.debug(f"No valid test cases for index {idx}, using string match")
            results[idx] = _basic_string_match(sol, str(gt)) if gt else 0.0
            continue

        try:
            score, metadata = sandbox_compute_score(
                sandbox_fusion_url=sandbox_fusion_url,
                concurrent_semaphore=concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
                completion=sol,  # Pass full completion, sandbox_compute_score extracts code
                test_cases=test_cases,
                continuous=True,
                timeout=timeout,
            )
            results[idx] = float(score)
        except Exception as e:
            logger.warning(f"Sandbox execution failed for index {idx}: {e}")
            results[idx] = 0.0

    return results


def _build_test_cases_from_extra_info(
    ground_truth: str,
    extra_info: Optional[Dict],
) -> Optional[Dict]:
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
# Other/Fallback Scoring
# =============================================================================


def _compute_other_scores(
    indices: List[int],
    solutions: List[str],
    ground_truths: List[str],
) -> Dict[int, float]:
    """Compute scores for other/unknown dataset sources.

    Uses math verification first, then falls back to string matching.

    Args:
        indices: Original indices in the batch.
        solutions: List of solution strings.
        ground_truths: List of ground truth answers.

    Returns:
        Dict mapping index to score.
    """
    results = {}

    for idx, sol, gt in zip(indices, solutions, ground_truths):
        if sol is None or gt is None:
            results[idx] = 0.0
            continue

        # Try math verification first
        try:
            from verl.utils.reward_score import math_dapo

            result = math_dapo.compute_score(sol, gt)
            score = result.get("score", 0.0) if isinstance(result, dict) else float(result)
            if score > 0:
                results[idx] = score
                continue
        except Exception:
            pass

        # Fallback to basic string matching
        results[idx] = _basic_string_match(sol, gt)

    return results


# =============================================================================
# Main Interface
# =============================================================================


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    **kwargs,
) -> float:
    """Compute the score for a single Dolci-Think-RL solution.

    NOTE: For batch processing, prefer using compute_score_batch() for efficiency,
    especially for code tasks.

    Routes to appropriate reward function based on dataset_source in extra_info:
    - math: math_verify (verifiable)
    - ifeval: IFEval constraint checking (verifiable)
    - code/code_stdio: sandbox_fusion code execution
    - other: math verification fallback then string matching

    Args:
        solution_str: The model's generated solution.
        ground_truth: The expected ground truth answer.
        extra_info: Dict with 'dataset_source' to determine scoring method.
        sandbox_fusion_url: URL of sandbox service for code tasks.
        concurrent_semaphore: Semaphore for concurrency control.
        memory_limit_mb: Memory limit for code execution.
        timeout: Timeout for code execution.
        **kwargs: Additional arguments.

    Returns:
        Score as a float (typically 0.0 to 1.0, may be -1.0 to 1.0 for math).
    """
    # Use batch function with single item
    scores = compute_score_batch(
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        sandbox_fusion_url=sandbox_fusion_url,
        concurrent_semaphore=concurrent_semaphore,
        memory_limit_mb=memory_limit_mb,
        timeout=timeout,
        **kwargs,
    )
    return scores[0]


def compute_score_batch(
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: Optional[List[Optional[Dict]]] = None,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    **kwargs,
) -> List[float]:
    """Compute scores for a batch of Dolci-Think-RL solutions.

    Groups samples by dataset_source and processes each group IN PARALLEL:
    - math: Processed with math_verify
    - ifeval: Processed with IFEval constraint checker
    - code/code_stdio: Batched together for sandbox_fusion execution
    - other: Processed with math verification fallback

    Args:
        solution_strs: List of model solutions.
        ground_truths: List of ground truth answers.
        extra_infos: List of extra_info dicts with 'dataset_source'.
        sandbox_fusion_url: URL of sandbox service for code tasks.
        concurrent_semaphore: Semaphore for sandbox concurrency control.
        memory_limit_mb: Memory limit for code execution.
        timeout: Timeout for code execution.
        max_workers: Maximum number of parallel workers for group processing.
        **kwargs: Additional arguments.

    Returns:
        List of scores (same order as input).
    """
    n = len(solution_strs)
    if n == 0:
        return []

    if extra_infos is None:
        extra_infos = [None] * n

    logger.info(f"compute_score_batch called with {n} samples")

    # Remove thinking sections from all solutions upfront
    cleaned_solutions = [remove_thinking_section(s) if s else "" for s in solution_strs]

    # Normalize ground truths
    normalized_gts = [_normalize_ground_truth(gt) for gt in ground_truths]

    # Initialize results
    scores = [0.0] * n

    # Group indices by dataset_source
    groups: Dict[str, Tuple[List[int], List[str], List[str], List[Optional[Dict]]]] = {
        "math": ([], [], [], []),
        "ifeval": ([], [], [], []),
        "code": ([], [], [], []),
        "other": ([], [], [], []),
    }

    for i in range(n):
        extra_info = extra_infos[i] or {}
        dataset_source = extra_info.get("dataset_source", "").lower()

        if dataset_source == "math":
            group_key = "math"
        elif dataset_source == "ifeval":
            group_key = "ifeval"
        elif dataset_source in ("code", "code_stdio"):
            group_key = "code"
        else:
            group_key = "other"

        indices, sols, gts, extras = groups[group_key]
        indices.append(i)
        sols.append(cleaned_solutions[i])
        gts.append(normalized_gts[i])
        extras.append(extra_infos[i])

    # Log group sizes
    for key, (indices, _, _, _) in groups.items():
        if indices:
            logger.info(f"  {key}: {len(indices)} samples")

    # Process groups in parallel using ThreadPoolExecutor
    def process_math_group():
        indices, sols, gts, _ = groups["math"]
        if not indices:
            return {}
        return _compute_math_scores(indices, sols, gts)

    def process_ifeval_group():
        indices, sols, gts, extras = groups["ifeval"]
        if not indices:
            return {}
        return _compute_ifeval_scores(indices, sols, gts, extras)

    def process_code_group():
        indices, sols, gts, extras = groups["code"]
        if not indices:
            return {}
        if sandbox_fusion_url:
            return _compute_code_scores_sandbox(
                indices,
                sols,
                gts,
                extras,
                sandbox_fusion_url=sandbox_fusion_url,
                concurrent_semaphore=concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
                timeout=timeout,
            )
        else:
            # Fallback to string matching if no sandbox URL
            logger.warning("No sandbox_fusion_url provided for code tasks, using string match")
            results = {}
            for idx, sol, gt in zip(indices, sols, gts):
                results[idx] = _basic_string_match(sol, str(gt)) if sol and gt else 0.0
            return results

    def process_other_group():
        indices, sols, gts, _ = groups["other"]
        if not indices:
            return {}
        return _compute_other_scores(indices, sols, gts)

    # Execute all groups in parallel
    group_results: Dict[int, float] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, max_workers)) as executor:
        futures = {
            executor.submit(process_math_group): "math",
            executor.submit(process_ifeval_group): "ifeval",
            executor.submit(process_code_group): "code",
            executor.submit(process_other_group): "other",
        }

        for future in concurrent.futures.as_completed(futures):
            group_name = futures[future]
            try:
                result = future.result()
                group_results.update(result)
                logger.debug(f"Completed processing {group_name} group: {len(result)} results")
            except Exception as e:
                logger.error(f"Error processing {group_name} group: {e}")
                # Mark failed group indices as 0.0
                indices = groups[group_name][0]
                for idx in indices:
                    group_results[idx] = 0.0

    # Assemble final scores in original order
    for i in range(n):
        scores[i] = group_results.get(i, 0.0)

    return scores


# =============================================================================
# Convenience functions for direct access
# =============================================================================


def compute_math_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> float:
    """Compute math score directly using math_verify.

    Args:
        solution_str: The model's solution.
        ground_truth: The expected answer.

    Returns:
        Score as float.
    """
    solution_str = remove_thinking_section(solution_str)
    results = _compute_math_scores([0], [solution_str], [ground_truth])
    return results.get(0, 0.0)


def compute_ifeval_score(
    solution_str: str,
    constraint_type: Optional[str] = None,
    constraint: Optional[str] = None,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """Compute IFEval constraint score directly.

    Args:
        solution_str: The model's response.
        constraint_type: Type of constraint.
        constraint: The constraint specification.
        extra_info: Additional context.

    Returns:
        Score as float (1.0 if constraint satisfied, 0.0 otherwise).
    """
    solution_str = remove_thinking_section(solution_str)
    if extra_info is None:
        extra_info = {}
    if constraint_type:
        extra_info["constraint_type"] = constraint_type
    if constraint:
        extra_info["constraint"] = constraint

    results = _compute_ifeval_scores([0], [solution_str], [None], [extra_info])
    return results.get(0, 0.0)


def compute_code_score(
    solution_str: str,
    test_cases: Any,
    sandbox_fusion_url: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    **kwargs,
) -> Tuple[float, List[Dict]]:
    """Compute code score directly using sandbox_fusion.

    Args:
        solution_str: The model's code solution.
        test_cases: Test cases (dict or JSON string with 'inputs'/'outputs').
        sandbox_fusion_url: URL of the sandbox service.
        concurrent_semaphore: Semaphore for concurrency control.
        memory_limit_mb: Memory limit for execution.
        timeout: Timeout for execution.

    Returns:
        Tuple of (score, metadata_list).
    """
    from verl.utils.reward_score.sandbox_fusion import compute_score as sandbox_compute_score

    solution_str = remove_thinking_section(solution_str)

    return sandbox_compute_score(
        sandbox_fusion_url=sandbox_fusion_url,
        concurrent_semaphore=concurrent_semaphore,
        memory_limit_mb=memory_limit_mb,
        completion=solution_str,
        test_cases=test_cases,
        continuous=True,
        timeout=timeout,
    )
