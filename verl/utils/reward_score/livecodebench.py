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
LiveCodeBench reward scoring with SandboxFusion support.

LiveCodeBench (https://github.com/LiveCodeBench/LiveCodeBench) is a benchmark for
holistic, contamination-free evaluation of LLM coding capabilities, collecting
competitive programming problems from LeetCode, AtCoder, and CodeForces.

This module supports two execution backends:
  - SandboxFusion (remote): Secure sandboxed code execution via HTTP API.
    Requires a running SandboxFusion service (https://github.com/bytedance/SandboxFusion).
  - prime_code (local): In-process execution using multiprocessing with safety guards.
"""

import base64
import json
import logging
import pickle
import zlib

logger = logging.getLogger(__name__)


def _decompress_test_cases(test_cases_raw):
    """Decompress LiveCodeBench test cases from the compressed format.

    LiveCodeBench stores test cases compressed as:
        base64 -> zlib -> pickle -> json string

    Args:
        test_cases_raw: Either a JSON string, a dict, or a compressed binary string.

    Returns:
        A dict with keys 'inputs', 'outputs', and optionally 'fn_name'.
    """
    if isinstance(test_cases_raw, dict):
        return test_cases_raw

    # Try parsing as plain JSON first
    try:
        parsed = json.loads(test_cases_raw)
        if isinstance(parsed, dict) and "inputs" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Try decompressing from the compressed format
    try:
        decompressed = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(test_cases_raw.encode("utf-8"))))
        )
        return decompressed
    except Exception as e:
        logger.error(f"Failed to decompress LiveCodeBench test cases: {e}")
        raise ValueError(f"Cannot parse LiveCodeBench test cases: {e}") from e


def _extract_code(completion):
    """Extract Python code from an LLM completion string.

    Handles markdown code blocks (```python ... ``` and ``` ... ```).

    Args:
        completion: The raw LLM completion string.

    Returns:
        The extracted code string, or None if no code block is found.
    """
    if "```python" in completion:
        return completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if "\n" in code:
                first_line, rest = code.split("\n", 1)
                if first_line.strip().isalpha():
                    return rest
            return code
    return None


def compute_score(
    solution_str,
    ground_truth,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=1024,
    timeout=10,
    continuous=False,
):
    """Compute score for a LiveCodeBench problem.

    Routes to SandboxFusion (remote) or prime_code (local) based on whether
    a sandbox_fusion_url is provided.

    Args:
        solution_str: The LLM completion containing the code solution.
        ground_truth: The test cases (compressed string, JSON string, or dict).
        sandbox_fusion_url: If provided, use SandboxFusion for remote execution.
        concurrent_semaphore: Semaphore for controlling concurrent sandbox requests.
        memory_limit_mb: Memory limit per sandbox process in MB.
        timeout: Execution timeout per test case in seconds.
        continuous: If True, score based on first N test cases (partial credit).

    Returns:
        float: Score from 0.0 to 1.0, or a tuple (score, metadata) when using SandboxFusion.
    """
    # Decompress test cases
    try:
        in_outs = _decompress_test_cases(ground_truth)
    except ValueError:
        return 0.0

    if sandbox_fusion_url:
        return _compute_score_sandbox_fusion(
            solution_str=solution_str,
            in_outs=in_outs,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
            timeout=timeout,
            continuous=continuous,
        )
    else:
        return _compute_score_local(
            solution_str=solution_str,
            in_outs=in_outs,
            timeout=timeout,
        )


def _compute_score_sandbox_fusion(
    solution_str,
    in_outs,
    sandbox_fusion_url,
    concurrent_semaphore=None,
    memory_limit_mb=1024,
    timeout=10,
    continuous=False,
):
    """Evaluate code using SandboxFusion remote sandbox."""
    from .sandbox_fusion.utils import check_correctness

    solution = _extract_code(solution_str)
    if solution is None:
        return 0.0, [{"error": "No code block found in completion"}]

    try:
        res_list, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_fusion_url,
            in_outs=in_outs,
            generation=solution,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            language="python",
            concurrent_semaphore=concurrent_semaphore,
        )

        if not res_list:
            return 0.0, metadata_list

        if continuous:
            num_to_consider = min(len(res_list), 10)
            if num_to_consider == 0:
                return 0.0, metadata_list
            passed = sum(1 for r in res_list[:num_to_consider] if r is True)
            score = passed / num_to_consider
        else:
            passed = sum(1 for r in res_list if r is True)
            score = passed / len(res_list) if res_list else 0.0

        return float(score), metadata_list
    except Exception as e:
        logger.error(f"SandboxFusion evaluation failed: {e}")
        return 0.0, [{"error": str(e)}]


def _compute_score_local(solution_str, in_outs, timeout=6):
    """Evaluate code using local prime_code execution (fallback)."""
    from .prime_code.testing_util import run_test

    import multiprocessing

    solution = solution_str.split("```python")[-1].split("```")[0]

    def _run(in_outs, generation, result, metadata_list, timeout):
        res, metadata = run_test(in_outs, test=generation, debug=False, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_run,
        args=(in_outs, solution, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
    if p.is_alive():
        p.kill()

    if not result:
        return False

    try:
        return all(r is True for r in result[0])
    except Exception:
        return False
