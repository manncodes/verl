# Copyright 2025 verl contributors
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
Prime Intellect reward scoring integration.

This module provides compute_score functions that integrate Prime Intellect's
sandbox execution and verifiers environments with verl's reward computation system.

Supported Data Sources:
    - prime_intellect/*: Uses Prime Intellect sandbox for code execution
    - prime_env/*: Uses verifiers environments from the Environments Hub
    - Custom environments: Can be registered dynamically

Usage:
    >>> from verl.utils.prime_intellect import compute_score
    >>> score = compute_score(
    ...     data_source="prime_intellect/code",
    ...     solution_str="def solution(n): return n * 2",
    ...     ground_truth='{"inputs": ["2"], "outputs": ["4"]}',
    ... )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from typing import Any

logger = logging.getLogger(__name__)

# Cache for clients and environments
_client_cache = {}
_environment_cache = {}


def _get_async_client():
    """Get or create a cached async Prime Intellect client."""
    from verl.utils.prime_intellect.client import AsyncPrimeIntellectClient

    if "async" not in _client_cache:
        _client_cache["async"] = AsyncPrimeIntellectClient()
    return _client_cache["async"]


def _get_sync_client():
    """Get or create a cached sync Prime Intellect client."""
    from verl.utils.prime_intellect.client import PrimeIntellectClient

    if "sync" not in _client_cache:
        _client_cache["sync"] = PrimeIntellectClient()
    return _client_cache["sync"]


def _get_environment(name: str):
    """Get or create a cached environment."""
    from verl.utils.prime_intellect.environments import PrimeIntellectEnvironment

    if name not in _environment_cache:
        _environment_cache[name] = PrimeIntellectEnvironment(name)
    return _environment_cache[name]


def _extract_code_from_response(completion: str) -> str:
    """Extract code from a completion that may contain markdown code blocks.

    Args:
        completion: The raw completion string.

    Returns:
        str: The extracted code.
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
    return completion


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict | None = None,
    prime_intellect_api_key: str | None = None,
    timeout: int = 30,
    docker_image: str = "python:3.11-slim",
    **kwargs,
) -> float | dict[str, Any]:
    """Compute reward score using Prime Intellect services.

    This function routes to the appropriate scoring method based on data_source:
    - prime_intellect/*: Uses Prime Intellect sandbox for code execution
    - prime_env/*: Uses verifiers environments from the Environments Hub

    Args:
        data_source: Identifier for the scoring method (e.g., "prime_intellect/code").
        solution_str: The solution/response to evaluate.
        ground_truth: The expected output or test cases.
        extra_info: Additional context for scoring.
        prime_intellect_api_key: API key for Prime Intellect services.
        timeout: Timeout for execution in seconds.
        docker_image: Docker image for sandbox execution.
        **kwargs: Additional arguments passed to scoring functions.

    Returns:
        float or dict: Score (0.0 to 1.0) or dict with score and metadata.

    Example:
        >>> score = compute_score(
        ...     data_source="prime_intellect/code",
        ...     solution_str="print('hello')",
        ...     ground_truth='{"inputs": [""], "outputs": ["hello"]}',
        ... )
    """
    extra_info = extra_info or {}

    # Set API key if provided
    if prime_intellect_api_key:
        os.environ["PRIME_API_KEY"] = prime_intellect_api_key

    try:
        # Route based on data source
        if data_source.startswith("prime_intellect/"):
            task_type = data_source.split("/", 1)[1] if "/" in data_source else "code"
            return _compute_sandbox_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                task_type=task_type,
                timeout=timeout,
                docker_image=docker_image,
                extra_info=extra_info,
            )
        elif data_source.startswith("prime_env/"):
            env_name = data_source.split("/", 1)[1] if "/" in data_source else ""
            return _compute_environment_score(
                env_name=env_name,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            # Try to find a matching environment
            env_name = data_source.replace("_", "/")
            return _compute_environment_score(
                env_name=env_name,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
    except Exception as e:
        logger.error(f"Error computing Prime Intellect score: {e}")
        traceback.print_exc()
        return {"score": 0.0, "error": str(e)}


def _compute_sandbox_score(
    solution_str: str,
    ground_truth: str | dict,
    task_type: str = "code",
    timeout: int = 30,
    docker_image: str = "python:3.11-slim",
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """Compute score using Prime Intellect sandbox execution.

    Args:
        solution_str: The code solution to evaluate.
        ground_truth: Test cases as JSON string or dict.
        task_type: Type of task ("code", "python", etc.).
        timeout: Timeout for execution.
        docker_image: Docker image for the sandbox.
        extra_info: Additional context.

    Returns:
        dict: Score and execution metadata.
    """
    extra_info = extra_info or {}

    # Extract code from potential markdown
    code = _extract_code_from_response(solution_str)

    if not code.strip():
        return {
            "score": 0.0,
            "error": "No valid code found in response",
            "passed": 0,
            "total": 0,
        }

    # Parse test cases
    try:
        if isinstance(ground_truth, str):
            test_cases = json.loads(ground_truth)
        else:
            test_cases = ground_truth
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse test cases: {e}")
        return {"score": 0.0, "error": f"Invalid test cases JSON: {e}"}

    # Handle different test case formats
    if "assert_case" in test_cases:
        # Format: {"assert_case": ["assert func(1) == 2", ...]}
        return _run_assert_tests(code, test_cases["assert_case"], timeout, docker_image)
    elif "inputs" in test_cases and "outputs" in test_cases:
        # Format: {"inputs": ["input1", ...], "outputs": ["output1", ...]}
        return _run_io_tests(code, test_cases, timeout, docker_image)
    elif "test_code" in test_cases:
        # Format: {"test_code": "import unittest; ..."}
        return _run_test_code(code, test_cases["test_code"], timeout, docker_image)
    else:
        return {"score": 0.0, "error": "Unrecognized test case format"}


def _run_assert_tests(
    code: str,
    assert_cases: list[str],
    timeout: int,
    docker_image: str,
) -> dict[str, Any]:
    """Run assertion-based tests."""
    client = _get_sync_client()

    passed = 0
    total = len(assert_cases)
    results = []

    for i, assert_case in enumerate(assert_cases):
        # Combine code with assertion
        test_code = f"{code}\n\n{assert_case}"

        try:
            result = client.execute_code(
                code=test_code,
                docker_image=docker_image,
                timeout=timeout,
            )

            test_passed = result.success
            if test_passed:
                passed += 1

            results.append({
                "case": i,
                "passed": test_passed,
                "exit_code": result.exit_code,
                "stdout": result.stdout[:500] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
            })
        except Exception as e:
            results.append({
                "case": i,
                "passed": False,
                "error": str(e),
            })

    score = passed / total if total > 0 else 0.0

    return {
        "score": score,
        "passed": passed,
        "total": total,
        "results": results,
    }


def _run_io_tests(
    code: str,
    test_cases: dict,
    timeout: int,
    docker_image: str,
) -> dict[str, Any]:
    """Run input/output based tests."""
    client = _get_sync_client()

    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])

    passed = 0
    total = len(inputs)
    results = []

    for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
        # Create test wrapper
        test_code = f"""
{code}

# Test execution
import sys
from io import StringIO

# Capture output
sys.stdin = StringIO({repr(test_input)})
old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    # Try to call main() or run the code
    if 'main' in dir():
        main()
    elif 'solution' in dir():
        result = solution({repr(test_input)})
        if result is not None:
            print(result)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)

# Get output
output = sys.stdout.getvalue().strip()
sys.stdout = old_stdout
print(output)
"""

        try:
            result = client.execute_code(
                code=test_code,
                docker_image=docker_image,
                timeout=timeout,
            )

            actual_output = result.stdout.strip() if result.stdout else ""
            expected = str(expected_output).strip() if expected_output else ""
            test_passed = actual_output == expected

            if test_passed:
                passed += 1

            results.append({
                "case": i,
                "passed": test_passed,
                "input": test_input[:100] if test_input else "",
                "expected": expected[:100],
                "actual": actual_output[:100],
            })
        except Exception as e:
            results.append({
                "case": i,
                "passed": False,
                "error": str(e),
            })

    score = passed / total if total > 0 else 0.0

    return {
        "score": score,
        "passed": passed,
        "total": total,
        "results": results,
    }


def _run_test_code(
    code: str,
    test_code: str,
    timeout: int,
    docker_image: str,
) -> dict[str, Any]:
    """Run custom test code."""
    client = _get_sync_client()

    full_code = f"{code}\n\n{test_code}"

    try:
        result = client.execute_code(
            code=full_code,
            docker_image=docker_image,
            timeout=timeout,
        )

        # Check if tests passed (exit code 0)
        if result.success:
            return {
                "score": 1.0,
                "passed": True,
                "stdout": result.stdout[:1000] if result.stdout else "",
            }
        else:
            return {
                "score": 0.0,
                "passed": False,
                "exit_code": result.exit_code,
                "stderr": result.stderr[:1000] if result.stderr else "",
            }
    except Exception as e:
        return {
            "score": 0.0,
            "passed": False,
            "error": str(e),
        }


def _compute_environment_score(
    env_name: str,
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """Compute score using a Prime Intellect verifiers environment.

    Args:
        env_name: Name of the environment (owner/name format).
        solution_str: The response to evaluate.
        ground_truth: The expected answer.
        extra_info: Additional context including prompt.

    Returns:
        dict: Score and evaluation metadata.
    """
    extra_info = extra_info or {}

    try:
        env = _get_environment(env_name)

        # Get prompt from extra_info if available
        prompt = extra_info.get("prompt", "")
        info = extra_info.get("info", {})

        # Convert ground_truth to string if needed
        answer = ground_truth if isinstance(ground_truth, str) else str(ground_truth)

        # Evaluate
        result = env.evaluate(
            response=solution_str,
            prompt=prompt,
            answer=answer,
            info=info,
        )

        return {
            "score": result.score,
            "passed": result.passed,
            "details": result.details,
            "metadata": result.metadata,
        }
    except Exception as e:
        logger.error(f"Environment evaluation failed: {e}")
        return {
            "score": 0.0,
            "error": str(e),
        }


async def compute_score_async(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict | None = None,
    prime_intellect_api_key: str | None = None,
    timeout: int = 30,
    docker_image: str = "python:3.11-slim",
    **kwargs,
) -> float | dict[str, Any]:
    """Asynchronous version of compute_score.

    This function provides the same functionality as compute_score but
    uses async I/O for better concurrency in high-throughput scenarios.

    Args:
        data_source: Identifier for the scoring method.
        solution_str: The solution/response to evaluate.
        ground_truth: The expected output or test cases.
        extra_info: Additional context for scoring.
        prime_intellect_api_key: API key for Prime Intellect services.
        timeout: Timeout for execution in seconds.
        docker_image: Docker image for sandbox execution.
        **kwargs: Additional arguments.

    Returns:
        float or dict: Score and metadata.
    """
    extra_info = extra_info or {}

    if prime_intellect_api_key:
        os.environ["PRIME_API_KEY"] = prime_intellect_api_key

    try:
        if data_source.startswith("prime_intellect/"):
            task_type = data_source.split("/", 1)[1] if "/" in data_source else "code"
            return await _compute_sandbox_score_async(
                solution_str=solution_str,
                ground_truth=ground_truth,
                task_type=task_type,
                timeout=timeout,
                docker_image=docker_image,
                extra_info=extra_info,
            )
        elif data_source.startswith("prime_env/"):
            env_name = data_source.split("/", 1)[1] if "/" in data_source else ""
            return await _compute_environment_score_async(
                env_name=env_name,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            env_name = data_source.replace("_", "/")
            return await _compute_environment_score_async(
                env_name=env_name,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
    except Exception as e:
        logger.error(f"Error computing Prime Intellect score async: {e}")
        return {"score": 0.0, "error": str(e)}


async def _compute_sandbox_score_async(
    solution_str: str,
    ground_truth: str | dict,
    task_type: str = "code",
    timeout: int = 30,
    docker_image: str = "python:3.11-slim",
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """Async version of sandbox score computation."""
    extra_info = extra_info or {}

    code = _extract_code_from_response(solution_str)

    if not code.strip():
        return {
            "score": 0.0,
            "error": "No valid code found in response",
        }

    try:
        if isinstance(ground_truth, str):
            test_cases = json.loads(ground_truth)
        else:
            test_cases = ground_truth
    except json.JSONDecodeError as e:
        return {"score": 0.0, "error": f"Invalid test cases JSON: {e}"}

    client = _get_async_client()

    if "assert_case" in test_cases:
        return await _run_assert_tests_async(
            client, code, test_cases["assert_case"], timeout, docker_image
        )
    elif "inputs" in test_cases and "outputs" in test_cases:
        return await _run_io_tests_async(
            client, code, test_cases, timeout, docker_image
        )
    else:
        return {"score": 0.0, "error": "Unrecognized test case format"}


async def _run_assert_tests_async(
    client,
    code: str,
    assert_cases: list[str],
    timeout: int,
    docker_image: str,
) -> dict[str, Any]:
    """Async assertion-based tests."""
    async def run_single_test(i: int, assert_case: str):
        test_code = f"{code}\n\n{assert_case}"
        try:
            result = await client.execute_code(
                code=test_code,
                docker_image=docker_image,
                timeout=timeout,
            )
            return {
                "case": i,
                "passed": result.success,
                "exit_code": result.exit_code,
            }
        except Exception as e:
            return {"case": i, "passed": False, "error": str(e)}

    results = await asyncio.gather(*[
        run_single_test(i, case) for i, case in enumerate(assert_cases)
    ])

    passed = sum(1 for r in results if r.get("passed", False))
    total = len(assert_cases)

    return {
        "score": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
        "results": results,
    }


async def _run_io_tests_async(
    client,
    code: str,
    test_cases: dict,
    timeout: int,
    docker_image: str,
) -> dict[str, Any]:
    """Async input/output tests."""
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])

    async def run_single_test(i: int, test_input: str, expected: str):
        test_code = f"""
{code}

import sys
from io import StringIO

sys.stdin = StringIO({repr(test_input)})
old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    if 'main' in dir():
        main()
    elif 'solution' in dir():
        result = solution({repr(test_input)})
        if result is not None:
            print(result)
except Exception as e:
    pass

output = sys.stdout.getvalue().strip()
sys.stdout = old_stdout
print(output)
"""
        try:
            result = await client.execute_code(
                code=test_code,
                docker_image=docker_image,
                timeout=timeout,
            )
            actual = result.stdout.strip() if result.stdout else ""
            expected_str = str(expected).strip() if expected else ""
            return {
                "case": i,
                "passed": actual == expected_str,
            }
        except Exception as e:
            return {"case": i, "passed": False, "error": str(e)}

    results = await asyncio.gather(*[
        run_single_test(i, inp, out)
        for i, (inp, out) in enumerate(zip(inputs, outputs))
    ])

    passed = sum(1 for r in results if r.get("passed", False))
    total = len(inputs)

    return {
        "score": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
        "results": results,
    }


async def _compute_environment_score_async(
    env_name: str,
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """Async environment score computation."""
    extra_info = extra_info or {}

    try:
        env = _get_environment(env_name)
        prompt = extra_info.get("prompt", "")
        info = extra_info.get("info", {})
        answer = ground_truth if isinstance(ground_truth, str) else str(ground_truth)

        result = await env.evaluate_async(
            response=solution_str,
            prompt=prompt,
            answer=answer,
            info=info,
        )

        return {
            "score": result.score,
            "passed": result.passed,
            "details": result.details,
            "metadata": result.metadata,
        }
    except Exception as e:
        logger.error(f"Async environment evaluation failed: {e}")
        return {"score": 0.0, "error": str(e)}
