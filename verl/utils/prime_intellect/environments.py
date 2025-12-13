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
Prime Intellect environments integration.

This module provides integration with Prime Intellect's verifiers library
and the Environments Hub for loading RL environments.

The verifiers library provides modular components for creating RL environments
that can be used for LLM evaluation, synthetic data pipelines, or agent training.

Environment Types:
    - SingleTurnEnv: Single-response tasks with rubric-based scoring
    - ToolEnv: Agentic loops with native tool-calling
    - StatefulToolEnv: Dynamic state management during execution
    - SandboxEnv: Long-running sandboxed code execution
    - MultiTurnEnv: Custom multi-turn interaction protocols

Usage:
    >>> from verl.utils.prime_intellect import load_environment
    >>> env = load_environment("will/wordle")
    >>> # Use env for evaluation or training
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RubricResult:
    """Result from evaluating a response against a rubric.

    Attributes:
        score: The computed score (0.0 to 1.0).
        passed: Whether the response passed the rubric criteria.
        details: Additional details about the scoring.
        metadata: Extra metadata from the rubric evaluation.
    """

    score: float = 0.0
    passed: bool = False
    details: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Configuration for a Prime Intellect environment.

    Attributes:
        name: Name of the environment (e.g., "will/wordle").
        env_type: Type of environment (e.g., "SingleTurnEnv", "ToolEnv").
        dataset_name: Hugging Face dataset used by the environment.
        rubric_weights: Weights for different rubric components.
        max_turns: Maximum turns for multi-turn environments.
        timeout: Timeout for execution in seconds.
        sandbox_config: Configuration for sandbox execution.
    """

    name: str = ""
    env_type: str = "SingleTurnEnv"
    dataset_name: str | None = None
    rubric_weights: dict[str, float] = field(default_factory=dict)
    max_turns: int = 10
    timeout: int = 60
    sandbox_config: dict[str, Any] = field(default_factory=dict)


class PrimeIntellectEnvironment:
    """Wrapper for Prime Intellect verifiers environments.

    This class provides a unified interface for working with environments
    from the Prime Intellect Environments Hub.

    Example:
        >>> env = PrimeIntellectEnvironment("will/wordle")
        >>> result = env.evaluate(response="CRANE")
        >>> print(result.score)

    Args:
        name: Name of the environment (owner/name format).
        config: Optional environment configuration.
        api_key: Prime Intellect API key for hub access.
    """

    def __init__(
        self,
        name: str,
        config: EnvironmentConfig | None = None,
        api_key: str | None = None,
    ):
        self.name = name
        self.config = config or EnvironmentConfig(name=name)
        self.api_key = api_key or os.environ.get("PRIME_API_KEY")
        self._verifiers_env = None
        self._rubric = None
        self._dataset = None
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure the environment is initialized."""
        if self._initialized:
            return

        try:
            # Try to import verifiers library
            import verifiers as vf

            self._verifiers_env = vf.load_environment(self.name)
            self._initialized = True
            logger.info(f"Loaded verifiers environment: {self.name}")
        except ImportError:
            logger.warning(
                "verifiers library not installed. "
                "Install with: pip install verifiers"
            )
            # Create a mock environment that works without verifiers
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load environment {self.name}: {e}")
            self._initialized = True

    @property
    def rubric(self) -> Callable | None:
        """Get the rubric function for this environment."""
        self._ensure_initialized()
        if self._verifiers_env is not None:
            return getattr(self._verifiers_env, "rubric", None)
        return None

    @property
    def dataset(self):
        """Get the dataset associated with this environment."""
        self._ensure_initialized()
        if self._verifiers_env is not None:
            return getattr(self._verifiers_env, "dataset", None)
        return None

    def evaluate(
        self,
        response: str,
        prompt: str | None = None,
        answer: str | None = None,
        info: dict | None = None,
        **kwargs,
    ) -> RubricResult:
        """Evaluate a response using the environment's rubric.

        Args:
            response: The model's response to evaluate.
            prompt: The original prompt (if applicable).
            answer: The ground truth answer (if applicable).
            info: Additional info dictionary.
            **kwargs: Additional arguments passed to the rubric.

        Returns:
            RubricResult: The evaluation result with score and details.
        """
        self._ensure_initialized()

        if self._verifiers_env is None:
            return RubricResult(
                score=0.0,
                passed=False,
                details="Environment not initialized (verifiers not installed)",
            )

        try:
            # Use the verifiers evaluation API
            if hasattr(self._verifiers_env, "evaluate_single"):
                result = self._verifiers_env.evaluate_single(
                    completion=response,
                    prompt=prompt,
                    answer=answer,
                    info=info or {},
                    **kwargs,
                )
            elif hasattr(self._verifiers_env, "rubric"):
                # Call rubric directly
                rubric = self._verifiers_env.rubric
                result = rubric(
                    completion=response,
                    prompt=prompt,
                    answer=answer,
                    info=info or {},
                    **kwargs,
                )
            else:
                return RubricResult(
                    score=0.0,
                    passed=False,
                    details="Environment has no evaluation method",
                )

            # Parse the result
            if isinstance(result, (int, float)):
                score = float(result)
                return RubricResult(score=score, passed=score > 0.5)
            elif isinstance(result, dict):
                score = result.get("score", result.get("reward", 0.0))
                return RubricResult(
                    score=float(score),
                    passed=result.get("passed", score > 0.5),
                    details=result.get("details", ""),
                    metadata=result,
                )
            else:
                return RubricResult(
                    score=float(result) if result else 0.0,
                    passed=bool(result),
                )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return RubricResult(
                score=0.0,
                passed=False,
                details=f"Evaluation error: {e}",
            )

    async def evaluate_async(
        self,
        response: str,
        prompt: str | None = None,
        answer: str | None = None,
        info: dict | None = None,
        **kwargs,
    ) -> RubricResult:
        """Asynchronously evaluate a response.

        This method uses the verifiers async evaluation API when available.

        Args:
            response: The model's response to evaluate.
            prompt: The original prompt.
            answer: The ground truth answer.
            info: Additional info dictionary.
            **kwargs: Additional arguments.

        Returns:
            RubricResult: The evaluation result.
        """
        self._ensure_initialized()

        if self._verifiers_env is None:
            return RubricResult(
                score=0.0,
                passed=False,
                details="Environment not initialized",
            )

        try:
            # Check for async evaluation method
            if hasattr(self._verifiers_env, "evaluate_single_async"):
                result = await self._verifiers_env.evaluate_single_async(
                    completion=response,
                    prompt=prompt,
                    answer=answer,
                    info=info or {},
                    **kwargs,
                )
            else:
                # Fall back to sync evaluation
                return self.evaluate(response, prompt, answer, info, **kwargs)

            # Parse result (same as sync)
            if isinstance(result, (int, float)):
                score = float(result)
                return RubricResult(score=score, passed=score > 0.5)
            elif isinstance(result, dict):
                score = result.get("score", result.get("reward", 0.0))
                return RubricResult(
                    score=float(score),
                    passed=result.get("passed", score > 0.5),
                    details=result.get("details", ""),
                    metadata=result,
                )
            else:
                return RubricResult(
                    score=float(result) if result else 0.0,
                    passed=bool(result),
                )
        except Exception as e:
            logger.error(f"Async evaluation failed: {e}")
            return RubricResult(
                score=0.0,
                passed=False,
                details=f"Evaluation error: {e}",
            )

    def get_sample(self, index: int = 0) -> dict[str, Any]:
        """Get a sample from the environment's dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            dict: Sample containing prompt, answer, and info.
        """
        self._ensure_initialized()

        if self._verifiers_env is None or self.dataset is None:
            return {"prompt": "", "answer": "", "info": {}}

        try:
            sample = self.dataset[index]
            return {
                "prompt": sample.get("prompt", sample.get("question", "")),
                "answer": sample.get("answer", ""),
                "info": sample.get("info", {}),
            }
        except Exception as e:
            logger.error(f"Failed to get sample: {e}")
            return {"prompt": "", "answer": "", "info": {}}

    def __repr__(self) -> str:
        return f"PrimeIntellectEnvironment(name={self.name!r})"


# Environment cache for reuse
_environment_cache: dict[str, PrimeIntellectEnvironment] = {}


def load_environment(
    name: str,
    cache: bool = True,
    **kwargs,
) -> PrimeIntellectEnvironment:
    """Load a Prime Intellect environment by name.

    This function provides a convenient way to load environments from
    the Prime Intellect Environments Hub.

    Args:
        name: Name of the environment (owner/name format, e.g., "will/wordle").
        cache: Whether to cache the environment for reuse.
        **kwargs: Additional arguments passed to PrimeIntellectEnvironment.

    Returns:
        PrimeIntellectEnvironment: The loaded environment.

    Example:
        >>> env = load_environment("will/wordle")
        >>> result = env.evaluate("CRANE")
    """
    global _environment_cache

    if cache and name in _environment_cache:
        return _environment_cache[name]

    env = PrimeIntellectEnvironment(name, **kwargs)

    if cache:
        _environment_cache[name] = env

    return env


def clear_environment_cache():
    """Clear the environment cache."""
    global _environment_cache
    _environment_cache.clear()


def list_available_environments() -> list[str]:
    """List available environments from the Prime Intellect Hub.

    Returns:
        list: List of environment names available on the hub.
    """
    try:
        # Try to use the prime CLI to list environments
        import subprocess

        result = subprocess.run(
            ["prime", "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            import json

            data = json.loads(result.stdout)
            return [env.get("name", "") for env in data if env.get("name")]
    except Exception as e:
        logger.debug(f"Could not list environments from CLI: {e}")

    # Return some well-known environments as fallback
    return [
        "will/wordle",
        "primeintellect/math-python",
        "primeintellect/code-contests",
    ]
