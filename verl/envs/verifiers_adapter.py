# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Adapter that wraps a *verifiers* library ``Environment`` as a verl ``BaseEnvironment``.

The `verifiers <https://github.com/PrimeIntellect-ai/verifiers>`_ library provides
a rich set of LLM evaluation environments (math, code, tool-use, multi-turn, etc.).
This adapter translates the verifiers API into verl's ``BaseEnvironment`` interface
so that any verifiers environment can be used seamlessly in verl's reward and
interaction pipelines.

Requires: ``pip install verifiers``
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from verl.envs.base import BaseEnvironment

logger = logging.getLogger(__name__)


class VerifiersEnvAdapter(BaseEnvironment):
    """Wraps a *verifiers* ``Environment`` instance for use in verl.

    Args:
        verifiers_env: An instantiated verifiers ``Environment`` (e.g.
            ``SingleTurnEnv``, ``MultiTurnEnv``, ``ToolEnv``).
        config: Optional verl-side configuration dict.
    """

    def __init__(self, verifiers_env: Any, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._vf_env = verifiers_env
        self._states: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def get_dataset(self, split: str = "train"):
        """Return the verifiers environment's dataset.

        Verifiers environments expose HuggingFace ``datasets.Dataset`` objects
        via ``get_dataset()`` / ``get_eval_dataset()``.
        """
        if split in ("test", "eval", "validation"):
            return self._vf_env.get_eval_dataset()
        return self._vf_env.get_dataset()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any] | None = None,
    ) -> float | dict[str, Any]:
        """Score a completion using the verifiers environment's rubric system.

        For single-turn environments the rubric is evaluated directly. For
        multi-turn environments this provides a simplified single-turn scoring
        path (e.g. for final-answer evaluation).
        """
        extra_info = extra_info or {}

        # Try using the environment's rubric(s) to score
        rubrics = getattr(self._vf_env, "rubrics", None) or getattr(self._vf_env, "rubric", None)
        if rubrics is not None:
            try:
                return self._score_with_rubrics(rubrics, solution_str, ground_truth, extra_info)
            except Exception:
                logger.debug("Rubric scoring failed, falling back to simple matching", exc_info=True)

        # Fallback: exact string match
        if str(solution_str).strip() == str(ground_truth).strip():
            return 1.0
        return 0.0

    def _score_with_rubrics(
        self,
        rubrics: Any,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any],
    ) -> float | dict[str, Any]:
        """Attempt to score using verifiers Rubric objects."""
        # Verifiers rubrics are callables that accept (completion, answer, **kwargs)
        if callable(rubrics):
            result = rubrics(completion=solution_str, answer=ground_truth, **extra_info)
            if isinstance(result, dict):
                return result
            return float(result)

        # RubricGroup: iterate and aggregate
        if hasattr(rubrics, "__iter__"):
            total = 0.0
            count = 0
            details: dict[str, Any] = {}
            for rubric in rubrics:
                if callable(rubric):
                    r = rubric(completion=solution_str, answer=ground_truth, **extra_info)
                    name = getattr(rubric, "name", f"rubric_{count}")
                    val = r if isinstance(r, (int, float)) else r.get("score", 0.0) if isinstance(r, dict) else float(r)
                    details[name] = val
                    total += val
                    count += 1
            if count > 0:
                details["score"] = total / count
                return details

        return 0.0

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str | None:
        sp = getattr(self._vf_env, "system_prompt", None)
        if sp:
            return str(sp)
        return None

    def get_tool_definitions(self) -> list[dict[str, Any]] | None:
        tools = getattr(self._vf_env, "tools", None)
        if tools:
            defs = getattr(self._vf_env, "tool_definitions", None)
            if defs:
                return list(defs)
        return None

    def is_multi_turn(self) -> bool:
        max_turns = getattr(self._vf_env, "max_turns", 1)
        return max_turns is not None and max_turns > 1

    def get_max_turns(self) -> int:
        return getattr(self._vf_env, "max_turns", 1) or 1

    # ------------------------------------------------------------------
    # Multi-turn interface
    # ------------------------------------------------------------------

    async def reset(self, problem: dict[str, Any], instance_id: str) -> dict[str, Any]:
        """Initialize a verifiers environment state for the given problem."""
        state = self._vf_env.init_state(problem)
        self._states[instance_id] = state
        return {"instance_id": instance_id, "state_initialized": True}

    async def step(
        self,
        instance_id: str,
        action: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute one step in the verifiers multi-turn environment.

        Calls the verifiers environment's ``env_response`` to get the
        environment's reply to the agent's action.
        """
        if instance_id not in self._states:
            return "Error: instance not initialized. Call reset() first.", 0.0, True, {}

        state = self._states[instance_id]

        # Build the assistant message and add to state
        if messages is not None:
            state["messages"] = messages
        else:
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append({"role": "assistant", "content": action})

        try:
            # Call the verifiers env's env_response method
            env_response_fn = getattr(self._vf_env, "env_response", None)
            if env_response_fn is None:
                return "Environment does not support multi-turn stepping.", 0.0, True, {}

            import asyncio

            if asyncio.iscoroutinefunction(env_response_fn):
                response = await env_response_fn(state)
            else:
                response = env_response_fn(state)

            # Parse the response
            if isinstance(response, tuple):
                content, done = response[0], response[1] if len(response) > 1 else False
                reward = response[2] if len(response) > 2 else 0.0
            elif isinstance(response, str):
                content = response
                done = False
                reward = 0.0
            elif isinstance(response, dict):
                content = response.get("content", str(response))
                done = response.get("done", False)
                reward = response.get("reward", 0.0)
            else:
                content = str(response) if response is not None else ""
                done = response is None
                reward = 0.0

            # Add environment response to messages
            if content:
                state["messages"].append({"role": "user", "content": content})

            # Check verifiers stop conditions
            is_completed_fn = getattr(self._vf_env, "is_completed", None)
            if is_completed_fn and not done:
                done = is_completed_fn(state)

            info = {"messages": state.get("messages", [])}
            return content, reward, done, info

        except Exception as e:
            logger.warning(f"Error in verifiers env step: {e}", exc_info=True)
            return f"Environment error: {e}", 0.0, True, {"error": str(e)}

    async def close(self, instance_id: str) -> None:
        self._states.pop(instance_id, None)


def create_adapter(verifiers_env: Any, config: dict[str, Any] | None = None) -> VerifiersEnvAdapter:
    """Convenience factory to wrap a verifiers environment instance."""
    return VerifiersEnvAdapter(verifiers_env, config)
