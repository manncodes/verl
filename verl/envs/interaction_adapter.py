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

"""Bridge between verl's ``BaseEnvironment`` and verl's ``BaseInteraction``.

This module provides ``EnvironmentInteraction``, a ``BaseInteraction`` subclass
that wraps any ``BaseEnvironment`` for use in verl's multi-turn agent loop
(``verl/experimental/agent_loop/``).  It follows the same pattern as
``verl.interactions.gsm8k_interaction.Gsm8kInteraction``.

Usage in an interaction config YAML::

    interaction:
      - name: "my_env"
        class_name: "verl.envs.interaction_adapter.EnvironmentInteraction"
        config:
          env_name: "taubench-retail"   # registered BaseEnvironment name
          env_kwargs:
            domain: retail
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_last_assistant_content(messages: list[dict[str, Any]]) -> str:
    """Extract the content of the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


class EnvironmentInteraction(BaseInteraction):
    """Adapts a ``BaseEnvironment`` to verl's ``BaseInteraction`` interface.

    This allows any registered environment (including verifiers-based ones and
    tau-bench) to be used in verl's multi-turn agent loop via the standard
    interaction config mechanism.

    Config keys:
        env_name (str): Name of the registered environment.
        env_kwargs (dict): Extra kwargs passed to the environment constructor.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        env_name = config["env_name"]
        env_kwargs = config.get("env_kwargs", {})

        from verl.envs.registry import get_env_cls

        env_cls = get_env_cls(env_name)
        self._env = env_cls(config=env_kwargs)
        self._instances: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        problem = {"ground_truth": ground_truth}
        problem.update(kwargs)

        # For extra_info or create_kwargs that carry task-specific data
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            problem.update(create_kwargs)

        state = await self._env.reset(problem, instance_id)

        self._instances[instance_id] = {
            "state": state,
            "ground_truth": ground_truth,
            "score": 0.0,
            "turns": 0,
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        if instance_id not in self._instances:
            return True, "Error: interaction not started.", 0.0, {}

        action = _extract_last_assistant_content(messages)
        response, reward, done, info = await self._env.step(instance_id, action, messages)

        inst = self._instances[instance_id]
        inst["score"] = reward
        inst["turns"] += 1

        # Enforce max turns
        max_turns = self._env.get_max_turns()
        if inst["turns"] >= max_turns:
            done = True

        return done, response, reward, info

    async def calculate_score(self, instance_id: str = "", **kwargs: Any) -> float:
        if instance_id not in self._instances:
            return 0.0
        return self._instances[instance_id]["score"]

    async def finalize_interaction(self, instance_id: str = "", **kwargs: Any) -> None:
        await self._env.close(instance_id)
        self._instances.pop(instance_id, None)
