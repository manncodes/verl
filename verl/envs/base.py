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

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):
    """Generalized environment abstraction for verl RL training.

    Environments provide:
    - Dataset loading (prompts/problems to train on)
    - Reward/scoring for completions (single-turn or trajectory-level)
    - Optional multi-turn stepping (for agent/interactive environments)
    - System prompt and tool definitions for the agent

    Single-turn environments only need to implement ``get_dataset`` and ``score``.
    Multi-turn environments should additionally override ``is_multi_turn``,
    ``get_max_turns``, ``reset``, ``step``, and optionally ``close``.

    Environments integrate into verl through two adapter layers:
    - **Reward adapter**: routes ``default_compute_score`` to ``env.score()``
    - **Interaction adapter**: bridges ``BaseEnvironment`` to verl's
      ``BaseInteraction`` for multi-turn agent loops
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    @abstractmethod
    def get_dataset(self, split: str = "train"):
        """Return a HuggingFace ``datasets.Dataset`` of problems/prompts.

        Each row should contain at minimum a ``prompt`` field (str) and a
        ``ground_truth`` field used for scoring.
        """
        ...

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @abstractmethod
    def score(
        self,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any] | None = None,
    ) -> float | dict[str, Any]:
        """Score a single completion against the ground truth.

        Returns either a float reward or a dict with at least a ``"score"``
        key plus optional extra metrics.
        """
        ...

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str | None:
        """Return the system prompt the agent should use, or ``None``."""
        return None

    def get_tool_definitions(self) -> list[dict[str, Any]] | None:
        """Return OpenAI-format tool schemas, or ``None`` if no tools."""
        return None

    def is_multi_turn(self) -> bool:
        """Whether this environment requires multi-turn interaction."""
        return False

    def get_max_turns(self) -> int:
        """Maximum number of assistant turns for multi-turn environments."""
        return 1

    # ------------------------------------------------------------------
    # Multi-turn interface (override for interactive environments)
    # ------------------------------------------------------------------

    async def reset(self, problem: dict[str, Any], instance_id: str) -> dict[str, Any]:
        """Initialize a new episode for the given problem.

        Args:
            problem: A row from ``get_dataset()`` containing the task spec.
            instance_id: Unique identifier for this rollout instance.

        Returns:
            Initial observation / state dict.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support multi-turn interaction")

    async def step(
        self,
        instance_id: str,
        action: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """Take one step in the environment.

        Args:
            instance_id: The rollout instance identifier.
            action: The agent's action (typically the assistant message content).
            messages: Full conversation history (optional, for context).

        Returns:
            A tuple of ``(response, reward, done, info)``:
            - response: The environment's textual response.
            - reward: Step-level reward (or cumulative reward if done).
            - done: Whether the episode has ended.
            - info: Extra information dict.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support multi-turn interaction")

    async def close(self, instance_id: str) -> None:
        """Clean up resources for a finished rollout instance."""
        pass
