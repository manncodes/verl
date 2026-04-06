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

"""Tau-bench environment integration for verl.

`tau-bench <https://github.com/sierra-research/tau-bench>`_ is a benchmark for
evaluating language model agents on real-world customer-service tasks across
multiple domains (retail, airline).  It features multi-turn conversations with a
simulated user, tool/API usage, and policy-compliance scoring.

Requires: ``pip install tau-bench`` (clone & ``pip install -e .`` from GitHub).

Three registry names are provided:

* ``taubench`` — defaults to the retail domain
* ``taubench-retail`` — retail customer service
* ``taubench-airline`` — airline customer service
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from verl.envs.base import BaseEnvironment
from verl.envs.registry import register_env

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
DOMAINS = ("retail", "airline")

DOMAIN_SYSTEM_PROMPTS = {
    "retail": (
        "You are a customer service agent for an online retail company. "
        "Help the customer with their request while strictly following company policies. "
        "Use the available tools to look up orders, process returns, and handle requests. "
        "Always verify the customer's identity before making changes to their account."
    ),
    "airline": (
        "You are a customer service agent for an airline company. "
        "Help the customer with their request while strictly following airline policies. "
        "Use the available tools to look up reservations, manage bookings, and handle requests. "
        "Always verify the customer's identity before making changes to their booking."
    ),
}


def _lazy_import_tau_bench():
    """Lazily import tau_bench and return the module."""
    try:
        import tau_bench
        return tau_bench
    except ImportError:
        raise ImportError(
            "tau-bench is required for the TauBenchEnvironment. "
            "Install it with: git clone https://github.com/sierra-research/tau-bench && "
            "cd tau-bench && pip install -e ."
        )


def _get_tau_env(domain: str, task_split: str = "test", task_index: int | None = None):
    """Create a tau-bench environment instance for the given domain."""
    tb = _lazy_import_tau_bench()
    from tau_bench.envs import get_env

    return get_env(
        env_name=domain,
        user_strategy="llm",
        user_model="gpt-4o-mini",
        task_split=task_split,
        task_index=task_index,
    )


def _get_domain_tasks(domain: str, split: str = "test") -> list[dict[str, Any]]:
    """Load task definitions for a tau-bench domain."""
    tb = _lazy_import_tau_bench()

    if domain == "retail":
        from tau_bench.envs.retail import tasks as task_module
    elif domain == "airline":
        from tau_bench.envs.airline import tasks as task_module
    else:
        raise ValueError(f"Unknown tau-bench domain: {domain}")

    # tau-bench stores tasks in module-level variables by split
    if split == "train":
        tasks = getattr(task_module, "TASKS_TRAIN", None)
        if tasks is None:
            from tau_bench.envs.retail import tasks_train

            tasks = tasks_train.TASKS
    elif split in ("dev", "validation"):
        tasks = getattr(task_module, "TASKS_DEV", None)
        if tasks is None:
            try:
                from tau_bench.envs.retail import tasks_dev

                tasks = tasks_dev.TASKS
            except (ImportError, AttributeError):
                tasks = getattr(task_module, "TASKS_TEST", task_module.TASKS)
    else:
        tasks = getattr(task_module, "TASKS_TEST", None)
        if tasks is None:
            tasks = getattr(task_module, "TASKS", [])

    # Normalize to list of dicts
    result = []
    for i, task in enumerate(tasks):
        if hasattr(task, "__dict__"):
            task_dict = {k: v for k, v in task.__dict__.items() if not k.startswith("_")}
        elif isinstance(task, dict):
            task_dict = task
        else:
            task_dict = {"task": str(task)}
        task_dict["task_index"] = i
        task_dict["domain"] = domain
        result.append(task_dict)

    return result


@register_env("taubench")
@register_env("taubench-retail")
@register_env("taubench-airline")
class TauBenchEnvironment(BaseEnvironment):
    """Multi-turn customer-service environment powered by tau-bench.

    Config keys:
        domain (str): ``"retail"`` or ``"airline"``. Defaults to ``"retail"``.
        task_split (str): Which task split to use. Defaults to ``"test"``.
        max_turns (int): Maximum conversation turns. Defaults to 20.
        user_strategy (str): User simulator strategy. Defaults to ``"llm"``.
        user_model (str): Model for user simulation. Defaults to ``"gpt-4o-mini"``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.domain = self.config.get("domain", "retail")

        # Infer domain from registry name if not explicitly set
        if self.domain == "retail" and self.config.get("_registry_name") == "taubench-airline":
            self.domain = "airline"

        self.task_split = self.config.get("task_split", "test")
        self.max_turns = self.config.get("max_turns", 20)
        self.user_strategy = self.config.get("user_strategy", "llm")
        self.user_model = self.config.get("user_model", "gpt-4o-mini")

        # Instance tracking for multi-turn sessions
        self._instances: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def get_dataset(self, split: str = "train"):
        """Return tau-bench tasks as a HuggingFace Dataset.

        Each row contains:
        - ``prompt``: The user's initial instruction
        - ``ground_truth``: Task metadata for scoring
        - ``task_index``: Index into tau-bench's task list
        - ``domain``: The environment domain
        """
        tasks = _get_domain_tasks(self.domain, split)

        from datasets import Dataset

        rows = []
        for task in tasks:
            user_instruction = task.get("user_instruction", task.get("instruction", ""))
            rows.append(
                {
                    "prompt": user_instruction,
                    "ground_truth": json.dumps(task),
                    "task_index": task.get("task_index", 0),
                    "domain": self.domain,
                    "data_source": f"taubench-{self.domain}",
                }
            )

        return Dataset.from_list(rows)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any] | None = None,
    ) -> float | dict[str, Any]:
        """Score a completed trajectory.

        For multi-turn evaluation, the reward comes from the environment's
        built-in ``calculate_reward()`` method called during ``step()``.
        For single-turn fallback, we check if key outputs are present in the
        response.
        """
        extra_info = extra_info or {}

        # If we have a cached score from multi-turn interaction, use it
        instance_id = extra_info.get("instance_id")
        if instance_id and instance_id in self._instances:
            return self._instances[instance_id].get("final_reward", 0.0)

        # Fallback: simple output matching
        if isinstance(ground_truth, str):
            try:
                task_data = json.loads(ground_truth)
            except (json.JSONDecodeError, TypeError):
                task_data = {"expected_output": ground_truth}
        else:
            task_data = ground_truth if isinstance(ground_truth, dict) else {}

        expected_outputs = task_data.get("outputs", task_data.get("expected_output", []))
        if isinstance(expected_outputs, str):
            expected_outputs = [expected_outputs]

        if not expected_outputs:
            return 0.0

        # Check if all expected outputs are present in the solution
        matches = sum(1 for out in expected_outputs if str(out).lower() in solution_str.lower())
        return matches / len(expected_outputs) if expected_outputs else 0.0

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str | None:
        return DOMAIN_SYSTEM_PROMPTS.get(self.domain)

    def get_tool_definitions(self) -> list[dict[str, Any]] | None:
        """Return tool definitions for the tau-bench domain.

        These are the customer service API tools (order lookup, returns, etc.).
        """
        try:
            env = _get_tau_env(self.domain, self.task_split, task_index=0)
            tool_defs = []
            for tool_name, tool in env.tools_map.items():
                schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": getattr(tool, "description", f"Execute {tool_name}"),
                        "parameters": getattr(tool, "parameters", {"type": "object", "properties": {}}),
                    },
                }
                tool_defs.append(schema)
            return tool_defs
        except Exception:
            logger.debug("Could not load tau-bench tool definitions", exc_info=True)
            return None

    def is_multi_turn(self) -> bool:
        return True

    def get_max_turns(self) -> int:
        return self.max_turns

    # ------------------------------------------------------------------
    # Multi-turn interface
    # ------------------------------------------------------------------

    async def reset(self, problem: dict[str, Any], instance_id: str) -> dict[str, Any]:
        """Initialize a tau-bench environment for a specific task.

        Args:
            problem: Must contain ``task_index`` or ``ground_truth`` (JSON string
                with task data).
            instance_id: Unique instance identifier.

        Returns:
            Dict with initial observation and metadata.
        """
        task_index = problem.get("task_index")

        # Parse ground_truth if it's a JSON string
        if task_index is None and "ground_truth" in problem:
            gt = problem["ground_truth"]
            if isinstance(gt, str):
                try:
                    gt_data = json.loads(gt)
                    task_index = gt_data.get("task_index", 0)
                except (json.JSONDecodeError, TypeError):
                    task_index = 0
            elif isinstance(gt, dict):
                task_index = gt.get("task_index", 0)

        task_index = task_index or 0

        try:
            env = _get_tau_env(
                domain=problem.get("domain", self.domain),
                task_split=self.task_split,
                task_index=task_index,
            )
            reset_response = env.reset(task_index=task_index)

            self._instances[instance_id] = {
                "env": env,
                "task_index": task_index,
                "turns": 0,
                "done": False,
                "final_reward": 0.0,
                "conversation": [],
                "initial_observation": str(reset_response.observation),
            }

            return {
                "observation": str(reset_response.observation),
                "info": getattr(reset_response, "info", {}),
            }
        except Exception as e:
            logger.warning(f"Failed to reset tau-bench env: {e}", exc_info=True)
            self._instances[instance_id] = {
                "env": None,
                "task_index": task_index,
                "turns": 0,
                "done": True,
                "final_reward": 0.0,
                "conversation": [],
                "initial_observation": f"Error initializing environment: {e}",
            }
            return {"observation": f"Error: {e}", "info": {"error": str(e)}}

    async def step(
        self,
        instance_id: str,
        action: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute one step in the tau-bench environment.

        The action is parsed to determine if it's a tool call or a message to
        the user. Tool calls are executed via tau-bench's tool system; messages
        are passed to the user simulator.
        """
        if instance_id not in self._instances:
            return "Error: instance not initialized.", 0.0, True, {}

        inst = self._instances[instance_id]
        if inst["done"]:
            return "Episode already finished.", inst["final_reward"], True, {}

        env = inst["env"]
        if env is None:
            return "Environment not available.", 0.0, True, {}

        inst["turns"] += 1
        inst["conversation"].append({"role": "assistant", "content": action})

        try:
            # Parse the action to create a tau-bench Action object
            tb_action = self._parse_action(action, env)

            # Step the environment
            env_response = env.step(tb_action)

            observation = str(env_response.observation)
            done = env_response.done
            reward = 0.0
            info = getattr(env_response, "info", {})

            if done:
                # Calculate final reward using tau-bench's scoring
                try:
                    reward_result = env.calculate_reward()
                    reward = float(reward_result.reward)
                    info["reward_info"] = {
                        "reward": reward,
                        "actions_correct": getattr(reward_result, "actions_correct", None),
                        "outputs_correct": getattr(reward_result, "outputs_correct", None),
                    }
                except Exception as e:
                    logger.warning(f"Reward calculation failed: {e}")
                    reward = 0.0

                inst["final_reward"] = reward
                inst["done"] = True

            # Check max turns
            if inst["turns"] >= self.max_turns and not done:
                done = True
                try:
                    reward_result = env.calculate_reward()
                    reward = float(reward_result.reward)
                except Exception:
                    reward = 0.0
                inst["final_reward"] = reward
                inst["done"] = True

            inst["conversation"].append({"role": "user", "content": observation})

            return observation, reward, done, info

        except Exception as e:
            logger.warning(f"Error in tau-bench step: {e}", exc_info=True)
            return f"Error: {e}", 0.0, False, {"error": str(e)}

    def _parse_action(self, action: str, env: Any) -> Any:
        """Parse the agent's action string into a tau-bench Action object.

        Supports:
        - Tool calls in JSON format: ``{"name": "tool_name", "arguments": {...}}``
        - Plain text responses (treated as RESPOND action)
        """
        from tau_bench.envs.tool import Action

        # Try to parse as a tool call
        try:
            if action.strip().startswith("{"):
                action_data = json.loads(action)
                if "name" in action_data:
                    return Action(
                        name=action_data["name"],
                        kwargs=action_data.get("arguments", action_data.get("kwargs", {})),
                    )
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract tool call from function-call format
        try:
            if "(" in action and action.strip().endswith(")"):
                func_name = action[: action.index("(")].strip()
                args_str = action[action.index("(") + 1 : -1]
                if func_name in env.tools_map:
                    try:
                        kwargs = json.loads("{" + args_str + "}")
                    except json.JSONDecodeError:
                        kwargs = {"input": args_str}
                    return Action(name=func_name, kwargs=kwargs)
        except Exception:
            pass

        # Default: treat as a RESPOND action (message to user)
        respond_name = "respond"
        # tau-bench uses a special action name for responding to user
        for name in ("respond", "RESPOND", "send_message", "reply"):
            if hasattr(env, "tools_map") and name in env.tools_map:
                respond_name = name
                break

        return Action(name=respond_name, kwargs={"message": action})

    async def close(self, instance_id: str) -> None:
        """Clean up the tau-bench environment instance."""
        self._instances.pop(instance_id, None)
