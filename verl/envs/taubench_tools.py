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

"""Tau-bench tool integration for verl's multi-turn agent loop.

Provides ``TauBenchTool``, a ``BaseTool`` subclass that bridges verl's tool
execution interface with tau-bench's domain-specific API tools.  When used
inside verl's agent loop, tool calls from the LLM are routed through this
class to the underlying tau-bench environment.

This follows the same pattern as ``verl.tools.gsm8k_tool.Gsm8kTool``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Global registry of tau-bench environment instances shared between
# TauBenchTool and EnvironmentInteraction via instance_id.
_TAUBENCH_ENV_INSTANCES: dict[str, Any] = {}


def register_taubench_instance(instance_id: str, env_instance: Any) -> None:
    """Register a tau-bench env instance so tools can access it."""
    _TAUBENCH_ENV_INSTANCES[instance_id] = env_instance


def unregister_taubench_instance(instance_id: str) -> None:
    """Remove a tau-bench env instance from the shared registry."""
    _TAUBENCH_ENV_INSTANCES.pop(instance_id, None)


class TauBenchTool(BaseTool):
    """A verl tool that delegates execution to a tau-bench environment.

    This tool acts as a generic proxy: when the LLM calls any tau-bench API
    tool (e.g. ``get_order_details``, ``process_return``), the call is
    forwarded to the active tau-bench environment instance.

    Config keys:
        domain (str): ``"retail"`` or ``"airline"``. Defaults to ``"retail"``.
    """

    def __init__(self, config: dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.domain = config.get("domain", "retail")
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "reward": 0.0,
            "tool_calls": [],
        }

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        """Execute a tool call by forwarding to the tau-bench environment.

        The ``parameters`` dict should contain:
        - ``action`` or ``name``: The tool/action name to execute
        - ``arguments`` or ``kwargs``: Arguments for the tool

        If a tau-bench environment instance is registered for this
        ``instance_id``, the tool call is executed directly in that env.
        Otherwise, returns a generic acknowledgement.
        """
        action_name = parameters.get("name", parameters.get("action", self.name))
        arguments = parameters.get("arguments", parameters.get("kwargs", {}))

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"input": arguments}

        # Track tool calls
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["tool_calls"].append(
                {"name": action_name, "arguments": arguments}
            )

        # Try to execute in the registered tau-bench env
        env_data = _TAUBENCH_ENV_INSTANCES.get(instance_id)
        if env_data is not None:
            env = env_data if not isinstance(env_data, dict) else env_data.get("env")
            if env is not None and hasattr(env, "tools_map"):
                try:
                    if action_name in env.tools_map:
                        tool_obj = env.tools_map[action_name]
                        result = tool_obj.invoke(**arguments)
                        result_str = str(result) if result is not None else "Action completed successfully."
                        return ToolResponse(text=result_str), 0.0, {}
                except Exception as e:
                    error_msg = f"Tool execution error: {e}"
                    logger.warning(error_msg, exc_info=True)
                    return ToolResponse(text=error_msg), -0.1, {"error": str(e)}

        # Fallback: return a generic response
        return (
            ToolResponse(text=f"Executed {action_name} with {json.dumps(arguments)}"),
            0.0,
            {},
        )

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id]["reward"]
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instance_dict.pop(instance_id, None)
        unregister_taubench_instance(instance_id)
