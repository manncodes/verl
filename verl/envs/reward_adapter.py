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

"""Bridge between verl's ``BaseEnvironment`` and verl's reward scoring system.

Provides ``env_compute_score`` which can be used as a drop-in replacement for
or extension of ``verl.utils.reward_score.default_compute_score``.  When a
``data_source`` matches a registered environment name, the environment's
``score()`` method is used for reward computation.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Cache environment instances to avoid re-constructing them for every sample.
_env_cache: dict[str, Any] = {}


def _get_or_create_env(data_source: str):
    """Get a cached environment instance, creating it if necessary."""
    if data_source not in _env_cache:
        from verl.envs.registry import get_env_cls

        env_cls = get_env_cls(data_source)
        _env_cache[data_source] = env_cls()
    return _env_cache[data_source]


def env_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """Compute reward score using a registered ``BaseEnvironment``.

    This function is intended to be called from
    ``verl.utils.reward_score.default_compute_score`` as a fallback when
    the ``data_source`` doesn't match any built-in scoring modules.

    Args:
        data_source: The dataset/environment identifier (must be a registered
            environment name, e.g. ``"taubench-retail"``).
        solution_str: The model's generated response.
        ground_truth: The ground truth answer/task specification.
        extra_info: Additional context passed from the reward manager.
        **kwargs: Extra keyword arguments (ignored).

    Returns:
        A float score or a dict with ``"score"`` and optional extra keys.

    Raises:
        ValueError: If ``data_source`` is not a registered environment.
    """
    env = _get_or_create_env(data_source)
    return env.score(solution_str, ground_truth, extra_info)


def is_env_data_source(data_source: str) -> bool:
    """Check whether a data_source corresponds to a registered environment."""
    try:
        from verl.envs.registry import ENV_REGISTRY

        return data_source in ENV_REGISTRY
    except ImportError:
        return False
