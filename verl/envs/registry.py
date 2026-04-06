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

from __future__ import annotations

from typing import Callable

from verl.envs.base import BaseEnvironment

__all__ = ["register_env", "get_env_cls", "list_envs"]

ENV_REGISTRY: dict[str, type[BaseEnvironment]] = {}


def register_env(name: str) -> Callable[[type[BaseEnvironment]], type[BaseEnvironment]]:
    """Decorator to register an environment class with a given name.

    Usage::

        @register_env("my_env")
        class MyEnvironment(BaseEnvironment):
            ...

    A single class can be registered under multiple names by stacking decorators.
    """

    def decorator(cls: type[BaseEnvironment]) -> type[BaseEnvironment]:
        if name in ENV_REGISTRY and ENV_REGISTRY[name] != cls:
            raise ValueError(f"Environment '{name}' already registered: {ENV_REGISTRY[name]} vs {cls}")
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


def get_env_cls(name: str) -> type[BaseEnvironment]:
    """Get the environment class registered under *name*.

    Raises:
        ValueError: If no environment is registered with that name.
    """
    if name not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys())) or "(none)"
        raise ValueError(f"Unknown environment: '{name}'. Available: {available}")
    return ENV_REGISTRY[name]


def list_envs() -> list[str]:
    """Return a sorted list of all registered environment names."""
    return sorted(ENV_REGISTRY.keys())
