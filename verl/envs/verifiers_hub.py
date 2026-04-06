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

"""Auto-registration of *verifiers* library environments into verl's registry.

When ``verifiers`` is installed (``pip install verifiers``), calling
``register_all_verifiers_envs()`` makes every known verifiers environment
available under the ``verifiers/`` namespace in verl's environment registry.

Example::

    from verl.envs import get_env_cls, list_envs

    # After registration:
    env_cls = get_env_cls("verifiers/singleturn")
    print(list_envs())
    # ['verifiers/singleturn', 'verifiers/multiturn', 'verifiers/tool', ...]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_REGISTERED = False


def _make_adapter_cls(vf_env_cls: type, env_name: str) -> type:
    """Dynamically create a BaseEnvironment subclass that wraps a verifiers env class."""
    from verl.envs.base import BaseEnvironment
    from verl.envs.verifiers_adapter import VerifiersEnvAdapter

    class _Adapter(VerifiersEnvAdapter):
        __doc__ = f"verl adapter for verifiers ``{vf_env_cls.__name__}``."

        def __init__(self, config: dict[str, Any] | None = None):
            config = config or {}
            env_kwargs = config.get("env_kwargs", {})
            try:
                vf_instance = vf_env_cls(**env_kwargs)
            except Exception:
                logger.debug(f"Could not instantiate {vf_env_cls.__name__} with kwargs, trying no-arg", exc_info=True)
                vf_instance = vf_env_cls()
            super().__init__(vf_instance, config)

    _Adapter.__name__ = f"Verifiers_{vf_env_cls.__name__}_Adapter"
    _Adapter.__qualname__ = _Adapter.__name__
    return _Adapter


# Known verifiers environment classes and their registry names.
# Each entry is (import_path, registry_name).
_VERIFIERS_ENV_MAP: list[tuple[str, str]] = [
    ("verifiers.envs.singleturn_env.SingleTurnEnv", "verifiers/singleturn"),
    ("verifiers.envs.multiturn_env.MultiTurnEnv", "verifiers/multiturn"),
    ("verifiers.envs.tool_env.ToolEnv", "verifiers/tool"),
    ("verifiers.envs.python_env.PythonEnv", "verifiers/python"),
    ("verifiers.envs.sandbox_env.SandboxEnv", "verifiers/sandbox"),
    # Integration environments
    ("verifiers.envs.integrations.reasoninggym_env.ReasoningGymEnv", "verifiers/reasoning-gym"),
    ("verifiers.envs.integrations.textarena_env.TextArenaEnv", "verifiers/textarena"),
    ("verifiers.envs.integrations.openenv_env.OpenEnvEnv", "verifiers/openenv"),
]


def register_all_verifiers_envs() -> None:
    """Auto-register all known verifiers environments into verl's ENV_REGISTRY.

    Safe to call multiple times — registration is idempotent.
    Environments whose dependencies are not installed are silently skipped.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from verl.envs.registry import ENV_REGISTRY

    registered_count = 0
    for import_path, registry_name in _VERIFIERS_ENV_MAP:
        try:
            module_path, cls_name = import_path.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            vf_cls = getattr(module, cls_name)

            adapter_cls = _make_adapter_cls(vf_cls, registry_name)
            if registry_name not in ENV_REGISTRY:
                ENV_REGISTRY[registry_name] = adapter_cls
                registered_count += 1
        except ImportError:
            logger.debug(f"Skipping verifiers env '{registry_name}': missing dependencies")
        except Exception:
            logger.debug(f"Failed to register verifiers env '{registry_name}'", exc_info=True)

    if registered_count > 0:
        logger.info(f"Registered {registered_count} verifiers environments in verl")


def create_from_verifiers(env_class_or_name: Any, **kwargs: Any):
    """Create a verl ``BaseEnvironment`` from a verifiers env class or instance.

    Args:
        env_class_or_name: Either a verifiers ``Environment`` class, an
            instantiated environment, or a string name from the registry.
        **kwargs: Passed to the verifiers environment constructor (if class).

    Returns:
        A ``VerifiersEnvAdapter`` wrapping the verifiers environment.
    """
    from verl.envs.verifiers_adapter import VerifiersEnvAdapter

    if isinstance(env_class_or_name, str):
        from verl.envs.registry import get_env_cls

        cls = get_env_cls(env_class_or_name)
        return cls(config={"env_kwargs": kwargs})

    # If it's a class, instantiate it
    if isinstance(env_class_or_name, type):
        instance = env_class_or_name(**kwargs)
        return VerifiersEnvAdapter(instance)

    # Already an instance
    return VerifiersEnvAdapter(env_class_or_name)
