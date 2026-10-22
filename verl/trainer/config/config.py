# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["CheckpointConfig", "ProfileConfig", "HybridRolloutSwitchConfig", "ElasticConfig", "BaseModelConfig"]


@dataclass
class CheckpointConfig(BaseConfig):
    """Configuration for model checkpointing.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Keep this base class limited to fields every backend understands.

    Args:
        save_contents (list[str]): What to include in saved checkpoints.
            Options: 'model', 'optimizer', 'extra', 'hf_model'.
        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.
        async_save (bool): Whether to save checkpoints asynchronously. Only implemented for Megatron as of now.
        strict (bool): Whether to perform strict validation during weight export
        save_lora_only (bool): When True and the model has LoRA adapters, only
            save LoRA adapter weights instead of the full model state dict.
            Dramatically reduces checkpoint size (e.g. ~150 MiB vs ~54 GiB for a
            27B model). Loaded LoRA-only checkpoints are auto-detected and merged
            into the current model state. Has no effect when the model has no LoRA
            adapters.
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False
    strict: bool = True
    save_lora_only: bool = False


@dataclass
class ProfileConfig(BaseConfig):
    """Configuration for profiling.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        profile_ranks (Optional[list[int]]): List of ranks to profile. None means all ranks.
        step_start (int): Starting step for profiling.
        step_end (int): Ending step for profiling.
        save_path (Optional[str]): Path to save profiling results.
    """

    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class HybridRolloutSwitchConfig(BaseConfig):
    """Configuration for lending idle trainer GPUs to rollout between training steps."""

    enable_switch: bool = False
    switch_threshold_ratio: float = 0.4
    adaptive_switch_threshold: bool = True
    switch_threshold_step_up: float = 0.05
    switch_threshold_step_down: float = 0.03
    switch_threshold_release_steps: int = 2
    switch_cost_window_size: int = 3

    def __post_init__(self):
        if not 0.0 < self.switch_threshold_ratio <= 1.0:
            raise ValueError(f"switch_threshold_ratio must be in (0, 1], got {self.switch_threshold_ratio}")
        if self.switch_cost_window_size <= 0:
            raise ValueError("switch_cost_window_size must be positive")


@dataclass
class ElasticConfig(BaseConfig):
    """Configuration for elastic, fault-tolerant training (v1 separate_async trainer).

    Rollout replicas follow the GPU nodes that join and leave the Ray cluster, each shaped for its node,
    and the trainer restarts in place from its latest checkpoint when one of its nodes is lost.

    Args:
        enable (bool): Turn elastic training on.
        health_check_interval_s (float): Period of the background loop that probes, fills and drains.
        request_retries (int): Times a generation request is retried, one second apart, after its server died.
        min_rollout_gpus (int): Lower bound of the rollout GPU target.
        max_rollout_gpus (Optional[int]): Upper bound of the rollout GPU target; None uses every free GPU.
        autoscale (bool): Publish the target to the Ray autoscaler with ``request_resources``.
        policy (str): Scaling policy that moves the target: "static" or "backlog".
        up_wait_ratio (float): backlog policy: scale up when the trainer waits for samples longer than
            this fraction of a step.
        patience (int): backlog policy: consecutive steps a condition must hold before the target moves.
        scale_step_gpus (int): backlog policy: GPUs added or removed per move.
        max_weight_fraction (float): Largest fraction of one GPU's memory a replica's weight shard may use.
        tp_by_accelerator (dict[str, int]): Tensor parallel size per Ray accelerator type, "*" for any;
            overrides the memory-based choice.
        stable_node_resource (Optional[str]): Ray resource advertised by nodes that never leave; control
            plane actors are pinned there.
        rollout_node_resource (Optional[str]): Ray resource advertised by nodes elastic rollout may use.
        max_consecutive_failures (int): Trainer restarts without a completed step before giving up.
        borrow_rollout_after_s (Optional[float]): How long a restarting trainer waits for a replacement node
            before taking a matching rollout node; None never borrows.
    """

    enable: bool = False
    health_check_interval_s: float = 5.0
    request_retries: int = 60
    min_rollout_gpus: int = 0
    max_rollout_gpus: Optional[int] = None
    autoscale: bool = False
    policy: str = "static"
    up_wait_ratio: float = 0.2
    patience: int = 3
    scale_step_gpus: int = 8
    max_weight_fraction: float = 0.5
    tp_by_accelerator: dict[str, int] = field(default_factory=dict)
    stable_node_resource: Optional[str] = None
    rollout_node_resource: Optional[str] = None
    max_consecutive_failures: int = 5
    borrow_rollout_after_s: Optional[float] = 120.0

    def __post_init__(self):
        if self.autoscale and (self.max_rollout_gpus is None or self.rollout_node_resource is None):
            raise ValueError("elastic.autoscale requires max_rollout_gpus and rollout_node_resource")
        if self.max_rollout_gpus is not None and self.max_rollout_gpus < self.min_rollout_gpus:
            raise ValueError("elastic.max_rollout_gpus must be >= min_rollout_gpus")
        if not 0.0 < self.max_weight_fraction <= 1.0:
            raise ValueError(f"elastic.max_weight_fraction must be in (0, 1], got {self.max_weight_fraction}")
        if self.policy == "backlog" and self.min_rollout_gpus < 1:
            raise ValueError("elastic.policy=backlog needs min_rollout_gpus >= 1, or rollout could scale to nothing")


@dataclass
class BaseModelConfig(BaseConfig):
    """Base configuration for a model.
    Contains core settings for loading and initializing a pretrained model checkpoint.

    Args:
        path (str): Path to pretrained model weights.
        tokenizer_path (Optional[str]): Tokenizer path (defaults to actor's model path if not set).
        override_config (dict): Hugging Face config override.
        external_lib (Optional[str]): External model implementation (optional).
        trust_remote_code (bool): Whether to trust remote code from Hugging Face models.
        lora (dict[str, Any]): LoRA configuration dictionary.
    """

    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False
    lora: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleConfig(BaseConfig):
    """Configuration for external Python module, which can be loaded, executed (and optionally, ``import``ed).

    Args:
        path (str, optional): Path to the module file to load and execute.
        name (str, optional): Name of the module to ``import``. Format: ``"import.path.to.module"``.
            If ``None``, the module will be loaded with a hased name and
                will not be added to ``sys.modules``, thus can not be ``import``ed as ``name``.
    """

    path: Optional[str] = None
    name: Optional[str] = None
