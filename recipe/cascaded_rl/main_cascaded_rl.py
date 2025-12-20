# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The verl Authors
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
Main entry point for Cascaded RL training.

This script orchestrates multi-task domain-wise RL training following the
Nemotron-Cascade methodology:
- Sequential domain training: RLHF → IF-RL → Math-RL → Code-RL → SWE-RL
- GRPO with on-policy training (no KL penalty)
- Domain-specific reward functions
- Checkpoint transfer between stages

Usage:
    python main_cascaded_rl.py --config config/cascaded_rl_config.yaml

References:
    - Nemotron-Cascade: https://arxiv.org/abs/2512.13607
    - verl: https://github.com/volcengine/verl
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

import ray
import torch

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType

from cascaded_trainer import CascadedRLTrainer, CascadeRLConfig, StageConfig
from tasks.reward_functions import (
    RLHFRewardManager,
    IFRewardManager,
    MathRewardManager,
    CodeRewardManager,
    SWERewardManager,
    CascadeRewardManager,
)


def get_tokenizer(config):
    """Load tokenizer from config."""
    from transformers import AutoTokenizer

    tokenizer_path = config.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(config):
    """Load processor for multimodal models if needed."""
    if not config.data.get("is_multimodal", False):
        return None

    from transformers import AutoProcessor
    processor_path = config.actor_rollout_ref.model.path
    return AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)


def create_stage_configs_from_yaml(config: DictConfig) -> list[StageConfig]:
    """Create StageConfig objects from YAML config."""
    stages = []

    for stage_cfg in config.cascade.stages:
        stage = StageConfig(
            name=stage_cfg.name,
            domain=stage_cfg.domain,
            train_files=list(stage_cfg.get("train_files", [])),
            val_files=list(stage_cfg.get("val_files", [])),
            total_epochs=stage_cfg.get("total_epochs", 1),
            total_training_steps=stage_cfg.get("total_training_steps", None),
            train_batch_size=stage_cfg.get("train_batch_size", 128),
            rollout_n=stage_cfg.get("rollout_n", 8),
            learning_rate=stage_cfg.get("learning_rate", 2e-6),
            temperature=stage_cfg.get("temperature", 0.6),
            top_p=stage_cfg.get("top_p", 0.95),
            max_new_tokens=stage_cfg.get("max_new_tokens", 8192),
            use_kl_in_reward=stage_cfg.get("use_kl_in_reward", False),
            use_kl_loss=stage_cfg.get("use_kl_loss", False),
            kl_loss_coef=stage_cfg.get("kl_loss_coef", 0.0),
            entropy_coef=stage_cfg.get("entropy_coef", 0.0),
            adv_estimator=stage_cfg.get("adv_estimator", "grpo"),
            norm_adv_by_std_in_grpo=stage_cfg.get("norm_adv_by_std_in_grpo", True),
            reward_fn_path=stage_cfg.get("reward_fn_path", None),
            reward_fn_name=stage_cfg.get("reward_fn_name", None),
            skip_validation=stage_cfg.get("skip_validation", False),
            save_checkpoint=stage_cfg.get("save_checkpoint", True),
        )
        stages.append(stage)

    return stages


def create_reward_fn_registry(tokenizer) -> dict:
    """Create reward function registry for all domains."""
    return {
        "rlhf": RLHFRewardManager(tokenizer),
        "instruction_following": IFRewardManager(tokenizer),
        "math": MathRewardManager(tokenizer),
        "code": CodeRewardManager(tokenizer),
        "swe": SWERewardManager(tokenizer),
        # Add unified manager for mixed domains
        "mixed": CascadeRewardManager(tokenizer),
    }


def create_role_worker_mapping(config) -> dict[Role, WorkerType]:
    """Create role to worker class mapping based on config."""
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.workers.fsdp_workers import CriticWorker

    mapping = {}

    # Actor/Rollout/Ref combined worker
    if config.actor_rollout_ref.hybrid_engine:
        mapping[Role.ActorRolloutRef] = ActorRolloutRefWorker

    # Critic (optional, GRPO doesn't need critic)
    use_critic = config.get("critic", {}).get("enable", False)
    if use_critic:
        mapping[Role.Critic] = CriticWorker

    return mapping


def create_resource_pool_manager(config) -> ResourcePoolManager:
    """Create resource pool manager based on config."""
    # Default: all workers share the same pool
    n_gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1

    resource_pool_spec = {
        "default": [n_gpus_per_node],  # Single node
    }

    mapping = {
        Role.ActorRolloutRef: "default",
        Role.ActorRollout: "default",
        Role.Critic: "default",
        Role.RefPolicy: "default",
        Role.RewardModel: "default",
    }

    # Override from config if provided
    if "resource_pool" in config:
        if "spec" in config.resource_pool:
            resource_pool_spec = OmegaConf.to_container(config.resource_pool.spec)
        if "mapping" in config.resource_pool:
            mapping = {Role[k]: v for k, v in config.resource_pool.mapping.items()}

    return ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )


@hydra.main(config_path="config", config_name="cascaded_rl_config", version_base=None)
def main(config: DictConfig):
    """Main entry point for Cascaded RL training."""

    print("="*60)
    print("Cascaded RL Training - Multi-Task Domain-Wise RL")
    print("Based on Nemotron-Cascade methodology")
    print("="*60)

    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(config))

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "false",
                }
            }
        )

    # Load tokenizer and processor
    tokenizer = get_tokenizer(config)
    processor = get_processor(config)

    # Create stage configurations
    stages = create_stage_configs_from_yaml(config)
    print(f"\nConfigured {len(stages)} training stages:")
    for i, stage in enumerate(stages):
        print(f"  {i+1}. {stage.name} (domain: {stage.domain})")

    # Create cascade config
    cascade_config = CascadeRLConfig(
        stages=stages,
        base_config=config,
        cascade_checkpoint_dir=config.cascade.get("checkpoint_dir", "./cascaded_rl_checkpoints"),
        resume_from_stage=config.cascade.get("resume_from_stage", None),
        validate_after_stage=config.cascade.get("validate_after_stage", True),
        log_stage_transitions=config.cascade.get("log_stage_transitions", True),
    )

    # Create reward function registry
    reward_fn_registry = create_reward_fn_registry(tokenizer)

    # Create role worker mapping
    role_worker_mapping = create_role_worker_mapping(config)

    # Create resource pool manager
    resource_pool_manager = create_resource_pool_manager(config)

    # Create and run the cascaded trainer
    trainer = CascadedRLTrainer(
        cascade_config=cascade_config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        processor=processor,
        reward_fn_registry=reward_fn_registry,
        device_name=config.trainer.get("device", "cuda"),
    )

    # Run training
    final_metrics = trainer.fit()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    pprint(final_metrics)

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
