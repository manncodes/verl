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
Cascaded RL Trainer for Multi-Task Domain-Wise Reinforcement Learning.

This implementation is based on the Nemotron-Cascade paper:
"Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models"
https://arxiv.org/abs/2512.13607

Key features:
- Sequential domain-wise RL training: SFT → RLHF → IF-RL → Math RL → Code RL → SWE RL
- GRPO algorithm with on-policy training (no KL penalty)
- Domain-specific reward functions and verifiers
- Checkpoint transfer between stages
- RL stages rarely degrade prior domain performance
"""

import os
import json
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pprint import pprint

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


@dataclass
class StageConfig:
    """Configuration for a single Cascade RL stage."""

    name: str  # Stage name: "rlhf", "if_rl", "math_rl", "code_rl", "swe_rl"
    domain: str  # Domain identifier for reward routing

    # Dataset configuration
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)

    # Training hyperparameters
    total_epochs: int = 1
    total_training_steps: Optional[int] = None
    train_batch_size: int = 128
    rollout_n: int = 8  # Number of rollouts per prompt

    # Learning rate and optimizer
    learning_rate: float = 2e-6

    # Generation parameters
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 8192

    # GRPO specific (following Nemotron-Cascade: no KL penalty)
    use_kl_in_reward: bool = False
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0
    entropy_coef: float = 0.0

    # Advantage estimator
    adv_estimator: str = "grpo"
    norm_adv_by_std_in_grpo: bool = True

    # Reward configuration
    reward_fn_path: Optional[str] = None  # Path to custom reward function module
    reward_fn_name: Optional[str] = None  # Function name in the module

    # Stage-specific settings
    skip_validation: bool = False
    save_checkpoint: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for config merging."""
        return {
            "name": self.name,
            "domain": self.domain,
            "data": {
                "train_files": self.train_files,
                "val_files": self.val_files,
                "train_batch_size": self.train_batch_size,
            },
            "trainer": {
                "total_epochs": self.total_epochs,
                "total_training_steps": self.total_training_steps,
            },
            "algorithm": {
                "adv_estimator": self.adv_estimator,
                "use_kl_in_reward": self.use_kl_in_reward,
                "norm_adv_by_std_in_grpo": self.norm_adv_by_std_in_grpo,
            },
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": self.use_kl_loss,
                    "kl_loss_coef": self.kl_loss_coef,
                    "optim": {
                        "lr": self.learning_rate,
                    },
                },
                "rollout": {
                    "n": self.rollout_n,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens,
                },
            },
        }


@dataclass
class CascadeRLConfig:
    """Configuration for the full Cascade RL training pipeline."""

    # List of stages to run sequentially
    stages: List[StageConfig] = field(default_factory=list)

    # Base configuration (shared across stages)
    base_config: Optional[DictConfig] = None

    # Checkpoint management
    cascade_checkpoint_dir: str = "./cascaded_rl_checkpoints"
    resume_from_stage: Optional[str] = None  # Stage name to resume from

    # Whether to validate after each stage
    validate_after_stage: bool = True

    # Logging
    log_stage_transitions: bool = True


class CascadedRLTrainer:
    """
    Cascaded RL Trainer for multi-task domain-wise reinforcement learning.

    This trainer orchestrates sequential RL training across multiple domains,
    following the Nemotron-Cascade approach:
    - Each stage uses GRPO with on-policy training
    - No KL penalty (simplified REINFORCE with group-normalized rewards)
    - Domain-specific reward functions for each stage
    - Checkpoint transfer between stages

    Example pipeline:
        Stage 1 (RLHF): General preference alignment with 72B reward model
        Stage 2 (IF-RL): Instruction following with rule-based + RLHF rewards
        Stage 3 (Math-RL): Mathematical reasoning with rule-based verifiers
        Stage 4 (Code-RL): Code generation with execution-based rewards
        Stage 5 (SWE-RL): Software engineering with hybrid rewards
    """

    def __init__(
        self,
        cascade_config: CascadeRLConfig,
        tokenizer,
        role_worker_mapping: Dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn_registry: Optional[Dict[str, Callable]] = None,
        device_name: Optional[str] = None,
    ):
        """
        Initialize the Cascaded RL Trainer.

        Args:
            cascade_config: Configuration for the cascade pipeline
            tokenizer: Tokenizer for text processing
            role_worker_mapping: Mapping from roles to worker classes
            resource_pool_manager: Manager for Ray resource pools
            ray_worker_group_cls: Class for Ray worker groups
            processor: Optional data processor for multimodal data
            reward_fn_registry: Dictionary mapping domain names to reward functions
            device_name: Device name for training
        """
        self.cascade_config = cascade_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name

        # Reward function registry for domain-specific rewards
        self.reward_fn_registry = reward_fn_registry or {}

        # Track training state
        self.current_stage_idx = 0
        self.stage_metrics: Dict[str, Dict] = {}
        self.completed_stages: List[str] = []

        # Ensure checkpoint directory exists
        os.makedirs(cascade_config.cascade_checkpoint_dir, exist_ok=True)

    def _get_stage_checkpoint_dir(self, stage_name: str) -> str:
        """Get the checkpoint directory for a specific stage."""
        return os.path.join(self.cascade_config.cascade_checkpoint_dir, f"stage_{stage_name}")

    def _merge_stage_config(self, stage: StageConfig) -> DictConfig:
        """Merge stage-specific config with base config."""
        base = OmegaConf.create(self.cascade_config.base_config) if self.cascade_config.base_config else OmegaConf.create({})
        stage_overrides = OmegaConf.create(stage.to_dict())

        # Deep merge with stage overrides taking precedence
        with open_dict(base):
            merged = OmegaConf.merge(base, stage_overrides)

            # Update checkpoint directory for this stage
            merged.trainer.default_local_dir = self._get_stage_checkpoint_dir(stage.name)

            # Set experiment name to include stage
            if OmegaConf.select(merged, "trainer.experiment_name"):
                merged.trainer.experiment_name = f"{merged.trainer.experiment_name}_{stage.name}"
            else:
                merged.trainer.experiment_name = f"cascade_rl_{stage.name}"

        return merged

    def _get_reward_fn_for_stage(self, stage: StageConfig) -> Optional[Callable]:
        """Get the reward function for a specific stage."""
        # First check the registry
        if stage.domain in self.reward_fn_registry:
            return self.reward_fn_registry[stage.domain]

        # Try to load from path if specified
        if stage.reward_fn_path and stage.reward_fn_name:
            import importlib.util
            spec = importlib.util.spec_from_file_location("reward_module", stage.reward_fn_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, stage.reward_fn_name)

        # Return None to use default reward function
        return None

    def _load_previous_stage_checkpoint(self, current_stage_idx: int, trainer: RayPPOTrainer):
        """Load checkpoint from the previous stage."""
        if current_stage_idx == 0:
            return

        prev_stage = self.cascade_config.stages[current_stage_idx - 1]
        prev_checkpoint_dir = self._get_stage_checkpoint_dir(prev_stage.name)

        # Find the latest checkpoint from previous stage
        checkpoint_path = find_latest_ckpt_path(prev_checkpoint_dir)

        if checkpoint_path:
            print(f"Loading checkpoint from previous stage '{prev_stage.name}': {checkpoint_path}")
            actor_path = os.path.join(checkpoint_path, "actor")
            trainer.actor_rollout_wg.load_checkpoint(actor_path)

            # Also load critic if it exists and is used
            if trainer.use_critic:
                critic_path = os.path.join(checkpoint_path, str(Role.Critic))
                if os.path.exists(critic_path):
                    trainer.critic_wg.load_checkpoint(critic_path)
        else:
            print(f"Warning: No checkpoint found from previous stage '{prev_stage.name}'")

    def _save_stage_summary(self, stage: StageConfig, metrics: Dict):
        """Save a summary of the completed stage."""
        summary = {
            "stage_name": stage.name,
            "domain": stage.domain,
            "final_metrics": metrics,
            "config": stage.to_dict(),
        }

        summary_path = os.path.join(
            self._get_stage_checkpoint_dir(stage.name),
            "stage_summary.json"
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def _find_resume_stage_idx(self) -> int:
        """Find the stage index to resume from."""
        if self.cascade_config.resume_from_stage is None:
            return 0

        for idx, stage in enumerate(self.cascade_config.stages):
            if stage.name == self.cascade_config.resume_from_stage:
                return idx

        raise ValueError(f"Stage '{self.cascade_config.resume_from_stage}' not found in config")

    def train_stage(self, stage_idx: int) -> Dict:
        """
        Train a single stage of the cascade.

        Args:
            stage_idx: Index of the stage to train

        Returns:
            Dictionary of final metrics for this stage
        """
        stage = self.cascade_config.stages[stage_idx]
        print(f"\n{'='*60}")
        print(f"Starting Stage {stage_idx + 1}/{len(self.cascade_config.stages)}: {stage.name}")
        print(f"Domain: {stage.domain}")
        print(f"{'='*60}\n")

        # Merge configs for this stage
        stage_config = self._merge_stage_config(stage)

        # Get reward function for this stage
        reward_fn = self._get_reward_fn_for_stage(stage)
        val_reward_fn = reward_fn  # Use same for validation by default

        # Create datasets for this stage
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(
            stage_config.data.train_files,
            stage_config.data,
            self.tokenizer,
            self.processor,
        )

        val_dataset = None
        if stage_config.data.val_files and not stage.skip_validation:
            val_dataset = create_rl_dataset(
                stage_config.data.val_files,
                stage_config.data,
                self.tokenizer,
                self.processor,
            )

        # Create the trainer for this stage
        trainer = RayPPOTrainer(
            config=stage_config,
            tokenizer=self.tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager,
            ray_worker_group_cls=self.ray_worker_group_cls,
            processor=self.processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device_name=self.device_name,
        )

        # Initialize workers (only for first stage, reuse for subsequent)
        if stage_idx == 0:
            trainer.init_workers()
        else:
            # Reuse workers from previous stage but load new checkpoint
            self._load_previous_stage_checkpoint(stage_idx, trainer)

        # Train this stage
        trainer.fit()

        # Get final metrics
        final_metrics = {
            "stage": stage.name,
            "global_steps": trainer.global_steps,
        }

        # Save stage summary
        self._save_stage_summary(stage, final_metrics)

        self.completed_stages.append(stage.name)
        self.stage_metrics[stage.name] = final_metrics

        print(f"\n{'='*60}")
        print(f"Completed Stage: {stage.name}")
        pprint(final_metrics)
        print(f"{'='*60}\n")

        return final_metrics

    def fit(self):
        """
        Run the full Cascade RL training pipeline.

        This trains each stage sequentially, transferring checkpoints
        between stages and using domain-specific reward functions.
        """
        print("\n" + "="*60)
        print("Starting Cascaded RL Training Pipeline")
        print(f"Total stages: {len(self.cascade_config.stages)}")
        print("Stages: " + " → ".join([s.name for s in self.cascade_config.stages]))
        print("="*60 + "\n")

        # Find starting stage (for resumption)
        start_idx = self._find_resume_stage_idx()
        if start_idx > 0:
            print(f"Resuming from stage {start_idx + 1}: {self.cascade_config.stages[start_idx].name}")

        # Train each stage sequentially
        for stage_idx in range(start_idx, len(self.cascade_config.stages)):
            self.current_stage_idx = stage_idx
            self.train_stage(stage_idx)

        # Print final summary
        print("\n" + "="*60)
        print("Cascaded RL Training Complete!")
        print("="*60)
        print("\nStage Summary:")
        for stage_name, metrics in self.stage_metrics.items():
            print(f"  {stage_name}: {metrics}")
        print("="*60 + "\n")

        return self.stage_metrics


def create_default_cascade_stages() -> List[StageConfig]:
    """
    Create default Cascade RL stages following Nemotron-Cascade.

    Pipeline: RLHF → IF-RL → Math-RL → Code-RL → SWE-RL
    """
    return [
        StageConfig(
            name="rlhf",
            domain="rlhf",
            total_training_steps=800,
            train_batch_size=128,
            rollout_n=8,
            learning_rate=2e-6,
            temperature=0.6,
            top_p=0.95,
            use_kl_in_reward=False,
            use_kl_loss=False,
        ),
        StageConfig(
            name="if_rl",
            domain="instruction_following",
            total_training_steps=500,
            train_batch_size=128,
            rollout_n=8,
            learning_rate=2e-6,
            temperature=0.6,
            top_p=0.95,
        ),
        StageConfig(
            name="math_rl",
            domain="math",
            total_training_steps=500,
            train_batch_size=128,
            rollout_n=8,
            learning_rate=2e-6,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=16384,  # Math may need longer reasoning
        ),
        StageConfig(
            name="code_rl",
            domain="code",
            total_training_steps=500,
            train_batch_size=128,
            rollout_n=8,
            learning_rate=2e-6,
            temperature=0.7,  # Slightly higher for code exploration
            top_p=0.95,
            max_new_tokens=16384,
        ),
        StageConfig(
            name="swe_rl",
            domain="swe",
            total_training_steps=300,
            train_batch_size=64,  # Smaller batch for longer sequences
            rollout_n=4,
            learning_rate=1e-6,  # Lower LR for fine-tuning
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=32768,  # SWE tasks often need very long context
        ),
    ]
