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
"""
Main entry point for GOLD (General Online Logit Distillation) training.

GOLD is an on-policy knowledge distillation algorithm that extends GKD with:
- Generalized JSD loss with configurable beta interpolation
- Lambda-controlled on-policy/off-policy mixing
- Temperature scaling for distribution smoothing

Usage:
    python -m recipe.gold.main_gold \
        --config-path=recipe/gold/config \
        --config-name=gold_trainer \
        actor_rollout_ref.model.path=/path/to/MODEL \
        data.train_files=/path/to/train.parquet \
        gold.beta=0.5 \
        gold.lmbda=0.5 \
        gold.temperature=0.9

References:
    - GOLD: https://huggingface.co/docs/trl/main/gold_trainer
    - GKD: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.gold.ray_trainer import GOLDTrainer

RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "false",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights
        "NCCL_CUMEM_ENABLE": "0",
    },
}


@hydra.main(config_path="config", config_name="gold_trainer", version_base=None)
def main(config):
    """Main entry point for GOLD training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    run_gold_training(config)


def run_gold_training(config) -> None:
    """Initialize Ray cluster and run distributed GOLD training.

    Args:
        config: Training configuration object containing all parameters
                for distributed GOLD training including model paths,
                GOLD hyperparameters (beta, lmbda, temperature), and
                training settings.
    """
    if not ray.is_initialized():
        ray.init(
            runtime_env=RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # Handle Nsight profiling
    if (
        config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()

    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing distributed GOLD training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main GOLD training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and the GOLD trainer, then starts training.

        Args:
            config: Training configuration object containing all parameters.
        """
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        # Print GOLD-specific configuration
        gold_config = OmegaConf.select(config, "gold") or {}
        print("\n=== GOLD Training Configuration ===")
        print(f"  beta (JSD interpolation): {gold_config.get('beta', 0.5)}")
        print(f"  lmbda (on-policy fraction): {gold_config.get('lmbda', 0.5)}")
        print(f"  temperature: {gold_config.get('temperature', 0.9)}")
        print(f"  seq_kd: {gold_config.get('seq_kd', False)}")
        print("=" * 40 + "\n")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download checkpoint to local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Initialize tokenizer
        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Version validation for vllm
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Import GOLD workers
        if config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray import RayWorkerGroup

            from .megatron_workers import (
                MegatronGOLDActorWorker,
                MegatronGOLDRolloutWorker,
            )

            rollout_cls = MegatronGOLDRolloutWorker
            actor_cls = MegatronGOLDActorWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported for GOLD")

        # Worker mapping and resource pools
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.Rollout: ray.remote(rollout_cls),
            Role.Actor: ray.remote(actor_cls),
        }

        # Validate configuration
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be > 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be > 0"
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be > 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be > 0"

        actor_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes

        resource_pool_spec = {
            "rollout_pool": rollout_pool,
            "actor_pool": actor_pool,
        }
        mapping = {
            Role.Rollout: "rollout_pool",
            Role.Actor: "actor_pool",
        }
        print(f"resource_pool_spec: {resource_pool_spec}")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Create datasets
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        train_dataset = RLHFDataset(config.data.train_files, tokenizer, config.data, None)

        if config.data.val_files:
            val_dataset = RLHFDataset(config.data.val_files, tokenizer, config.data, None)
        else:
            val_dataset = None

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize GOLD trainer
        trainer = GOLDTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )

        # Initialize workers and start training
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
