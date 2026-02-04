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
"""
Main entry point for structured output RL training with GRPO.

This recipe trains LLMs to produce valid structured outputs (JSON, etc.)
using Group Relative Policy Optimization (GRPO) with schema validation rewards.

Supports three reward modes:
- fine_grained: Weighted combination of JSON parsability, schema validity,
  field coverage, and content correctness scores.
- binary: 0/1 reward based on whether output passes schema validation.
- crane: CRANE-style hybrid decoding where the model reasons freely then
  produces constrained output within delimiters.

The structured output reward manager is configured via:
  - reward_model.reward_manager=structured_output (selects StructuredOutputRewardManager)
  - reward_model.reward_kwargs (passes reward_mode, reasoning_delimiter, etc.)

Usage:
    python -m recipe.structured_output.main_structured_output \
        data.train_files=~/data/structured_output/structured_output_train.parquet \
        data.val_files=~/data/structured_output/structured_output_val.parquet \
        actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="structured_output_grpo", version_base=None)
def main(config):
    run_structured_output_training(config)


def run_structured_output_training(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

        # Select worker class based on rollout mode (sync vs async)
        rollout_mode = config.actor_rollout_ref.rollout.mode
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker if rollout_mode == "async" else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker if rollout_mode == "async" else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Reward model (if using model-based rewards)
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Reference model for KL divergence
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id

        # Load reward manager with structured output kwargs from config
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        reward_fn = load_reward_manager(config, tokenizer, 0, **reward_kwargs)
        val_reward_fn = load_reward_manager(config, tokenizer, 1, **reward_kwargs)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
