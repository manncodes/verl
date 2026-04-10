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
Entry point for on-policy prompt distillation training.

This script orchestrates distributed training where a teacher model (with a system
prompt) guides a student model (without the system prompt) via KL divergence loss.

Usage:
    python -m recipe.prompt_distillation.main_prompt_distill \
        data.train_files=/path/to/train.parquet \
        actor_rollout_ref.model.path=/path/to/student_model \
        prompt_distillation.system_prompt_path=/path/to/system_prompt.txt \
        actor_rollout_ref.teacher.server_ip=localhost \
        actor_rollout_ref.teacher.server_port=15555

Prerequisites:
    Start the teacher server first (see recipe/gkd/teacher/start_server.sh):
        cd recipe/gkd/teacher
        bash start_server.sh --ckpt-path /path/to/teacher_model --port 15555
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.prompt_distillation.ray_trainer import PromptDistillTrainer

RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "false",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # Prevent hanging during weight sync between actor and rollout.
        # See: https://docs.vllm.ai/en/latest/usage/troubleshooting.html
        "NCCL_CUMEM_ENABLE": "0",
    },
}


@hydra.main(config_path="config", config_name="prompt_distill_trainer", version_base=None)
def main(config):
    """Main entry point for prompt distillation with Hydra configuration."""
    run_prompt_distill(config)


def run_prompt_distill(config) -> None:
    """Initialize Ray cluster and run distributed prompt distillation."""
    if not ray.is_initialized():
        ray.init(
            runtime_env=RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

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
    """Ray remote class for executing distributed prompt distillation."""

    def run(self, config):
        """Execute the prompt distillation training workflow."""
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download model checkpoint to local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
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

        # Megatron-only workers (reuse GKD's Megatron workers)
        if config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray import RayWorkerGroup

            from recipe.gkd.megatron_workers import (
                MegatronOnPolicyDistillActorWorker,
                MegatronOnPolicyDistillRolloutWorker,
            )

            rollout_cls = MegatronOnPolicyDistillRolloutWorker
            actor_cls = MegatronOnPolicyDistillActorWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(
                f"Strategy '{config.actor_rollout_ref.actor.strategy}' is not yet supported. "
                "Currently only 'megatron' strategy is supported for on-policy prompt distillation."
            )

        # Worker mapping and resource pools
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.Rollout: ray.remote(rollout_cls),
            Role.Actor: ray.remote(actor_cls),
        }

        assert config.trainer.n_gpus_per_node > 0
        assert config.trainer.nnodes > 0
        assert config.rollout.n_gpus_per_node > 0
        assert config.rollout.nnodes > 0

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

        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # Create datasets
        train_dataset = RLHFDataset(config.data.train_files, tokenizer, config.data, None)

        if config.data.val_files:
            val_dataset = RLHFDataset(config.data.val_files, tokenizer, config.data, None)
        else:
            val_dataset = None

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the prompt distillation trainer
        trainer = PromptDistillTrainer(
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

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
