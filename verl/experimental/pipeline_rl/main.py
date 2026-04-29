# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hydra entrypoint for the PipelineRL recipe.

This is a near-drop-in replacement for
``verl.experimental.fully_async_policy.fully_async_main``: it constructs the
same Ray task runner topology but instantiates the PipelineRL-specialised
trainer and rollouter actors so that weight updates flow through
:class:`InflightWeightSync` and trajectories are version-tagged.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from time import time

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.pipeline_rl.pipeline_rl_rollouter import PipelineRLRollouter
from verl.experimental.pipeline_rl.pipeline_rl_trainer import PipelineRLTrainer
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


@ray.remote(num_cpus=1)
class PipelineRLTaskRunner:
    """Ray driver actor that wires up the PipelineRL trainer and rollouter."""

    def __init__(self):
        self.running = False
        self.components: dict = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[PIPELINE-RL MAIN] Starting PipelineRL training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[PIPELINE-RL MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self._create_trainer, config).result()
            executor.submit(self._create_rollouter, config).result()

        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(message_queue_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(message_queue_client))

        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        print("[PIPELINE-RL MAIN] Initial weight broadcast...")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

    def _create_rollouter(self, config) -> None:
        rollouter = PipelineRLRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter

    def _create_trainer(self, config) -> None:
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }
        trainer = PipelineRLTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer

    def _run_training_loop(self):
        self.running = True
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()
        futures = [rollouter_future, trainer_future]
        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)
                for future in done_futures:
                    try:
                        ray.get(future)
                    except Exception as e:
                        print(f"[PIPELINE-RL MAIN] Component failed with error: {e}")
                        for remaining in remaining_futures:
                            ray.cancel(remaining)
                        raise
                futures = remaining_futures
        except Exception as e:
            print(f"[PIPELINE-RL MAIN] Training failed: {e}")
            for f in futures:
                ray.cancel(f)
            raise
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())


@hydra.main(config_path="config", config_name="pipeline_rl_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("PipelineRL requires async_training config block")
    assert config.async_training.use_trainer_do_validate is False, (
        "use_trainer_do_validate is not supported by PipelineRL yet."
    )

    start_time = time()
    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(config, task_runner_class=PipelineRLTaskRunner)
    print(f"[PIPELINE-RL MAIN] total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
