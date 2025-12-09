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
GOLD (General Online Logit Distillation) Trainer with Ray-based single controller.

This trainer extends the OnPolicyDistillTrainer from GKD with:
1. Generalized JSD loss with configurable beta interpolation
2. Lambda-controlled on-policy/off-policy mixing
3. Temperature scaling for both teacher and student distributions

Key Features:
- On-policy generation: With probability lambda, generate responses from student
- Off-policy sampling: With probability (1-lambda), use dataset completions
- JSD loss: Generalized Jensen-Shannon Divergence with beta parameter
- Temperature: Control softmax sharpness for both distributions

References:
- GOLD: https://huggingface.co/docs/trl/main/gold_trainer
- GKD: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
"""

import random
import time
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from recipe.gkd.teacher import TeacherClient
from recipe.gkd.teacher_utils import get_teacher_knowledge
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]


class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results.

    This class holds futures for both on-policy rollout generation and
    teacher knowledge retrieval, enabling asynchronous processing.
    """

    def __init__(self, epoch, batch, gen_batch_output, is_on_policy=True):
        """
        Initialize the generation batch future.

        Args:
            epoch: Current training epoch
            batch: Input batch data
            gen_batch_output: Generated sequences from the student model (DataProtoFuture)
            is_on_policy: Whether this batch uses on-policy generated samples
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.teacher_batch_output = None
        self.is_on_policy = is_on_policy

    def set_teacher_batch_output(self, teacher_batch_output):
        """Set the teacher batch output for this generation batch."""
        self.teacher_batch_output = teacher_batch_output

    def get(self):
        """
        Get the actual results by resolving futures.

        Returns:
            tuple: (epoch, batch, gen_batch_result, teacher_batch_result) if teacher output exists
                   (epoch, batch, gen_batch_result) otherwise
        """
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
            self.gen_batch_output = gen_batch_result

        if self.teacher_batch_output is None:
            return self.epoch, self.batch, self.gen_batch_output

        if hasattr(self.teacher_batch_output, "get"):
            try:
                teacher_batch_result = self.teacher_batch_output.get()
            except Exception as e:
                teacher_batch_result = None
                print(f"Error getting teacher batch output: {e}")
        else:
            teacher_batch_result = self.teacher_batch_output

        return self.epoch, self.batch, self.gen_batch_output, teacher_batch_result


class GOLDTrainer(RayPPOTrainer):
    """
    GOLD (General Online Logit Distillation) Trainer with Ray backend.

    This trainer implements the GOLD algorithm which extends GKD with:
    - Generalized JSD loss instead of pure KL divergence
    - Lambda-controlled mixing of on-policy and off-policy samples
    - Temperature scaling for distribution smoothing

    The training loop alternates between:
    1. Generating responses from the student (on-policy) or using dataset (off-policy)
    2. Getting teacher's top-k predictions for distillation
    3. Updating the student model with JSD loss
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize the GOLD trainer.

        Args:
            config: Configuration object containing training parameters including:
                - gold.beta: JSD interpolation coefficient (0.0-1.0, default 0.5)
                - gold.lmbda: On-policy fraction (0.0-1.0, default 0.5)
                - gold.temperature: Softmax temperature (default 0.9)
            tokenizer: Tokenizer for encoding/decoding text
            role_worker_mapping: Mapping from roles to worker classes
            resource_pool_manager: Manager for Ray resource pools
            ray_worker_group_cls: Class for Ray worker groups
            train_dataset: Training dataset
            val_dataset: Validation dataset
            collate_fn: Function to collate data samples
            train_sampler: Sampler for training dataset
            device_name: Device for training
        """
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine, "GOLD trainer does not support hybrid engine"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.use_critic = False

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # Teacher configuration
        self.teacher_config = self.config.actor_rollout_ref.teacher
        self.n_server_workers = self.teacher_config.n_server_workers
        self.teacher_client = TeacherClient(
            self.teacher_config.server_ip,
            self.teacher_config.server_port,
            n_server_workers=self.n_server_workers,
        )

        # GOLD-specific parameters
        gold_config = OmegaConf.select(self.config, "gold") or {}
        self.beta = gold_config.get("beta", 0.5)
        self.lmbda = gold_config.get("lmbda", 0.5)
        self.temperature = gold_config.get("temperature", 0.9)
        self.seq_kd = gold_config.get("seq_kd", False)

        print(f"GOLD Trainer initialized with:")
        print(f"  beta (JSD interpolation): {self.beta}")
        print(f"  lmbda (on-policy fraction): {self.lmbda}")
        print(f"  temperature: {self.temperature}")
        print(f"  seq_kd: {self.seq_kd}")

        self.params_dtype = PrecisionType.to_dtype("bfloat16")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """Create train and validation dataloaders."""
        from verl.trainer.main_ppo import create_rl_sampler

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        if self.val_dataset:
            val_batch_size = self.config.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
            print(f"Size of train dataloader: {len(self.train_dataloader)}, "
                  f"Size of val dataloader: {len(self.val_dataloader)}")
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(self.config.trainer.total_training_steps, total_training_steps)

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config: {e}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        self.resource_pool_manager.create_resource_pool()

        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Rollout group
        rollout_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role="rollout",
        )
        resource_pool_to_cls[rollout_pool]["rollout"] = rollout_cls

        # Actor group
        actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        resource_pool_to_cls[actor_pool]["actor"] = actor_cls

        # Initialize WorkerGroups
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            time.sleep(20)

        self.rollout_wg = all_wg["rollout"]
        self.actor_wg = all_wg["actor"]

        # Initialize both groups
        self.rollout_wg.init_model()
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)

        from ray.util.collective import collective
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend="nccl",
            group_name="actor_rollout",
        )

    def sync_rollout_weights(self):
        """Synchronize weights from actor to rollout workers."""
        assert not self.hybrid_engine
        self.actor_wg.sync_rollout_weights()
        ray.get(self.rollout_wg.sync_rollout_weights())

    def _create_continuous_iterator(self):
        """Create a continuous data iterator across epochs."""
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    def _should_use_on_policy(self):
        """
        Determine whether to use on-policy generation for this step.

        Returns True with probability self.lmbda (on-policy fraction).
        """
        return random.random() < self.lmbda

    def _async_gen_next_batch(self, epoch, batch_dict, sync_before_generation=True):
        """
        Generate on-policy samples from the student model.

        Args:
            epoch: Current epoch
            batch_dict: Input batch dictionary
            sync_before_generation: Whether to sync weights before generation

        Returns:
            GenerationBatchFuture with on-policy samples
        """
        batch = DataProto.from_single_dict(batch_dict)

        # Keys to extract for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info["global_steps"] = self.global_steps

        # Sync weights if needed
        if sync_before_generation:
            self.sync_rollout_weights()

        # Non-blocking rollout
        gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)

        return GenerationBatchFuture(epoch, batch, gen_batch_output, is_on_policy=True)

    def _prepare_off_policy_batch(self, epoch, batch_dict):
        """
        Prepare off-policy batch using dataset completions.

        For off-policy training, we use the completions from the dataset
        rather than generating new ones from the student model.

        Args:
            epoch: Current epoch
            batch_dict: Input batch dictionary containing dataset completions

        Returns:
            GenerationBatchFuture with off-policy samples
        """
        batch = DataProto.from_single_dict(batch_dict)

        # For off-policy, we assume the batch contains pre-computed responses
        # from the dataset (similar to SFT data)
        batch.meta_info["global_steps"] = self.global_steps
        batch.meta_info["is_on_policy"] = False

        # Create a "fake" future that immediately returns the batch
        class ImmediateResult:
            def __init__(self, result):
                self._result = result

            def get(self):
                return self._result

        # The batch should already contain responses from the dataset
        return GenerationBatchFuture(
            epoch,
            batch,
            ImmediateResult(batch),
            is_on_policy=False,
        )

    def _async_get_teacher_knowledge(self, future: GenerationBatchFuture):
        """
        Asynchronously obtain teacher model knowledge for generated sequences.

        Args:
            future: Future object containing generated sequences

        Returns:
            GenerationBatchFuture with teacher knowledge set
        """
        _, _, gen_batch_output = future.get()
        gen_batch_output.meta_info["response_length"] = self.config.data.max_response_length

        future.set_teacher_batch_output(
            get_teacher_knowledge(gen_batch_output, self.teacher_client, self.n_server_workers, is_async=True)
        )
        return future

    def one_step_off_scheduler(self, continuous_iterator):
        """
        One-step-off scheduler with GOLD on/off-policy mixing.

        This scheduler extends the GKD one-step-off scheduler with
        probabilistic selection between on-policy and off-policy samples.
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            # Determine on-policy vs off-policy for this batch
            use_on_policy = self._should_use_on_policy()

            if i == 0:
                if use_on_policy:
                    with marked_timer("sync_rollout_weights", timing):
                        fut = self._async_gen_next_batch(epoch, batch_dict)
                else:
                    fut = self._prepare_off_policy_batch(epoch, batch_dict)

                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)

            if i == 1:
                use_on_policy = self._should_use_on_policy()
                if use_on_policy:
                    fut = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                else:
                    fut = self._prepare_off_policy_batch(epoch, batch_dict)

                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                timing = {}
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
            else:
                use_on_policy = self._should_use_on_policy()
                if use_on_policy:
                    with marked_timer("sync_rollout_weights", timing):
                        fut = self._async_gen_next_batch(epoch, batch_dict)
                else:
                    fut = self._prepare_off_policy_batch(epoch, batch_dict)

                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                timing = {}
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)

        # Last step
        with marked_timer("wait_prev_teacher", timing):
            prev_result = prev_fut.get()
        yield *prev_result, timing

    def two_step_off_scheduler(self, continuous_iterator):
        """
        Two-step-off scheduler with GOLD on/off-policy mixing.

        This scheduler extends the GKD two-step-off scheduler with
        probabilistic selection between on-policy and off-policy samples.
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            use_on_policy = self._should_use_on_policy()

            if i == 0:
                if use_on_policy:
                    with marked_timer("sync_rollout_weights", timing):
                        rollout_future = self._async_gen_next_batch(epoch, batch_dict)
                else:
                    rollout_future = self._prepare_off_policy_batch(epoch, batch_dict)
                continue

            elif i == 1:
                teacher_future = self._async_get_teacher_knowledge(rollout_future)
                if use_on_policy:
                    rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                else:
                    rollout_future = self._prepare_off_policy_batch(epoch, batch_dict)
                continue

            elif i == 2:
                with marked_timer("wait_prev_prev_teacher", timing):
                    result = teacher_future.get()
                with marked_timer("wait_prev_gen", timing):
                    teacher_future = self._async_get_teacher_knowledge(rollout_future)
                if use_on_policy:
                    rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                else:
                    rollout_future = self._prepare_off_policy_batch(epoch, batch_dict)
                yield *result, timing
                timing = {}

            else:
                with marked_timer("wait_prev_prev_teacher", timing):
                    result = teacher_future.get()
                with marked_timer("wait_prev_gen", timing):
                    teacher_future = self._async_get_teacher_knowledge(rollout_future)
                if use_on_policy:
                    with marked_timer("sync_rollout_weights", timing):
                        rollout_future = self._async_gen_next_batch(epoch, batch_dict)
                else:
                    rollout_future = self._prepare_off_policy_batch(epoch, batch_dict)
                yield *result, timing
                timing = {}

        # Second to last step
        with marked_timer("wait_prev_prev_teacher", timing):
            result = teacher_future.get()
        with marked_timer("wait_prev_gen", timing):
            teacher_future = self._async_get_teacher_knowledge(rollout_future)
        yield *result, timing

        # Last step
        with marked_timer("wait_prev_prev_teacher", timing):
            result = teacher_future.get()
        yield *result, timing

    def fit(self):
        """
        Main training loop for GOLD.

        This loop:
        1. Selects on-policy or off-policy samples based on lambda
        2. Gets teacher knowledge for distillation
        3. Updates the student model with JSD loss
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="GOLD Training")

        self.global_steps += 1
        max_steps_duration = 0

        continuous_iterator = self._create_continuous_iterator()

        scheduler_type = self.config.trainer.scheduler
        if scheduler_type == "one_step_off":
            scheduler = self.one_step_off_scheduler(continuous_iterator)
        elif scheduler_type == "two_step_off":
            scheduler = self.two_step_off_scheduler(continuous_iterator)
        else:
            raise TypeError(f"Unrecognized scheduler type: {scheduler_type}")

        # Track on-policy vs off-policy statistics
        on_policy_count = 0
        off_policy_count = 0

        while True:
            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )
            if do_profile:
                self.rollout_wg.start_profile()
                self.actor_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                _, batch, gen_batch_output, teacher_batch_output, schedule_timing = next(scheduler)

                if teacher_batch_output is None:
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        self._save_checkpoint()
                    print("Error in getting teacher knowledge. Skip this batch.")
                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        progress_bar.close()
                        return
                    continue

                timing_raw.update(schedule_timing)

                # Track on-policy vs off-policy
                is_on_policy = gen_batch_output.meta_info.get("is_on_policy", True)
                if is_on_policy:
                    on_policy_count += 1
                else:
                    off_policy_count += 1

                gen_timing = gen_batch_output.meta_info.pop("timing", {})
                for k, v in gen_timing.items():
                    if isinstance(v, list):
                        array_v = np.array(v)
                        timing_raw[k + "_mean"] = array_v.mean().item()
                        timing_raw[k + "_min"] = array_v.min().item()
                        timing_raw[k + "_max"] = array_v.max().item()
                        timing_raw[k] = array_v.max().item()
                    else:
                        timing_raw[k] = v

                timing_raw.update(teacher_batch_output.meta_info.pop("timing"))

                # Response length statistics
                response_lens = (
                    (gen_batch_output.batch["responses"] != self.tokenizer.pad_token_id).sum(dim=-1).tolist()
                )
                metrics.update({
                    "response_seq_len/average": sum(response_lens) / len(response_lens),
                    "response_seq_len/max": max(response_lens),
                    "response_seq_len/min": min(response_lens),
                    "response_seq_len/max_count": response_lens.count(max(response_lens)),
                    "response_seq_len/min_count": response_lens.count(min(response_lens)),
                })

                # GOLD-specific metrics
                total_samples = on_policy_count + off_policy_count
                metrics.update({
                    "gold/on_policy_ratio": on_policy_count / total_samples if total_samples > 0 else 0,
                    "gold/beta": self.beta,
                    "gold/lmbda": self.lmbda,
                    "gold/temperature": self.temperature,
                    "gold/is_on_policy": float(is_on_policy),
                })

                # Merge outputs
                batch = batch.union(gen_batch_output)

                # Debug print
                one_attention_mask = batch.batch["attention_mask"][0].to(torch.bool)
                one_sentence = batch.batch["input_ids"][0]
                print("INFO:", "generate text done." if is_on_policy else "using dataset completion.")
                print("DEBUG:", self.tokenizer.decode(one_sentence[one_attention_mask].tolist()))

                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                batch = batch.union(teacher_batch_output)

                # Add GOLD parameters to batch for worker
                batch.meta_info["gold_beta"] = self.beta
                batch.meta_info["gold_temperature"] = self.temperature

                # Update actor with JSD loss
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self.actor_wg.update_actor(batch)

                print("INFO:", "update actor done.")
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                # Save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            # Metrics and bookkeeping
            steps_duration = timing_raw["step"]
            max_steps_duration = max(max_steps_duration, steps_duration)
            metrics["training/global_step"] = self.global_steps

            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.rollout_wg.stop_profile()
                self.actor_wg.stop_profile()

            if is_last_step:
                progress_bar.close()
                print(f"\nGOLD Training Complete!")
                print(f"  On-policy samples: {on_policy_count}")
                print(f"  Off-policy samples: {off_policy_count}")
                print(f"  On-policy ratio: {on_policy_count / (on_policy_count + off_policy_count):.2%}")
                return
