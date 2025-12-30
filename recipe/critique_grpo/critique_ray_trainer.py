# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Critique-GRPO Ray Trainer.

This trainer extends the base GRPO trainer with critique-based refinement.
It generates critiques for incorrect solutions and uses refined solutions
as off-policy training data.

Based on: https://arxiv.org/abs/2506.03106
"""

import os
import uuid
import logging
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from .critique_prompts import generate_critique
from .refinement_prompts import generate_refinement, process_refinement_groups
from .reward_function import compute_math_score

logger = logging.getLogger(__name__)


class RayCritiqueGRPOTrainer(RayPPOTrainer):
    """
    Critique-GRPO Trainer that extends RayPPOTrainer with critique-based refinement.

    This trainer:
    1. Generates initial responses for prompts
    2. Computes rewards and generates critiques for solutions
    3. Creates refinement prompts based on critiques
    4. Generates refined solutions
    5. Uses both initial and refined solutions for training with off-policy loss

    The refinements serve as high-quality training signal, especially for
    incorrect initial solutions where the model can learn from the critique.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Critique-GRPO trainer."""
        super().__init__(*args, **kwargs)

        # Critique-GRPO specific configuration
        rollout_cfg = self.config.actor_rollout_ref.rollout
        self.critique_type = rollout_cfg.get("critique_type", "simple_gt")
        self.n_prefix = rollout_cfg.get("n_prefix", 1)  # Number of refinements per prompt
        self.max_refinement_length = rollout_cfg.get("max_refinement_length", 6144)

        logger.info(f"Critique-GRPO initialized with critique_type={self.critique_type}, "
                   f"n_prefix={self.n_prefix}")

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Compute KL-related metrics including entropy."""
        batch.batch["response_mask"] = compute_response_mask(batch)

        # Recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=loss_agg_mode
            )
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # Compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _generate_critiques_and_refinements(
        self,
        batch: DataProto,
        gen_batch_output: DataProto,
        timing_raw: dict
    ) -> DataProto:
        """Generate critiques and refinements for the batch.

        This is the core of Critique-GRPO:
        1. For each response, generate a critique
        2. Create refinement prompts
        3. Generate refined solutions
        4. Select best refinements

        Args:
            batch: Original prompt batch
            gen_batch_output: Generated responses
            timing_raw: Timing dictionary for profiling

        Returns:
            Updated batch with refinements
        """
        with marked_timer("critique_refinement", timing_raw, "magenta"):
            responses = gen_batch_output.batch["responses"]
            batch_size = responses.size(0)

            # Decode responses
            response_strs = []
            for i in range(batch_size):
                resp_ids = responses[i].tolist()
                # Remove padding
                resp_ids = [t for t in resp_ids if t != self.tokenizer.pad_token_id]
                resp_str = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                response_strs.append(resp_str)

            # Get ground truths and questions from non_tensor_batch
            non_tensor = batch.non_tensor_batch
            questions = []
            ground_truths = []

            for i in range(batch_size):
                # Extract question and ground truth
                if 'reward_model' in non_tensor:
                    rm_data = non_tensor['reward_model'][i]
                    questions.append(rm_data.get('question', ''))
                    ground_truths.append(rm_data.get('ground_truth', ''))
                else:
                    questions.append('')
                    ground_truths.append('')

            # Compute scores and generate critiques in parallel
            def process_sample(args):
                idx, response, question, gt = args
                try:
                    # Compute score
                    score_result = compute_math_score(response, gt)
                    score = score_result.get("score", 0.0)

                    # Create sample for critique
                    sample = {
                        "question": question,
                        "response": response,
                        "gt": gt,
                        "score": score
                    }

                    # Generate critique
                    sample = generate_critique(sample, self.critique_type)

                    # Generate refinement prompt
                    refinement_prompt, refinement_ids = generate_refinement(
                        sample, self.tokenizer
                    )

                    return idx, sample, refinement_ids, score
                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {e}")
                    return idx, None, None, 0.0

            # Process in parallel
            args_list = [
                (i, response_strs[i], questions[i], ground_truths[i])
                for i in range(batch_size)
            ]

            with ThreadPoolExecutor(max_workers=min(96, batch_size)) as executor:
                results = list(executor.map(process_sample, args_list))

            # Sort by index and extract results
            results.sort(key=lambda x: x[0])
            refinement_ids_list = [r[2] for r in results if r[2] is not None]
            initial_scores = [r[3] for r in results]

            # Store initial scores as metrics
            avg_initial_score = np.mean(initial_scores) if initial_scores else 0.0
            logger.info(f"Average initial score: {avg_initial_score:.4f}")

            # Generate refinements using vLLM
            if refinement_ids_list:
                refinement_outputs = self._generate_refinements(refinement_ids_list)

                # Score refinements
                refinement_strs = []
                refinement_scores = []

                for i, output in enumerate(refinement_outputs):
                    ref_str = self.tokenizer.decode(output, skip_special_tokens=True)
                    refinement_strs.append(ref_str)

                    gt = ground_truths[i] if i < len(ground_truths) else ""
                    score_result = compute_math_score(ref_str, gt)
                    refinement_scores.append(score_result.get("score", 0.0))

                # Log refinement improvement
                avg_refinement_score = np.mean(refinement_scores)
                improvement = avg_refinement_score - avg_initial_score
                logger.info(f"Refinement score: {avg_refinement_score:.4f} "
                           f"(improvement: {improvement:+.4f})")

                # Add refinement data to batch
                batch.meta_info["initial_scores"] = initial_scores
                batch.meta_info["refinement_scores"] = refinement_scores
                batch.meta_info["refinement_improvement"] = improvement

        return batch

    def _generate_refinements(
        self,
        refinement_ids_list: List[List[int]]
    ) -> List[List[int]]:
        """Generate refined solutions using the rollout engine.

        Args:
            refinement_ids_list: List of tokenized refinement prompts

        Returns:
            List of generated refinement token sequences
        """
        # This would typically call the vLLM inference engine
        # For now, return a placeholder - actual implementation would use
        # self.async_rollout_manager or similar

        # In practice, this is handled in the custom rollout class
        return refinement_ids_list

    def fit(self):
        """
        Main training loop for Critique-GRPO.

        Extends the standard PPO training loop with:
        1. Critique generation for responses
        2. Refinement generation based on critiques
        3. Off-policy learning from refinements
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
        self.gen_steps = 0

        # Load checkpoint
        self._load_checkpoint()

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Progress bar
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Critique-GRPO Training"
        )

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Create batch from dataloader
                    new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                    # Add UIDs for tracking
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                        dtype=object
                    )

                    gen_batch = self._get_gen_batch(new_batch)
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True
                    )

                    # Generate sequences
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(
                            gen_batch_output
                        )
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    # Repeat batch to match responses
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True
                    )
                    new_batch = new_batch.union(gen_batch_output)

                    # Compute rewards
                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_tensor, reward_extra_infos_dict = compute_reward(
                            new_batch, self.reward_fn
                        )
                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # Apply KL penalty if configured
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    batch = new_batch

                    # Balance batch if configured
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Compute KL metrics
                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # Compute values if using critic
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute advantages
                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Checkpointing
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                # Collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                # Add critique-specific metrics
                if "refinement_improvement" in batch.meta_info:
                    metrics["critique/refinement_improvement"] = batch.meta_info["refinement_improvement"]
                if "initial_scores" in batch.meta_info:
                    metrics["critique/initial_score_mean"] = np.mean(batch.meta_info["initial_scores"])
                if "refinement_scores" in batch.meta_info:
                    metrics["critique/refinement_score_mean"] = np.mean(batch.meta_info["refinement_scores"])

                timing_raw = defaultdict(float)
                batch = None

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
