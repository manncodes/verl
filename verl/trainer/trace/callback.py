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
TRACE Training Callback: Integrates TRACE detection into the RL training loop.

This callback hooks into the verl PPO training loop to:
1. Compute TRACE scores for rollout batches
2. Collect baseline scores during initial training
3. Detect hacking samples after calibration
4. Optionally filter or penalize hacking samples
5. Run loophole discovery at configured intervals
6. Log metrics throughout training

Reference:
    Wang et al. (2025). "Is It Thinking or Cheating? Detecting Implicit
    Reward Hacking by Measuring Reasoning Effort." arXiv:2510.01367
"""

import logging
from typing import Any, Callable, Optional

import torch

from verl import DataProto
from verl.trainer.trace.config import TRACEConfig
from verl.trainer.trace.core import (
    compute_expected_reward_at_truncation,
    compute_trace_auc,
    truncate_response_batch,
)
from verl.trainer.trace.detector import TRACEDetector
from verl.trainer.trace.loophole_discovery import TRACELoopholeDiscovery

logger = logging.getLogger(__name__)

__all__ = ["TRACECallback"]


class TRACECallback:
    """Callback for integrating TRACE detection into the PPO training loop.

    This callback is designed to be called at specific points in the training
    loop to compute TRACE scores, detect hacking, and apply corrective actions.

    Typical usage in the training loop:

        trace_callback = TRACECallback(trace_config, tokenizer, reward_fn)

        for step, batch in enumerate(dataloader):
            # After rollout generation
            batch = trace_callback.on_after_rollout(batch, step, generate_fn)

            # After reward computation
            batch = trace_callback.on_after_reward(batch, step)

            # Get metrics for logging
            metrics.update(trace_callback.get_metrics())

    Args:
        config: TRACE configuration.
        tokenizer: Tokenizer for encoding/decoding.
        reward_fn: Reward scoring function (same as used for training).
    """

    def __init__(
        self,
        config: TRACEConfig,
        tokenizer: Any,
        reward_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.detector = TRACEDetector(config)
        self.loophole_discovery: Optional[TRACELoopholeDiscovery] = None
        if config.enable_loophole_discovery:
            self.loophole_discovery = TRACELoopholeDiscovery(config)
        self._current_metrics: dict[str, float] = {}

    def on_after_rollout(
        self,
        data: DataProto,
        step: int,
        generate_fn: Optional[Callable] = None,
    ) -> DataProto:
        """Called after rollout generation to compute TRACE scores.

        This is the main entry point for TRACE score computation. It truncates
        the generated CoTs at various fractions, generates completions, computes
        rewards, and calculates the TRACE AUC for each sample.

        Args:
            data: The rollout batch containing prompts, responses, and metadata.
            step: Current training step.
            generate_fn: Function for generating completions from truncated prefixes.
                If None, uses a simplified scoring approach that only evaluates
                the existing truncated response without re-generation.

        Returns:
            Modified DataProto with "trace_scores" added to the batch.
        """
        if not self.config.enable:
            return data

        # Only run at configured frequency (or during baseline collection)
        is_baseline_phase = step <= self.config.baseline_steps
        is_detection_step = step % self.config.detection_frequency == 0

        if not is_baseline_phase and not is_detection_step:
            return data

        logger.info(f"TRACE: computing scores at step {step}")

        if generate_fn is not None and self.reward_fn is not None:
            trace_scores = self._compute_scores_with_generation(
                data, generate_fn
            )
        else:
            trace_scores = self._compute_scores_simplified(data)

        # Store TRACE scores in the batch
        data.batch["trace_scores"] = trace_scores

        return data

    def on_after_reward(
        self,
        data: DataProto,
        step: int,
    ) -> DataProto:
        """Called after reward computation to apply TRACE-based corrections.

        Performs baseline collection, detection, and optional filtering/penalty
        based on TRACE scores.

        Args:
            data: The batch with computed rewards and TRACE scores.
            step: Current training step.

        Returns:
            Modified DataProto with potential reward adjustments.
        """
        if not self.config.enable:
            return data

        if "trace_scores" not in data.batch.keys():
            return data

        trace_scores = data.batch["trace_scores"]

        # Phase 1: Baseline collection
        if step <= self.config.baseline_steps:
            self.detector.update_baseline(trace_scores)
            if step == self.config.baseline_steps:
                self.detector.calibrate()
                logger.info("TRACE baseline calibration complete.")
            self._current_metrics = self.detector.get_metrics()
            return data

        # Phase 2: Detection
        if not self.detector.is_calibrated:
            return data

        if step % self.config.detection_frequency != 0:
            self._current_metrics = self.detector.get_metrics()
            return data

        detection = self.detector.detect(trace_scores)
        is_hacking = detection["is_hacking"]

        self._current_metrics = {
            "trace/hacking_fraction": detection["hacking_fraction"],
            "trace/mean_score": detection["mean_score"],
            "trace/mean_hacking_score": detection["mean_hacking_score"],
            "trace/mean_legitimate_score": detection["mean_legitimate_score"],
            "trace/threshold": detection["threshold"],
        }

        # Apply reward penalty if configured
        if self.config.use_for_reward_penalty and is_hacking.any():
            if "token_level_rewards" in data.batch.keys():
                data.batch["token_level_rewards"] = self.detector.apply_reward_penalty(
                    data.batch["token_level_rewards"], is_hacking
                )
            elif "token_level_scores" in data.batch.keys():
                data.batch["token_level_scores"] = self.detector.apply_reward_penalty(
                    data.batch["token_level_scores"], is_hacking
                )

        # Store hacking mask for potential filtering downstream
        data.batch["trace_is_hacking"] = is_hacking.float()

        # Run loophole discovery if configured
        if (
            self.loophole_discovery is not None
            and step % self.config.loophole_discovery_frequency == 0
        ):
            self._run_loophole_discovery(data, trace_scores)

        return data

    def _compute_scores_with_generation(
        self,
        data: DataProto,
        generate_fn: Callable,
    ) -> torch.Tensor:
        """Compute TRACE scores by generating completions at each truncation point.

        This is the full TRACE algorithm as described in the paper:
        1. For each truncation fraction:
            a. Truncate the CoT
            b. Append answer prompt
            c. Generate N completions
            d. Score each completion
            e. Average to get E[R̂]
        2. Compute AUC over all truncation fractions

        Args:
            data: The rollout batch.
            generate_fn: Generation function.

        Returns:
            TRACE scores, shape (batch_size,).
        """
        response_ids = data.batch["responses"]
        response_mask = data.batch["attention_mask"][:, data.batch["prompts"].shape[-1]:]
        prompt_ids = data.batch["prompts"]

        batch_size = response_ids.shape[0]
        num_fractions = len(self.config.truncation_fractions)
        device = response_ids.device

        all_expected_rewards = torch.zeros(batch_size, num_fractions, device=device)
        answer_prompt_ids = self.tokenizer.encode(
            self.config.answer_prompt, add_special_tokens=False
        )
        answer_prompt_tensor = torch.tensor(answer_prompt_ids, device=device)

        for frac_idx, fraction in enumerate(self.config.truncation_fractions):
            trunc_ids, trunc_mask, trunc_lengths = truncate_response_batch(
                response_ids, response_mask, fraction
            )

            # Collect rewards for all completions at this truncation fraction
            fraction_rewards = torch.zeros(
                batch_size, self.config.num_completions, device=device
            )

            for sample_idx in range(batch_size):
                trunc_len = trunc_lengths[sample_idx].item()
                sample_prompt = prompt_ids[sample_idx]

                # Build input: prompt + truncated response + answer prompt
                prompt_len = sample_prompt.shape[0]
                attn_mask = data.batch["attention_mask"][sample_idx]
                valid_prompt_start = (attn_mask[:prompt_len] == 0).sum().item()
                valid_prompt = sample_prompt[valid_prompt_start:]
                truncated_response = trunc_ids[sample_idx][:trunc_len]
                input_ids = torch.cat([
                    valid_prompt, truncated_response, answer_prompt_tensor
                ])
                input_mask = torch.ones(input_ids.shape[0], device=device)

                # Generate completions
                input_ids_batched = input_ids.unsqueeze(0).expand(
                    self.config.num_completions, -1
                )
                input_mask_batched = input_mask.unsqueeze(0).expand(
                    self.config.num_completions, -1
                )

                with torch.no_grad():
                    generated = generate_fn(
                        input_ids=input_ids_batched,
                        attention_mask=input_mask_batched,
                        num_return_sequences=1,
                        temperature=self.config.temperature,
                        max_new_tokens=self.config.max_completion_tokens,
                        do_sample=True,
                    )

                # Score completions
                for comp_idx in range(self.config.num_completions):
                    if generated is not None and comp_idx < generated.shape[0]:
                        full_output = generated[comp_idx]
                        completion = full_output[input_ids.shape[0]:]
                        completion_str = self.tokenizer.decode(
                            completion, skip_special_tokens=True
                        )
                        trunc_str = self.tokenizer.decode(
                            truncated_response, skip_special_tokens=True
                        )
                        full_response = (
                            trunc_str + self.config.answer_prompt + completion_str
                        )

                        gt = data.non_tensor_batch.get("reward_model", {})
                        if isinstance(gt, dict):
                            gt = gt.get("ground_truth", None)
                        elif hasattr(gt, "__getitem__"):
                            gt = gt[sample_idx].get("ground_truth", None) if sample_idx < len(gt) else None

                        ds = data.non_tensor_batch.get("data_source", "default")
                        if hasattr(ds, "__getitem__") and not isinstance(ds, str):
                            ds = ds[sample_idx] if sample_idx < len(ds) else "default"

                        try:
                            score = self.reward_fn(
                                data_source=ds,
                                solution_str=full_response,
                                ground_truth=gt,
                                extra_info={},
                            )
                            if isinstance(score, dict):
                                score = score.get("score", 0.0)
                            fraction_rewards[sample_idx, comp_idx] = float(score)
                        except Exception:
                            fraction_rewards[sample_idx, comp_idx] = 0.0

            all_expected_rewards[:, frac_idx] = compute_expected_reward_at_truncation(
                fraction_rewards
            )

        return compute_trace_auc(all_expected_rewards, self.config.truncation_fractions)

    def _compute_scores_simplified(
        self,
        data: DataProto,
    ) -> torch.Tensor:
        """Compute simplified TRACE scores without re-generation.

        When a generate_fn is not available, this method approximates TRACE
        scores by evaluating how the reward changes as we consider progressively
        more of the response. This is a lightweight proxy that works by:
        1. Computing the reward for the full response
        2. For each truncation fraction, computing the fraction of "reward mass"
           accumulated (based on the assumption that reward is assigned at the
           end of the response)
        3. Computing the AUC of reward accumulation vs. truncation fraction

        This simplified version doesn't require generation and is much faster,
        but provides a noisier signal than the full method.

        Args:
            data: The rollout batch.

        Returns:
            Approximate TRACE scores, shape (batch_size,).
        """
        response_mask = data.batch["attention_mask"][:, data.batch["prompts"].shape[-1]:]
        batch_size = response_mask.shape[0]
        num_fractions = len(self.config.truncation_fractions)
        device = response_mask.device

        # Use existing rewards if available
        if "token_level_rewards" in data.batch.keys():
            token_rewards = data.batch["token_level_rewards"]
        elif "token_level_scores" in data.batch.keys():
            token_rewards = data.batch["token_level_scores"]
        else:
            # No rewards available, return zero scores
            return torch.zeros(batch_size, device=device)

        # Total reward per sample
        total_rewards = (token_rewards * response_mask).sum(dim=-1)  # (batch_size,)

        # For each truncation fraction, compute cumulative reward
        all_expected_rewards = torch.zeros(batch_size, num_fractions, device=device)
        valid_lengths = response_mask.sum(dim=-1)  # (batch_size,)

        for frac_idx, fraction in enumerate(self.config.truncation_fractions):
            trunc_lengths = torch.clamp((valid_lengths * fraction).long(), min=1)
            positions = torch.arange(token_rewards.shape[-1], device=device).unsqueeze(0)
            trunc_mask = (positions < trunc_lengths.unsqueeze(-1)).float()
            cumulative_reward = (token_rewards * trunc_mask).sum(dim=-1)

            # Normalize by total reward to get fraction of reward accumulated
            # Handle zero total reward
            safe_total = total_rewards.clone()
            safe_total[safe_total == 0] = 1.0
            all_expected_rewards[:, frac_idx] = cumulative_reward / safe_total

        return compute_trace_auc(all_expected_rewards, self.config.truncation_fractions)

    def _run_loophole_discovery(
        self,
        data: DataProto,
        trace_scores: torch.Tensor,
    ) -> None:
        """Run loophole discovery clustering and analysis.

        Args:
            data: The batch data (for extracting metadata).
            trace_scores: TRACE scores for the batch.
        """
        if self.loophole_discovery is None:
            return

        # Cluster samples
        result = self.loophole_discovery.cluster_samples(trace_scores)

        # Analyze clusters with available metadata
        data_sources = None
        if "data_source" in data.non_tensor_batch:
            ds = data.non_tensor_batch["data_source"]
            if isinstance(ds, list):
                data_sources = ds
            elif hasattr(ds, "tolist"):
                data_sources = ds.tolist()

        prompt_ids = None
        if "uid" in data.non_tensor_batch:
            uid = data.non_tensor_batch["uid"]
            if isinstance(uid, list):
                prompt_ids = uid
            elif hasattr(uid, "tolist"):
                prompt_ids = uid.tolist()

        analysis = self.loophole_discovery.analyze_clusters(
            result,
            data_sources=data_sources,
            prompt_ids=prompt_ids,
        )

        # Update metrics
        discovery_metrics = self.loophole_discovery.get_metrics()
        self._current_metrics.update(discovery_metrics)

        if analysis.get("loophole_candidates"):
            logger.warning(
                f"TRACE loophole discovery found {len(analysis['loophole_candidates'])} "
                f"candidate loopholes at step {self.detector.step_count}"
            )

    def get_metrics(self) -> dict[str, float]:
        """Get current TRACE metrics for logging.

        Returns:
            Dictionary of metrics suitable for logging to wandb/tensorboard.
        """
        metrics = dict(self._current_metrics)
        metrics.update(self.detector.get_metrics())
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Serialize callback state for checkpointing.

        Returns:
            State dictionary for checkpointing.
        """
        return {
            "detector": self.detector.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore callback state from checkpoint.

        Args:
            state: State dictionary from state_dict().
        """
        if "detector" in state:
            self.detector.load_state_dict(state["detector"])
