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
TRACE-aware Reward Manager.

Wraps an existing reward manager to incorporate TRACE-based reward hacking
detection. When enabled, it can:
1. Penalize rewards for samples detected as hacking
2. Filter out hacking samples from the batch
3. Log TRACE metrics alongside standard reward metrics

Reference:
    Wang et al. (2025). "Is It Thinking or Cheating? Detecting Implicit
    Reward Hacking by Measuring Reasoning Effort." arXiv:2510.01367
"""

import logging
from typing import Any, Optional

import torch

from verl import DataProto
from verl.trainer.trace.config import TRACEConfig
from verl.trainer.trace.detector import TRACEDetector
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)

__all__ = ["TRACERewardManager"]


@register("trace")
class TRACERewardManager(AbstractRewardManager):
    """Reward manager that wraps another reward manager with TRACE detection.

    This reward manager delegates actual reward computation to an inner reward
    manager, then applies TRACE-based reward hacking detection to optionally
    penalize or filter hacking samples.

    Usage:
        # In config, specify the reward manager as "trace"
        # and provide the inner reward manager config under trace.inner_reward_manager

    Args:
        tokenizer: The tokenizer used to decode token IDs.
        num_examine: Number of batches of decoded responses to print for debugging.
        compute_score: Reward scoring function.
        reward_fn_key: Key for accessing data source in non-tensor batch.
        trace_config: TRACE configuration.
        inner_reward_manager: The inner reward manager to delegate to.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        trace_config: Optional[TRACEConfig] = None,
        inner_reward_manager: Optional[AbstractRewardManager] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

        if trace_config is None:
            trace_config = TRACEConfig()
        self.trace_config = trace_config
        self.detector = TRACEDetector(trace_config)
        self.inner_reward_manager = inner_reward_manager

        # Track the training step for baseline collection and detection frequency
        self._step = 0

    def set_inner_reward_manager(self, manager: AbstractRewardManager) -> None:
        """Set the inner reward manager after construction.

        Args:
            manager: The reward manager to delegate reward computation to.
        """
        self.inner_reward_manager = manager

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Compute rewards with optional TRACE-based hacking detection.

        First delegates to the inner reward manager for actual reward computation,
        then applies TRACE detection if enabled and conditions are met.

        Args:
            data: DataProto containing the batch data.
            return_dict: If True, return a dict with reward tensor and extra info.

        Returns:
            Reward tensor or dict with reward tensor and TRACE metrics.
        """
        if self.inner_reward_manager is None:
            raise RuntimeError(
                "Inner reward manager not set. Call set_inner_reward_manager() "
                "or pass inner_reward_manager to constructor."
            )

        # Delegate to inner reward manager
        result = self.inner_reward_manager(data, return_dict=True)
        if isinstance(result, dict):
            reward_tensor = result["reward_tensor"]
            reward_extra_info = result.get("reward_extra_info", {})
        else:
            reward_tensor = result
            reward_extra_info = {}

        self._step += 1

        # Apply TRACE detection if enabled
        if self.trace_config.enable:
            trace_info = self._apply_trace(data, reward_tensor)
            reward_extra_info.update(trace_info.get("metrics", {}))

            if trace_info.get("penalized_rewards") is not None:
                reward_tensor = trace_info["penalized_rewards"]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _apply_trace(
        self,
        data: DataProto,
        reward_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        """Apply TRACE detection to the batch.

        During the baseline collection phase, TRACE scores are collected but no
        detection is performed. After calibration, samples are classified and
        optionally penalized.

        Args:
            data: The batch data.
            reward_tensor: The computed rewards.

        Returns:
            Dictionary with:
                "metrics": TRACE metrics for logging.
                "penalized_rewards": Modified rewards (if penalty is enabled).
                "is_hacking": Hacking mask (if detection is active).
        """
        result: dict[str, Any] = {"metrics": {}, "penalized_rewards": None}

        # We need pre-computed TRACE scores in the data
        # These are computed by the TRACE callback during the training loop
        if "trace_scores" not in data.batch.keys():
            return result

        trace_scores = data.batch["trace_scores"]

        # Phase 1: Baseline collection
        if self._step <= self.trace_config.baseline_steps:
            self.detector.update_baseline(trace_scores)
            if self._step == self.trace_config.baseline_steps:
                self.detector.calibrate()
            result["metrics"] = self.detector.get_metrics()
            return result

        # Phase 2: Detection (only at configured frequency)
        if self._step % self.trace_config.detection_frequency != 0:
            result["metrics"] = self.detector.get_metrics()
            return result

        detection = self.detector.detect(trace_scores)
        is_hacking = detection["is_hacking"]

        # Log metrics
        result["metrics"] = {
            "trace/hacking_fraction": detection["hacking_fraction"],
            "trace/mean_score": detection["mean_score"],
            "trace/mean_hacking_score": detection["mean_hacking_score"],
            "trace/mean_legitimate_score": detection["mean_legitimate_score"],
            "trace/threshold": detection["threshold"],
        }
        result["is_hacking"] = is_hacking

        # Apply reward penalty if configured
        if self.trace_config.use_for_reward_penalty and is_hacking.any():
            result["penalized_rewards"] = self.detector.apply_reward_penalty(
                reward_tensor, is_hacking
            )
            n_penalized = is_hacking.sum().item()
            logger.info(
                f"TRACE: penalized {n_penalized}/{len(is_hacking)} samples "
                f"(hacking fraction: {detection['hacking_fraction']:.2%})"
            )

        return result
