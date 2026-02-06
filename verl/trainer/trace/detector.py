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
TRACE Detector: Detects implicit reward hacking via truncated reasoning AUC.

The detector maintains a baseline distribution of TRACE scores from the initial
policy (before RL training) and classifies samples as hacking if their TRACE
score significantly exceeds this baseline.

Reference:
    Wang et al. (2025). "Is It Thinking or Cheating? Detecting Implicit
    Reward Hacking by Measuring Reasoning Effort." arXiv:2510.01367
"""

import logging
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.trace.config import TRACEConfig
from verl.trainer.trace.core import compute_trace_auc

logger = logging.getLogger(__name__)

__all__ = ["TRACEDetector"]


class TRACEDetector:
    """Detects implicit reward hacking using TRACE scores.

    The detector operates in two phases:
    1. Calibration: Collects TRACE scores from the initial (pre-RL) policy to
       establish a baseline distribution of "normal" reasoning effort.
    2. Detection: Classifies samples as hacking if their TRACE score exceeds
       a threshold derived from the baseline distribution.

    The key insight is that a model exploiting a reward loophole will achieve
    high reward with minimal reasoning, producing a high TRACE AUC. A
    legitimately reasoning model needs most of its CoT to reach the correct
    answer, producing a lower TRACE AUC.

    Attributes:
        config: TRACE configuration.
        baseline_scores: Collected baseline TRACE scores.
        threshold: Computed detection threshold.
        is_calibrated: Whether the baseline has been established.
        step_count: Number of detection steps performed.
    """

    def __init__(self, config: TRACEConfig):
        self.config = config
        self.baseline_scores: list[float] = []
        self.threshold: Optional[float] = None
        self.is_calibrated: bool = False
        self.step_count: int = 0
        self._score_history: list[dict[str, Any]] = []

    def update_baseline(self, trace_scores: torch.Tensor) -> None:
        """Add TRACE scores to the baseline distribution.

        Should be called during the initial training steps (before the model
        has had a chance to learn reward hacking).

        Args:
            trace_scores: TRACE scores from the initial policy, shape (batch_size,).
        """
        scores = trace_scores.detach().cpu().numpy().tolist()
        self.baseline_scores.extend(scores)
        logger.info(
            f"TRACE baseline updated: {len(self.baseline_scores)} samples collected"
        )

    def calibrate(self) -> float:
        """Compute the detection threshold from the baseline distribution.

        Must be called after enough baseline scores have been collected
        (typically after `config.baseline_steps` training steps).

        Returns:
            The computed threshold value.

        Raises:
            ValueError: If no baseline scores have been collected.
        """
        if len(self.baseline_scores) == 0:
            raise ValueError(
                "Cannot calibrate: no baseline scores collected. "
                "Call update_baseline() first during initial training steps."
            )

        scores = np.array(self.baseline_scores)
        method = self.config.threshold_method

        if method == "baseline_mean_std":
            mean = scores.mean()
            std = scores.std()
            self.threshold = mean + self.config.threshold_n_sigma * std
            logger.info(
                f"TRACE calibrated (mean+{self.config.threshold_n_sigma}*std): "
                f"mean={mean:.4f}, std={std:.4f}, threshold={self.threshold:.4f}"
            )
        elif method == "baseline_percentile":
            self.threshold = float(np.percentile(scores, self.config.threshold_percentile))
            logger.info(
                f"TRACE calibrated (p{self.config.threshold_percentile}): "
                f"threshold={self.threshold:.4f}"
            )
        elif method == "fixed":
            self.threshold = self.config.fixed_threshold
            logger.info(f"TRACE calibrated (fixed): threshold={self.threshold:.4f}")
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        self.is_calibrated = True
        return self.threshold

    def detect(
        self,
        trace_scores: torch.Tensor,
    ) -> dict[str, Any]:
        """Classify samples as hacking or legitimate based on TRACE scores.

        A sample is classified as hacking if its TRACE score exceeds the
        calibrated threshold. The threshold represents the upper bound of
        "normal" reasoning effort from the pre-RL baseline model.

        Args:
            trace_scores: TRACE scores for the batch, shape (batch_size,).

        Returns:
            Dictionary with:
                "is_hacking": Boolean mask, True for hacking samples, shape (batch_size,).
                "trace_scores": The input TRACE scores.
                "threshold": The detection threshold used.
                "hacking_fraction": Fraction of samples detected as hacking.
                "mean_score": Mean TRACE score in the batch.
                "mean_hacking_score": Mean TRACE score among hacking samples.
                "mean_legitimate_score": Mean TRACE score among legitimate samples.
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "TRACE detector not calibrated. Call calibrate() after collecting "
                "baseline scores."
            )

        self.step_count += 1
        scores_np = trace_scores.detach().cpu().numpy()

        is_hacking = trace_scores > self.threshold
        hacking_fraction = is_hacking.float().mean().item()

        result = {
            "is_hacking": is_hacking,
            "trace_scores": trace_scores,
            "threshold": self.threshold,
            "hacking_fraction": hacking_fraction,
            "mean_score": float(scores_np.mean()),
        }

        hacking_mask = is_hacking.cpu().numpy()
        if hacking_mask.any():
            result["mean_hacking_score"] = float(scores_np[hacking_mask].mean())
        else:
            result["mean_hacking_score"] = 0.0

        legitimate_mask = ~hacking_mask
        if legitimate_mask.any():
            result["mean_legitimate_score"] = float(scores_np[legitimate_mask].mean())
        else:
            result["mean_legitimate_score"] = 0.0

        if self.config.log_detailed_scores:
            self._score_history.append(
                {
                    "step": self.step_count,
                    "scores": scores_np.tolist(),
                    "hacking_fraction": hacking_fraction,
                }
            )

        return result

    def apply_reward_penalty(
        self,
        reward_tensor: torch.Tensor,
        is_hacking: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a reward penalty to samples detected as hacking.

        Reduces the reward for hacking samples to discourage the model from
        exploiting reward loopholes.

        Args:
            reward_tensor: Original reward tensor, shape (batch_size, response_length).
            is_hacking: Boolean mask from detect(), shape (batch_size,).

        Returns:
            Modified reward tensor with penalties applied.
        """
        penalty = self.config.reward_penalty_coef
        # Expand is_hacking to match reward tensor shape
        hacking_mask = is_hacking.unsqueeze(-1).float()
        # Reduce reward for hacking samples
        penalized = reward_tensor * (1.0 - hacking_mask * penalty)
        return penalized

    def filter_batch(
        self,
        is_hacking: torch.Tensor,
    ) -> torch.Tensor:
        """Return indices of legitimate (non-hacking) samples for batch filtering.

        Args:
            is_hacking: Boolean mask from detect(), shape (batch_size,).

        Returns:
            Indices of legitimate samples, shape (num_legitimate,).
        """
        return torch.where(~is_hacking)[0]

    def get_metrics(self) -> dict[str, float]:
        """Get current TRACE metrics for logging.

        Returns:
            Dictionary of metrics suitable for logging to wandb/tensorboard.
        """
        metrics = {
            "trace/is_calibrated": float(self.is_calibrated),
            "trace/num_baseline_samples": float(len(self.baseline_scores)),
        }
        if self.is_calibrated:
            metrics["trace/threshold"] = self.threshold
        if self.baseline_scores:
            scores = np.array(self.baseline_scores)
            metrics["trace/baseline_mean"] = float(scores.mean())
            metrics["trace/baseline_std"] = float(scores.std())
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Serialize the detector state for checkpointing.

        Returns:
            Dictionary containing all state needed to restore the detector.
        """
        return {
            "baseline_scores": self.baseline_scores,
            "threshold": self.threshold,
            "is_calibrated": self.is_calibrated,
            "step_count": self.step_count,
            "config": {
                "threshold_method": self.config.threshold_method,
                "threshold_n_sigma": self.config.threshold_n_sigma,
                "threshold_percentile": self.config.threshold_percentile,
                "fixed_threshold": self.config.fixed_threshold,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore detector state from a checkpoint.

        Args:
            state: State dictionary from state_dict().
        """
        self.baseline_scores = state["baseline_scores"]
        self.threshold = state["threshold"]
        self.is_calibrated = state["is_calibrated"]
        self.step_count = state["step_count"]
