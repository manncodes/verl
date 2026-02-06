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
Configuration for TRACE (Truncated Reasoning AUC Evaluation).

Reference:
    Wang et al. (2025). "Is It Thinking or Cheating? Detecting Implicit
    Reward Hacking by Measuring Reasoning Effort." arXiv:2510.01367
"""

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["TRACEConfig"]


@dataclass
class TRACEConfig(BaseConfig):
    """Configuration for the TRACE reward hacking detector.

    TRACE works by progressively truncating a model's chain-of-thought (CoT) at
    various percentages, forcing the model to produce an answer from the truncated
    reasoning, and measuring the expected reward at each truncation level. The area
    under the reward-vs-truncation curve (AUC) quantifies how much "reasoning effort"
    the model is actually using. A high AUC indicates the model can achieve high reward
    with little reasoning -- a hallmark of reward hacking.

    Args:
        enable: Whether to enable TRACE detection during training.
        truncation_fractions: List of fractions at which to truncate the CoT.
            E.g., [0.1, 0.2, ..., 1.0] means truncate at 10%, 20%, ... 100%.
        num_completions: Number of completions to sample at each truncation point
            to estimate E[R̂]. More completions give a better estimate but cost
            more compute.
        temperature: Sampling temperature for generating completions from truncated
            prefixes.
        max_completion_tokens: Maximum number of new tokens to generate for the
            completion after truncation (answer extraction).
        detection_frequency: How often to run TRACE detection, in training steps.
            E.g., 10 means run every 10 steps.
        baseline_steps: Number of initial steps to use for computing the baseline
            TRACE score distribution (from the initial policy before RL training).
        threshold_method: How to set the hacking detection threshold.
            "baseline_mean_std": mean + n_sigma * std of baseline scores.
            "baseline_percentile": A fixed percentile of the baseline distribution.
            "fixed": Use a fixed threshold value.
        threshold_n_sigma: Number of standard deviations above baseline mean for
            the "baseline_mean_std" method.
        threshold_percentile: Percentile for the "baseline_percentile" method.
        fixed_threshold: Fixed threshold value for the "fixed" method.
        use_for_filtering: If True, filter out samples detected as hacking from
            the training batch before policy updates.
        use_for_reward_penalty: If True, apply a reward penalty to samples detected
            as hacking.
        reward_penalty_coef: Coefficient for the reward penalty.
        enable_loophole_discovery: Whether to enable unsupervised loophole discovery
            via clustering on TRACE scores.
        loophole_discovery_frequency: How often to run loophole discovery, in steps.
        n_clusters: Number of clusters for K-means clustering in loophole discovery.
        answer_prompt: The prompt appended after truncated CoT to force the model
            to produce an answer. E.g., "Therefore, the answer is".
        log_detailed_scores: Whether to log per-sample TRACE scores for analysis.
    """

    enable: bool = False
    truncation_fractions: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    num_completions: int = 4
    temperature: float = 0.7
    max_completion_tokens: int = 128
    detection_frequency: int = 10
    baseline_steps: int = 5
    threshold_method: str = "baseline_mean_std"
    threshold_n_sigma: float = 2.0
    threshold_percentile: float = 95.0
    fixed_threshold: float = 0.7
    use_for_filtering: bool = False
    use_for_reward_penalty: bool = False
    reward_penalty_coef: float = 1.0
    enable_loophole_discovery: bool = False
    loophole_discovery_frequency: int = 50
    n_clusters: int = 2
    answer_prompt: str = "\n\nTherefore, the answer is"
    log_detailed_scores: bool = False
