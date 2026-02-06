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
"""Tests for the TRACE module."""

import numpy as np
import pytest
import torch

from verl.trainer.trace.config import TRACEConfig
from verl.trainer.trace.core import (
    compute_expected_reward_at_truncation,
    compute_trace_auc,
    truncate_response,
    truncate_response_batch,
)
from verl.trainer.trace.detector import TRACEDetector
from verl.trainer.trace.loophole_discovery import TRACELoopholeDiscovery


class TestTruncateResponse:
    """Tests for response truncation functions."""

    def test_truncate_single_response(self):
        response_ids = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0])
        response_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0])

        trunc_ids, trunc_mask, trunc_len = truncate_response(
            response_ids, response_mask, fraction=0.5
        )

        # 50% of 5 valid tokens = 2 (rounded down, min 1)
        assert trunc_len == 2
        assert trunc_mask[:2].sum() == 2
        assert trunc_mask[2:].sum() == 0
        assert trunc_ids[2:].sum() == 0

    def test_truncate_at_full_length(self):
        response_ids = torch.tensor([1, 2, 3, 4, 5])
        response_mask = torch.tensor([1, 1, 1, 1, 1])

        trunc_ids, trunc_mask, trunc_len = truncate_response(
            response_ids, response_mask, fraction=1.0
        )

        assert trunc_len == 5
        assert torch.equal(trunc_ids, response_ids)
        assert torch.equal(trunc_mask, response_mask)

    def test_truncate_at_minimum(self):
        response_ids = torch.tensor([1, 2, 3, 4, 5])
        response_mask = torch.tensor([1, 1, 1, 1, 1])

        trunc_ids, trunc_mask, trunc_len = truncate_response(
            response_ids, response_mask, fraction=0.01
        )

        # Should keep at least 1 token
        assert trunc_len >= 1
        assert trunc_mask.sum() >= 1

    def test_truncate_batch(self):
        response_ids = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0],
            [10, 20, 30, 0, 0, 0, 0],
        ])
        response_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ])

        trunc_ids, trunc_mask, trunc_lengths = truncate_response_batch(
            response_ids, response_mask, fraction=0.5
        )

        # First sample: 50% of 5 = 2 tokens
        assert trunc_lengths[0].item() == 2
        # Second sample: 50% of 3 = 1 token
        assert trunc_lengths[1].item() == 1


class TestComputeExpectedReward:
    """Tests for expected reward computation."""

    def test_single_sample(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        expected = compute_expected_reward_at_truncation(rewards)
        assert expected.item() == pytest.approx(0.5)

    def test_batch(self):
        rewards = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        expected = compute_expected_reward_at_truncation(rewards)
        assert expected[0].item() == pytest.approx(1.0)
        assert expected[1].item() == pytest.approx(0.0)


class TestComputeTraceAUC:
    """Tests for TRACE AUC computation."""

    def test_constant_reward(self):
        """A model that always gets reward 1.0 should have AUC of 1.0."""
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        expected_rewards = torch.ones(len(fractions))
        auc = compute_trace_auc(expected_rewards, fractions)
        assert auc.item() == pytest.approx(1.0, abs=0.01)

    def test_zero_reward(self):
        """A model that always gets reward 0.0 should have AUC of 0.0."""
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        expected_rewards = torch.zeros(len(fractions))
        auc = compute_trace_auc(expected_rewards, fractions)
        assert auc.item() == pytest.approx(0.0, abs=0.01)

    def test_hacking_pattern(self):
        """A hacking model gets high reward even at early truncation."""
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # High reward from the start
        expected_rewards = torch.tensor([0.9, 0.95, 0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        auc_hacking = compute_trace_auc(expected_rewards, fractions)

        # Legitimate model: reward increases gradually
        expected_rewards_legit = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
        auc_legit = compute_trace_auc(expected_rewards_legit, fractions)

        # Hacking AUC should be higher
        assert auc_hacking > auc_legit

    def test_batch_auc(self):
        """Test AUC computation for a batch."""
        fractions = [0.1, 0.5, 1.0]
        expected_rewards = torch.tensor([
            [1.0, 1.0, 1.0],  # Always high (hacking)
            [0.0, 0.5, 1.0],  # Gradual (legitimate)
        ])
        auc = compute_trace_auc(expected_rewards, fractions)
        assert auc.shape == (2,)
        assert auc[0] > auc[1]


class TestTRACEDetector:
    """Tests for the TRACE detector."""

    def test_baseline_collection(self):
        config = TRACEConfig(
            enable=True,
            baseline_steps=3,
            threshold_method="baseline_mean_std",
            threshold_n_sigma=2.0,
        )
        detector = TRACEDetector(config)

        # Collect baseline scores
        for _ in range(3):
            scores = torch.randn(10) * 0.1 + 0.3  # ~N(0.3, 0.1)
            detector.update_baseline(scores)

        assert len(detector.baseline_scores) == 30
        assert not detector.is_calibrated

    def test_calibration_mean_std(self):
        config = TRACEConfig(
            enable=True,
            threshold_method="baseline_mean_std",
            threshold_n_sigma=2.0,
        )
        detector = TRACEDetector(config)

        # Add baseline scores with known distribution
        scores = torch.tensor([0.2, 0.3, 0.25, 0.35, 0.28, 0.32, 0.27, 0.33])
        detector.update_baseline(scores)
        threshold = detector.calibrate()

        assert detector.is_calibrated
        expected_mean = scores.mean().item()
        expected_std = scores.std().item()
        expected_threshold = expected_mean + 2.0 * expected_std
        assert threshold == pytest.approx(expected_threshold, abs=0.01)

    def test_calibration_percentile(self):
        config = TRACEConfig(
            enable=True,
            threshold_method="baseline_percentile",
            threshold_percentile=90.0,
        )
        detector = TRACEDetector(config)

        scores = torch.arange(100).float() / 100.0
        detector.update_baseline(scores)
        threshold = detector.calibrate()

        assert detector.is_calibrated
        assert threshold == pytest.approx(0.9, abs=0.02)

    def test_calibration_fixed(self):
        config = TRACEConfig(
            enable=True,
            threshold_method="fixed",
            fixed_threshold=0.75,
        )
        detector = TRACEDetector(config)

        scores = torch.randn(10)
        detector.update_baseline(scores)
        threshold = detector.calibrate()

        assert threshold == 0.75

    def test_detection(self):
        config = TRACEConfig(
            enable=True,
            threshold_method="fixed",
            fixed_threshold=0.5,
        )
        detector = TRACEDetector(config)

        scores = torch.randn(10)
        detector.update_baseline(scores)
        detector.calibrate()

        # Test detection
        test_scores = torch.tensor([0.2, 0.8, 0.3, 0.9, 0.4])
        result = detector.detect(test_scores)

        assert result["is_hacking"].shape == (5,)
        # Scores 0.8 and 0.9 should be detected as hacking
        assert result["is_hacking"][1].item() is True
        assert result["is_hacking"][3].item() is True
        assert result["is_hacking"][0].item() is False
        assert result["hacking_fraction"] == pytest.approx(0.4)

    def test_reward_penalty(self):
        config = TRACEConfig(
            enable=True,
            threshold_method="fixed",
            fixed_threshold=0.5,
            reward_penalty_coef=1.0,
        )
        detector = TRACEDetector(config)

        scores = torch.randn(10)
        detector.update_baseline(scores)
        detector.calibrate()

        reward_tensor = torch.ones(3, 10)  # (batch_size=3, response_length=10)
        is_hacking = torch.tensor([False, True, False])

        penalized = detector.apply_reward_penalty(reward_tensor, is_hacking)

        # Non-hacking samples should be unchanged
        assert torch.equal(penalized[0], reward_tensor[0])
        assert torch.equal(penalized[2], reward_tensor[2])
        # Hacking sample should be penalized (zeroed with coef=1.0)
        assert penalized[1].sum().item() == pytest.approx(0.0)

    def test_filter_batch(self):
        config = TRACEConfig(enable=True)
        detector = TRACEDetector(config)

        is_hacking = torch.tensor([False, True, False, True, False])
        indices = detector.filter_batch(is_hacking)

        assert torch.equal(indices, torch.tensor([0, 2, 4]))

    def test_state_dict_roundtrip(self):
        config = TRACEConfig(
            enable=True,
            threshold_method="fixed",
            fixed_threshold=0.6,
        )
        detector = TRACEDetector(config)

        scores = torch.randn(10)
        detector.update_baseline(scores)
        detector.calibrate()

        state = detector.state_dict()

        # Create new detector and load state
        detector2 = TRACEDetector(config)
        detector2.load_state_dict(state)

        assert detector2.is_calibrated
        assert detector2.threshold == detector.threshold
        assert len(detector2.baseline_scores) == len(detector.baseline_scores)


class TestTRACELoopholeDiscovery:
    """Tests for loophole discovery."""

    def test_cluster_two_groups(self):
        config = TRACEConfig(n_clusters=2)
        discovery = TRACELoopholeDiscovery(config)

        # Create two clearly separated groups
        group1 = torch.randn(50) * 0.1 + 0.2  # ~N(0.2, 0.1) -- legitimate
        group2 = torch.randn(50) * 0.1 + 0.8  # ~N(0.8, 0.1) -- hacking
        scores = torch.cat([group1, group2])

        result = discovery.cluster_samples(scores)

        assert "labels" in result
        assert "centroids" in result
        assert "hacking_mask" in result
        assert "separation" in result
        assert result["separation"] > 0.3  # Should be well-separated

    def test_analyze_clusters_with_sources(self):
        config = TRACEConfig(n_clusters=2)
        discovery = TRACELoopholeDiscovery(config)

        # Create scores where one data source is mostly hacking
        scores = torch.tensor([0.2, 0.3, 0.8, 0.9, 0.25, 0.85])
        result = discovery.cluster_samples(scores)

        data_sources = ["math", "math", "code", "code", "math", "code"]
        analysis = discovery.analyze_clusters(result, data_sources=data_sources)

        assert "hacking_by_source" in analysis
        assert "source_vulnerability" in analysis
        assert "total_samples" in analysis
        assert analysis["total_samples"] == 6

    def test_single_cluster_fallback(self):
        config = TRACEConfig(n_clusters=3)
        discovery = TRACELoopholeDiscovery(config)

        # Only 2 samples, less than n_clusters=3
        scores = torch.tensor([0.5, 0.6])
        result = discovery.cluster_samples(scores)

        assert "labels" in result

    def test_metrics(self):
        config = TRACEConfig(n_clusters=2)
        discovery = TRACELoopholeDiscovery(config)

        scores = torch.randn(100) * 0.3 + 0.5
        discovery.cluster_samples(scores)

        metrics = discovery.get_metrics()
        assert "trace_loophole/separation" in metrics
        assert "trace_loophole/hacking_fraction" in metrics


class TestTRACEConfig:
    """Tests for TRACE configuration."""

    def test_default_config(self):
        config = TRACEConfig()
        assert config.enable is False
        assert len(config.truncation_fractions) == 10
        assert config.num_completions == 4
        assert config.temperature == 0.7

    def test_custom_config(self):
        config = TRACEConfig(
            enable=True,
            truncation_fractions=[0.25, 0.5, 0.75, 1.0],
            num_completions=8,
            threshold_method="fixed",
            fixed_threshold=0.8,
        )
        assert config.enable is True
        assert len(config.truncation_fractions) == 4
        assert config.num_completions == 8
        assert config.fixed_threshold == 0.8
