"""Standalone TRACE tests that don't import the full verl package."""

import sys
import os

# Add verl root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import torch


# ---- Test core functions directly (without verl imports) ----

def truncate_response(response_ids, response_mask, fraction):
    valid_length = int(response_mask.sum().item())
    truncation_length = max(1, int(valid_length * fraction))
    truncated_ids = response_ids.clone()
    truncated_mask = response_mask.clone()
    truncated_ids[truncation_length:] = 0
    truncated_mask[truncation_length:] = 0
    return truncated_ids, truncated_mask, truncation_length


def truncate_response_batch(response_ids, response_mask, fraction):
    valid_lengths = response_mask.sum(dim=-1)
    truncation_lengths = torch.clamp((valid_lengths * fraction).long(), min=1)
    truncated_ids = response_ids.clone()
    truncated_mask = response_mask.clone()
    positions = torch.arange(response_ids.shape[-1], device=response_ids.device).unsqueeze(0)
    beyond_trunc = positions >= truncation_lengths.unsqueeze(-1)
    truncated_ids[beyond_trunc] = 0
    truncated_mask[beyond_trunc] = 0
    return truncated_ids, truncated_mask, truncation_lengths


def compute_expected_reward_at_truncation(rewards):
    return rewards.float().mean(dim=-1)


def compute_trace_auc(expected_rewards, truncation_fractions):
    fractions = torch.tensor(truncation_fractions, dtype=torch.float32, device=expected_rewards.device)
    if expected_rewards.dim() == 1:
        auc = torch.trapezoid(expected_rewards, fractions)
        x_range = fractions[-1] - fractions[0]
        if x_range > 0:
            auc = auc / x_range
        return auc
    else:
        auc = torch.trapezoid(expected_rewards, fractions, dim=-1)
        x_range = fractions[-1] - fractions[0]
        if x_range > 0:
            auc = auc / x_range
        return auc


# ---- Tests ----

def test_truncate_single_response():
    response_ids = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0])
    response_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0])
    trunc_ids, trunc_mask, trunc_len = truncate_response(response_ids, response_mask, 0.5)
    assert trunc_len == 2
    assert trunc_mask[:2].sum() == 2
    assert trunc_mask[2:].sum() == 0
    print("  PASS: test_truncate_single_response")


def test_truncate_at_full_length():
    response_ids = torch.tensor([1, 2, 3, 4, 5])
    response_mask = torch.tensor([1, 1, 1, 1, 1])
    trunc_ids, trunc_mask, trunc_len = truncate_response(response_ids, response_mask, 1.0)
    assert trunc_len == 5
    assert torch.equal(trunc_ids, response_ids)
    print("  PASS: test_truncate_at_full_length")


def test_truncate_batch():
    response_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [10, 20, 30, 0, 0, 0, 0]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    trunc_ids, trunc_mask, trunc_lengths = truncate_response_batch(response_ids, response_mask, 0.5)
    assert trunc_lengths[0].item() == 2
    assert trunc_lengths[1].item() == 1
    print("  PASS: test_truncate_batch")


def test_expected_reward_single():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expected = compute_expected_reward_at_truncation(rewards)
    assert abs(expected.item() - 0.5) < 1e-6
    print("  PASS: test_expected_reward_single")


def test_expected_reward_batch():
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    expected = compute_expected_reward_at_truncation(rewards)
    assert abs(expected[0].item() - 1.0) < 1e-6
    assert abs(expected[1].item() - 0.0) < 1e-6
    print("  PASS: test_expected_reward_batch")


def test_auc_constant_reward():
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    expected_rewards = torch.ones(len(fractions))
    auc = compute_trace_auc(expected_rewards, fractions)
    assert abs(auc.item() - 1.0) < 0.01
    print("  PASS: test_auc_constant_reward")


def test_auc_zero_reward():
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    expected_rewards = torch.zeros(len(fractions))
    auc = compute_trace_auc(expected_rewards, fractions)
    assert abs(auc.item()) < 0.01
    print("  PASS: test_auc_zero_reward")


def test_auc_hacking_vs_legitimate():
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Hacking: high reward from the start
    hacking_rewards = torch.tensor([0.9, 0.95, 0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    auc_hacking = compute_trace_auc(hacking_rewards, fractions)
    # Legitimate: reward increases gradually
    legit_rewards = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    auc_legit = compute_trace_auc(legit_rewards, fractions)
    assert auc_hacking > auc_legit, f"Hacking AUC ({auc_hacking:.3f}) should be > Legitimate AUC ({auc_legit:.3f})"
    print(f"  PASS: test_auc_hacking_vs_legitimate (hacking={auc_hacking:.3f}, legit={auc_legit:.3f})")


def test_auc_batch():
    fractions = [0.1, 0.5, 1.0]
    expected_rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.5, 1.0]])
    auc = compute_trace_auc(expected_rewards, fractions)
    assert auc.shape == (2,)
    assert auc[0] > auc[1]
    print("  PASS: test_auc_batch")


def test_kmeans_clustering():
    """Test K-means on TRACE scores with two clear clusters."""
    np.random.seed(42)
    group1 = np.random.normal(0.2, 0.05, 50)  # legitimate
    group2 = np.random.normal(0.8, 0.05, 50)  # hacking
    scores = np.concatenate([group1, group2]).reshape(-1, 1)

    # Simple k-means
    n_clusters = 2
    centroids = np.array([[0.0], [1.0]])
    labels = np.zeros(100, dtype=int)

    for _ in range(50):
        distances = np.abs(scores - centroids.T)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centroids[k] = scores[mask].mean(axis=0)

    # Check separation
    c0, c1 = sorted(centroids.flatten())
    separation = c1 - c0
    assert separation > 0.3, f"Clusters should be well-separated, got separation={separation:.3f}"
    print(f"  PASS: test_kmeans_clustering (separation={separation:.3f})")


def test_detector_baseline_and_calibration():
    """Test detector baseline collection and threshold calibration."""
    # Simulate baseline scores
    baseline_scores = torch.tensor([0.2, 0.3, 0.25, 0.35, 0.28, 0.32, 0.27, 0.33])
    mean = baseline_scores.mean().item()
    std = baseline_scores.std().item()
    threshold = mean + 2.0 * std

    # Simulate detection
    test_scores = torch.tensor([0.2, 0.8, 0.3, 0.9, 0.4])
    is_hacking = test_scores > threshold
    hacking_fraction = is_hacking.float().mean().item()

    assert is_hacking[1].item() is True, "Score 0.8 should be detected as hacking"
    assert is_hacking[3].item() is True, "Score 0.9 should be detected as hacking"
    assert is_hacking[0].item() is False, "Score 0.2 should not be hacking"
    print(f"  PASS: test_detector_baseline_and_calibration (threshold={threshold:.3f}, hacking_frac={hacking_fraction:.2f})")


def test_reward_penalty():
    """Test reward penalty application."""
    reward_tensor = torch.ones(3, 10)
    is_hacking = torch.tensor([False, True, False])
    penalty_coef = 1.0

    hacking_mask = is_hacking.unsqueeze(-1).float()
    penalized = reward_tensor * (1.0 - hacking_mask * penalty_coef)

    assert torch.equal(penalized[0], reward_tensor[0])
    assert torch.equal(penalized[2], reward_tensor[2])
    assert abs(penalized[1].sum().item()) < 1e-6
    print("  PASS: test_reward_penalty")


def test_filter_batch():
    """Test batch filtering."""
    is_hacking = torch.tensor([False, True, False, True, False])
    indices = torch.where(~is_hacking)[0]
    assert torch.equal(indices, torch.tensor([0, 2, 4]))
    print("  PASS: test_filter_batch")


if __name__ == "__main__":
    print("Running TRACE standalone tests...\n")

    print("Truncation tests:")
    test_truncate_single_response()
    test_truncate_at_full_length()
    test_truncate_batch()

    print("\nExpected reward tests:")
    test_expected_reward_single()
    test_expected_reward_batch()

    print("\nTRACE AUC tests:")
    test_auc_constant_reward()
    test_auc_zero_reward()
    test_auc_hacking_vs_legitimate()
    test_auc_batch()

    print("\nClustering tests:")
    test_kmeans_clustering()

    print("\nDetector tests:")
    test_detector_baseline_and_calibration()
    test_reward_penalty()
    test_filter_batch()

    print("\n" + "=" * 50)
    print("All TRACE standalone tests passed!")
    print("=" * 50)
