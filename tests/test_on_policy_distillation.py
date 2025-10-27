#!/usr/bin/env python3
"""Unit tests for on-policy distillation implementation."""

import torch
import pytest

from verl.trainer.ppo.on_policy_distillation import (
    compute_generalized_jsd_loss,
    compute_reverse_kl_loss,
    compute_forward_kl_loss,
    compute_on_policy_distillation_loss,
    compute_hybrid_rl_distillation_loss,
)


def test_reverse_kl_loss():
    """Test reverse KL divergence computation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    # Create random logits
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Compute loss
    loss = compute_reverse_kl_loss(student_logits, teacher_logits)

    # Check shape
    assert loss.shape == (batch_size, seq_len)

    # Check non-negative (KL divergence is always >= 0)
    assert torch.all(loss >= 0), "Reverse KL should be non-negative"

    # Test that identical distributions give zero loss
    loss_same = compute_reverse_kl_loss(student_logits, student_logits)
    assert torch.allclose(loss_same, torch.zeros_like(loss_same), atol=1e-5)


def test_forward_kl_loss():
    """Test forward KL divergence computation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Compute loss
    loss = compute_forward_kl_loss(student_logits, teacher_logits)

    # Check shape and non-negativity
    assert loss.shape == (batch_size, seq_len)
    assert torch.all(loss >= 0), "Forward KL should be non-negative"

    # Test identical distributions
    loss_same = compute_forward_kl_loss(student_logits, student_logits)
    assert torch.allclose(loss_same, torch.zeros_like(loss_same), atol=1e-5)


def test_generalized_jsd_loss():
    """Test generalized JSD loss computation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Test with beta=0 (should be similar to forward KL)
    loss_beta0 = compute_generalized_jsd_loss(student_logits, teacher_logits, beta=0.0)
    assert loss_beta0.shape == (batch_size, seq_len)

    # Test with beta=1 (should be similar to reverse KL)
    loss_beta1 = compute_generalized_jsd_loss(student_logits, teacher_logits, beta=1.0)
    assert loss_beta1.shape == (batch_size, seq_len)

    # Test with beta=0.5 (middle ground)
    loss_beta05 = compute_generalized_jsd_loss(student_logits, teacher_logits, beta=0.5)
    assert loss_beta05.shape == (batch_size, seq_len)


def test_temperature_scaling():
    """Test that temperature affects loss."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Compute with different temperatures
    loss_t1 = compute_reverse_kl_loss(student_logits, teacher_logits, temperature=1.0)
    loss_t2 = compute_reverse_kl_loss(student_logits, teacher_logits, temperature=2.0)

    # Losses should be different
    assert not torch.allclose(loss_t1, loss_t2)


def test_on_policy_distillation_loss():
    """Test the full on-policy distillation loss function."""
    batch_size = 4
    seq_len = 20
    vocab_size = 100

    # Create dummy inputs
    response_mask = torch.ones(batch_size, seq_len)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Dummy values (not used in distillation)
    old_log_prob = torch.randn(batch_size, seq_len)
    log_prob = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)

    # Test with reverse KL
    config = {
        "distillation_type": "reverse_kl",
        "distillation_temperature": 1.0,
    }

    loss, clipfrac, kl, clipfrac_lower = compute_on_policy_distillation_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        loss_agg_mode="token-mean",
        config=config,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
    )

    # Check outputs
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert clipfrac == 0.0, "Clipfrac should be 0 for distillation"
    assert kl.item() >= 0, "KL should be non-negative"


def test_distillation_types():
    """Test all three distillation types."""
    batch_size = 2
    seq_len = 10
    vocab_size = 50

    response_mask = torch.ones(batch_size, seq_len)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    dummy_args = [
        torch.randn(batch_size, seq_len),  # old_log_prob
        torch.randn(batch_size, seq_len),  # log_prob
        torch.randn(batch_size, seq_len),  # advantages
        response_mask,
        "token-mean",
    ]

    # Test reverse KL
    config_reverse = {"distillation_type": "reverse_kl"}
    loss_reverse, _, _, _ = compute_on_policy_distillation_loss(
        *dummy_args,
        config=config_reverse,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
    )
    assert loss_reverse.item() >= 0

    # Test forward KL
    config_forward = {"distillation_type": "forward_kl"}
    loss_forward, _, _, _ = compute_on_policy_distillation_loss(
        *dummy_args,
        config=config_forward,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
    )
    assert loss_forward.item() >= 0

    # Test GKD
    config_gkd = {"distillation_type": "gkd", "distillation_beta": 0.5}
    loss_gkd, _, _, _ = compute_on_policy_distillation_loss(
        *dummy_args,
        config=config_gkd,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
    )
    assert loss_gkd.item() >= 0


def test_missing_logits_error():
    """Test that missing teacher/student logits raises error."""
    batch_size = 2
    seq_len = 10

    response_mask = torch.ones(batch_size, seq_len)
    dummy_args = [
        torch.randn(batch_size, seq_len),
        torch.randn(batch_size, seq_len),
        torch.randn(batch_size, seq_len),
        response_mask,
        "token-mean",
    ]

    # Should raise ValueError without teacher_logits
    with pytest.raises(ValueError, match="teacher_logits"):
        compute_on_policy_distillation_loss(*dummy_args, config={})


def test_response_mask():
    """Test that response mask correctly masks loss."""
    batch_size = 2
    seq_len = 10
    vocab_size = 50

    # Create mask with some zeros
    response_mask = torch.ones(batch_size, seq_len)
    response_mask[:, 5:] = 0  # Mask out second half

    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    dummy_args = [
        torch.randn(batch_size, seq_len),
        torch.randn(batch_size, seq_len),
        torch.randn(batch_size, seq_len),
        response_mask,
        "token-mean",
    ]

    config = {"distillation_type": "reverse_kl"}
    loss, _, _, _ = compute_on_policy_distillation_loss(
        *dummy_args,
        config=config,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
    )

    # Loss should still be valid
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_hybrid_mode():
    """Test hybrid RL + distillation mode."""
    batch_size = 2
    seq_len = 10
    vocab_size = 50

    response_mask = torch.ones(batch_size, seq_len)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    old_log_prob = torch.randn(batch_size, seq_len)
    log_prob = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)

    config = {
        "distillation_type": "reverse_kl",
        "hybrid_lambda_rl": 0.5,
        "hybrid_lambda_distill": 0.5,
        "cliprange": 0.2,  # For RL component
    }

    loss, clipfrac, kl, clipfrac_lower = compute_hybrid_rl_distillation_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        loss_agg_mode="token-mean",
        config=config,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
    )

    # Check outputs
    assert loss.ndim == 0
    assert loss.item() >= 0
    # Clipfrac should be non-zero from RL component
    assert kl.item() >= 0


def test_loss_aggregation_modes():
    """Test different loss aggregation modes."""
    batch_size = 2
    seq_len = 10
    vocab_size = 50

    response_mask = torch.ones(batch_size, seq_len)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    dummy_args = [
        torch.randn(batch_size, seq_len),
        torch.randn(batch_size, seq_len),
        torch.randn(batch_size, seq_len),
        response_mask,
    ]

    config = {"distillation_type": "reverse_kl"}

    # Test different aggregation modes
    agg_modes = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"]

    losses = []
    for agg_mode in agg_modes:
        loss, _, _, _ = compute_on_policy_distillation_loss(
            *dummy_args,
            agg_mode,
            config=config,
            teacher_logits=teacher_logits,
            student_logits=student_logits,
        )
        losses.append(loss.item())
        assert loss.item() >= 0

    # Losses should be different for different aggregation modes
    assert len(set(losses)) == len(losses), "Different agg modes should give different losses"


if __name__ == "__main__":
    print("Running on-policy distillation tests...")

    test_reverse_kl_loss()
    print("✓ Reverse KL loss test passed")

    test_forward_kl_loss()
    print("✓ Forward KL loss test passed")

    test_generalized_jsd_loss()
    print("✓ Generalized JSD loss test passed")

    test_temperature_scaling()
    print("✓ Temperature scaling test passed")

    test_on_policy_distillation_loss()
    print("✓ On-policy distillation loss test passed")

    test_distillation_types()
    print("✓ Distillation types test passed")

    test_response_mask()
    print("✓ Response mask test passed")

    test_hybrid_mode()
    print("✓ Hybrid mode test passed")

    test_loss_aggregation_modes()
    print("✓ Loss aggregation modes test passed")

    print("\nAll tests passed! ✓")
