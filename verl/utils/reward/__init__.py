"""Reward utilities for RLVR/GRPO training."""

from verl.utils.reward.repetition_penalty import (
    apply_repetition_penalty,
    apply_repetition_penalty_batch,
    get_repetition_scores_batch,
    is_repetitive,
    repetition_score,
)

__all__ = [
    "repetition_score",
    "is_repetitive",
    "apply_repetition_penalty",
    "apply_repetition_penalty_batch",
    "get_repetition_scores_batch",
]
