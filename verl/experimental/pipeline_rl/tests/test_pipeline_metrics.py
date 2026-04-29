# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for version-aware off-policy metrics.

Run with: ``pytest verl/experimental/pipeline_rl/tests/test_pipeline_metrics.py``
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from verl.experimental.pipeline_rl.pipeline_metrics import (
    compute_pipeline_metrics_from_batch,
    compute_pipeline_version_metrics,
)
from verl.experimental.pipeline_rl.version_tracker import (
    TOKEN_POLICY_VERSION_KEY,
    TRAJECTORY_END_VERSION_KEY,
    TRAJECTORY_START_VERSION_KEY,
    GenerationVersionRecord,
    attach_versions_to_batch,
)


def _mk_batch(token_versions, response_mask, *, log_ratio=None, starts=None, ends=None):
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={})
    batch.batch[TOKEN_POLICY_VERSION_KEY] = torch.as_tensor(token_versions, dtype=torch.int32)
    batch.batch["response_mask"] = torch.as_tensor(response_mask)
    if log_ratio is not None:
        # Place rollout/old logprobs such that the difference equals log_ratio.
        lr = torch.as_tensor(log_ratio, dtype=torch.float32)
        batch.batch["old_log_probs"] = lr
        batch.batch["rollout_log_probs"] = torch.zeros_like(lr)
    if starts is not None:
        batch.non_tensor_batch[TRAJECTORY_START_VERSION_KEY] = np.asarray(starts, dtype=np.int32)
    if ends is not None:
        batch.non_tensor_batch[TRAJECTORY_END_VERSION_KEY] = np.asarray(ends, dtype=np.int32)
    return batch


def test_metrics_no_versions_returns_empty():
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={})
    out = compute_pipeline_metrics_from_batch(batch, current_version=5)
    assert out == {}


def test_metrics_basic_staleness():
    versions = [[3, 3, 3, -1], [4, 4, -1, -1]]
    mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
    out = compute_pipeline_version_metrics(
        token_policy_version=torch.as_tensor(versions, dtype=torch.int32),
        response_mask=torch.as_tensor(mask),
        current_version=5,
    )
    assert out["pipeline_rl/valid_tokens"] == 5.0
    assert out["pipeline_rl/token_version_min"] == 3.0
    assert out["pipeline_rl/token_version_max"] == 4.0
    # 3 tokens at staleness 2 + 2 tokens at staleness 1 -> mean = (3*2+2*1)/5 = 1.6
    assert out["pipeline_rl/token_staleness_mean"] == pytest.approx(1.6, abs=1e-6)
    assert out["pipeline_rl/token_staleness_max"] == 2.0
    assert out["pipeline_rl/token_off_policy_fraction"] == 1.0  # all tokens are stale


def test_metrics_staleness_buckets_with_log_ratio():
    # Two rows: one fully fresh (current), one one-step stale.
    versions = [[5, 5], [4, 4]]
    mask = [[1, 1], [1, 1]]
    log_ratio = [[0.1, -0.2], [1.0, -1.0]]
    batch = _mk_batch(versions, mask, log_ratio=log_ratio)
    out = compute_pipeline_metrics_from_batch(batch, current_version=5)
    # Bucket 0 (fresh): 2 tokens, log_ratio 0.1, -0.2 -> mean = -0.05
    assert out["pipeline_rl/staleness_0/count"] == 2.0
    assert out["pipeline_rl/staleness_0/log_ratio_mean"] == pytest.approx(-0.05, abs=1e-6)
    # Bucket 1: 2 tokens, log_ratio 1.0, -1.0 -> abs mean = 1.0
    assert out["pipeline_rl/staleness_1/count"] == 2.0
    assert out["pipeline_rl/staleness_1/log_ratio_abs_mean"] == pytest.approx(1.0, abs=1e-6)
    # Aggregate stale bucket
    assert out["pipeline_rl/staleness_ge1/count"] == 2.0


def test_metrics_inflight_trajectory_fraction():
    # 3 trajectories, two were updated mid-flight.
    starts = [0, 1, 1]
    ends = [0, 2, 3]
    versions = [[0, -1], [1, 2], [1, 3]]
    mask = [[1, 0], [1, 1], [1, 1]]
    batch = _mk_batch(versions, mask, starts=starts, ends=ends)
    out = compute_pipeline_metrics_from_batch(batch, current_version=3)
    assert out["pipeline_rl/trajectory_inflight_update_fraction"] == pytest.approx(2 / 3, abs=1e-6)
    assert out["pipeline_rl/trajectory_version_span_max"] == 2.0


def test_metrics_round_trip_with_attach():
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={})
    records = [
        GenerationVersionRecord(start_version=0, end_version=1),
        GenerationVersionRecord(start_version=1, end_version=1),
    ]
    response_lens = [4, 2]
    attach_versions_to_batch(batch, records, response_lens=response_lens)

    # Add a response_mask consistent with the lengths.
    mask = torch.zeros((2, 4), dtype=torch.int32)
    mask[0, :4] = 1
    mask[1, :2] = 1
    batch.batch["response_mask"] = mask

    out = compute_pipeline_metrics_from_batch(batch, current_version=2)
    assert out["pipeline_rl/valid_tokens"] == 6.0
    # Trajectory 0 spans v0..v1 (in-flight), trajectory 1 stays at v1.
    assert out["pipeline_rl/trajectory_inflight_update_fraction"] == pytest.approx(0.5, abs=1e-6)


def test_metrics_shape_mismatch_raises():
    versions = torch.zeros((2, 3), dtype=torch.int32)
    mask = torch.ones((2, 4))
    with pytest.raises(ValueError):
        compute_pipeline_version_metrics(
            token_policy_version=versions,
            response_mask=mask,
            current_version=0,
        )


def test_metrics_no_valid_tokens_returns_zero_count():
    versions = torch.full((1, 3), -1, dtype=torch.int32)
    mask = torch.zeros((1, 3))
    out = compute_pipeline_version_metrics(
        token_policy_version=versions,
        response_mask=mask,
        current_version=0,
    )
    assert out["pipeline_rl/valid_tokens"] == 0.0
    # No further metrics are emitted when there's nothing valid.
    assert "pipeline_rl/token_staleness_mean" not in out
