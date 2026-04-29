# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the per-token version tracker.

Run with: ``pytest verl/experimental/pipeline_rl/tests/test_version_tracker.py``
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from verl.experimental.pipeline_rl.version_tracker import (
    TOKEN_POLICY_VERSION_KEY,
    TRAJECTORY_END_VERSION_KEY,
    TRAJECTORY_START_VERSION_KEY,
    GenerationVersionRecord,
    VersionClock,
    attach_versions_to_batch,
    is_in_flight_update,
    staleness_from_record,
)


def test_version_clock_monotonic():
    clock = VersionClock(initial=3)
    assert clock.current() == 3
    assert clock.bump() == 4
    assert clock.bump(to=10) == 10
    assert clock.current() == 10
    with pytest.raises(ValueError):
        clock.bump(to=5)


def test_to_dense_constant_when_no_swap():
    rec = GenerationVersionRecord(start_version=2, end_version=2)
    arr = rec.to_dense(5)
    np.testing.assert_array_equal(arr, np.full((5,), 2, dtype=np.int32))


def test_to_dense_interpolation_two_versions():
    rec = GenerationVersionRecord(start_version=1, end_version=2)
    arr = rec.to_dense(4)
    # Two equally-spaced buckets.
    assert arr.dtype == np.int32
    assert (arr == 1).sum() == 2
    assert (arr == 2).sum() == 2
    # Versions appear in order.
    assert arr[0] == 1 and arr[-1] == 2


def test_to_dense_interpolation_three_versions_short_response():
    # More versions than tokens -> consecutive assignment.
    rec = GenerationVersionRecord(start_version=5, end_version=8)
    arr = rec.to_dense(2)
    np.testing.assert_array_equal(arr, np.array([5, 6], dtype=np.int32))


def test_to_dense_explicit_per_token():
    explicit = np.array([7, 7, 8, 9], dtype=np.int32)
    rec = GenerationVersionRecord(start_version=7, end_version=9, per_token_versions=explicit)
    np.testing.assert_array_equal(rec.to_dense(4), explicit)
    # Truncation
    np.testing.assert_array_equal(rec.to_dense(2), np.array([7, 7], dtype=np.int32))
    # Pad with end_version
    np.testing.assert_array_equal(rec.to_dense(6), np.array([7, 7, 8, 9, 9, 9], dtype=np.int32))


def test_attach_versions_padding_marker():
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={})
    records = [
        GenerationVersionRecord(start_version=0, end_version=0),
        GenerationVersionRecord(start_version=1, end_version=2),
    ]
    response_lens = [3, 2]
    attach_versions_to_batch(batch, records, response_lens=response_lens, pad_to=4)

    versions = batch.batch[TOKEN_POLICY_VERSION_KEY]
    assert versions.shape == (2, 4)
    assert versions.dtype == torch.int32
    # Row 0: three valid tokens at version 0, one pad cell of -1.
    assert torch.equal(versions[0], torch.tensor([0, 0, 0, -1], dtype=torch.int32))
    # Row 1: two valid tokens (one at v1, one at v2), two pad cells.
    assert versions[1, 0].item() == 1
    assert versions[1, 1].item() == 2
    assert versions[1, 2].item() == -1 and versions[1, 3].item() == -1

    starts = batch.non_tensor_batch[TRAJECTORY_START_VERSION_KEY]
    ends = batch.non_tensor_batch[TRAJECTORY_END_VERSION_KEY]
    np.testing.assert_array_equal(starts, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(ends, np.array([0, 2], dtype=np.int32))


def test_staleness_and_inflight_predicate():
    rec = GenerationVersionRecord(start_version=2, end_version=4)
    assert is_in_flight_update(rec)
    assert staleness_from_record(rec, current_version=4) == 0
    assert staleness_from_record(rec, current_version=6) == 2
    # Negative staleness clamps to 0.
    assert staleness_from_record(rec, current_version=1) == 0


def test_attach_versions_length_mismatch_raises():
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={})
    with pytest.raises(ValueError):
        attach_versions_to_batch(
            batch,
            records=[GenerationVersionRecord(0, 0)],
            response_lens=[1, 2],  # mismatched
        )


def test_attach_versions_empty_records_is_noop():
    batch = types.SimpleNamespace(batch={}, non_tensor_batch={})
    attach_versions_to_batch(batch, records=[], response_lens=[])
    assert TOKEN_POLICY_VERSION_KEY not in batch.batch
    assert TRAJECTORY_START_VERSION_KEY not in batch.non_tensor_batch
