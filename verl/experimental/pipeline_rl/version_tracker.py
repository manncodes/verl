# Copyright 2026 The verl-project authors
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

"""Per-token policy-version tracking for PipelineRL.

In PipelineRL, weights are broadcast to rollout replicas *mid-generation*; consequently
different tokens of the same emitted sequence may have been sampled under different
policy versions. To compute correct importance-sampling weights and meaningful
off-policy diagnostics we need to know, for each token, which policy version
produced it.

This module implements a lightweight version clock owned by the rollout replica.
Every time the trainer pushes new weights, the rollout replica calls
:meth:`VersionClock.bump`. Tokens emitted between bump events are tagged with the
version that was current at the time of emission.

The tracker is intentionally conservative: when no streaming hook into the
inference engine is available, every token of a sequence is tagged with the
``starting`` version (the version that was current when the prefill started),
which matches the behavior of fully-async pause/resume. When the inference engine
exposes per-token timestamps the same primitive can be used to record exact
versions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# Key for the per-token policy-version tensor on a DataProto's ``batch``.
TOKEN_POLICY_VERSION_KEY = "token_policy_version"

# Key for the per-trajectory starting-version array on ``non_tensor_batch``.
TRAJECTORY_START_VERSION_KEY = "trajectory_start_version"

# Key for the per-trajectory ending-version array on ``non_tensor_batch``.
TRAJECTORY_END_VERSION_KEY = "trajectory_end_version"


class VersionClock:
    """Monotonic policy-version counter shared by a single rollout replica.

    Thread-safe. Use one instance per replica process; read with :meth:`current`
    and increment with :meth:`bump` from the weight-update callback.
    """

    def __init__(self, initial: int = 0) -> None:
        self._version = int(initial)
        self._lock = threading.Lock()

    def current(self) -> int:
        with self._lock:
            return self._version

    def bump(self, to: Optional[int] = None) -> int:
        with self._lock:
            if to is None:
                self._version += 1
            else:
                if int(to) < self._version:
                    raise ValueError(f"VersionClock.bump cannot move backwards: current={self._version}, to={to}")
                self._version = int(to)
            return self._version


@dataclass
class GenerationVersionRecord:
    """Snapshot of policy versions for a single generated trajectory.

    ``start_version`` is the version that was current when prefill began.
    ``end_version`` is the version that was current when the last emitted token
    was sampled. ``per_token_versions``, when present, is a dense 1-D array of
    shape ``(response_len,)`` recording the exact version per emitted token.

    For the common case where the inference engine does not expose per-token
    timestamps, ``per_token_versions`` may be ``None`` and the trainer falls
    back to interpolation between ``start_version`` and ``end_version``.
    """

    start_version: int
    end_version: int
    per_token_versions: Optional[np.ndarray] = None
    response_len: int = 0
    extra: dict = field(default_factory=dict)

    def to_dense(self, response_len: int) -> np.ndarray:
        """Return a dense ``(response_len,)`` int32 array of per-token versions.

        If exact per-token versions were recorded, they are returned (truncated
        or zero-padded as needed). Otherwise versions are linearly interpolated
        between ``start_version`` and ``end_version``; integer-valued so floating
        rounding does not introduce phantom versions. Sequences without any
        version transition return a constant array.
        """
        if response_len <= 0:
            return np.zeros((0,), dtype=np.int32)

        if self.per_token_versions is not None:
            arr = np.asarray(self.per_token_versions, dtype=np.int32)
            if arr.shape[0] >= response_len:
                return arr[:response_len].copy()
            # Pad with the end version for any tail beyond what was recorded.
            padded = np.full((response_len,), self.end_version, dtype=np.int32)
            padded[: arr.shape[0]] = arr
            return padded

        if self.start_version == self.end_version:
            return np.full((response_len,), self.start_version, dtype=np.int32)

        # Linear interpolation: tokens are split as evenly as possible across the
        # versions [start_version, ..., end_version]. This is the conservative
        # estimate when only the begin/end versions are known.
        num_versions = self.end_version - self.start_version + 1
        if num_versions >= response_len:
            # More versions than tokens: assign the first response_len versions
            # consecutively and clamp.
            return np.arange(self.start_version, self.start_version + response_len, dtype=np.int32)

        boundaries = np.linspace(0, response_len, num_versions + 1, dtype=np.int64)
        out = np.empty((response_len,), dtype=np.int32)
        for i, version in enumerate(range(self.start_version, self.end_version + 1)):
            out[boundaries[i] : boundaries[i + 1]] = version
        return out


def attach_versions_to_batch(
    batch,
    records: list[GenerationVersionRecord],
    response_lens: list[int],
    *,
    pad_to: Optional[int] = None,
) -> None:
    """Attach per-token version metadata to a verl ``DataProto``-like batch.

    Stores three keys:
      * ``batch[TOKEN_POLICY_VERSION_KEY]`` -- ``int32`` tensor of shape
        ``(B, T_resp)`` aligned with the response tokens.
      * ``non_tensor_batch[TRAJECTORY_START_VERSION_KEY]`` -- ``int32`` array
        of shape ``(B,)``.
      * ``non_tensor_batch[TRAJECTORY_END_VERSION_KEY]`` -- ``int32`` array of
        shape ``(B,)``.

    Args:
        batch: object with ``batch`` (TensorDict-like) and ``non_tensor_batch``
            (dict-of-ndarrays) attributes.
        records: one :class:`GenerationVersionRecord` per row.
        response_lens: ``len(records) == len(response_lens) == B``; the actual
            number of emitted tokens for each row.
        pad_to: total response width to pad the version tensor to. Defaults to
            ``max(response_lens)``. Padding cells are filled with ``-1`` so they
            can be unambiguously masked in metrics.
    """
    if len(records) != len(response_lens):
        raise ValueError(
            f"records and response_lens must have equal length, got {len(records)} vs {len(response_lens)}"
        )
    if not records:
        return

    width = pad_to if pad_to is not None else max(response_lens)
    if width < 0:
        raise ValueError(f"pad_to must be non-negative, got {pad_to}")

    versions = np.full((len(records), width), -1, dtype=np.int32)
    starts = np.empty((len(records),), dtype=np.int32)
    ends = np.empty((len(records),), dtype=np.int32)
    for i, (rec, n) in enumerate(zip(records, response_lens, strict=True)):
        n_clamped = min(n, width)
        if n_clamped > 0:
            versions[i, :n_clamped] = rec.to_dense(n_clamped)
        starts[i] = rec.start_version
        ends[i] = rec.end_version

    batch.batch[TOKEN_POLICY_VERSION_KEY] = torch.from_numpy(versions)
    batch.non_tensor_batch[TRAJECTORY_START_VERSION_KEY] = starts
    batch.non_tensor_batch[TRAJECTORY_END_VERSION_KEY] = ends


def staleness_from_record(record: GenerationVersionRecord, current_version: int) -> int:
    """Number of policy updates that occurred between trajectory end and now."""
    return max(int(current_version) - int(record.end_version), 0)


def is_in_flight_update(record: GenerationVersionRecord) -> bool:
    """True iff the trajectory was sampled across more than one policy version."""
    return record.end_version > record.start_version
