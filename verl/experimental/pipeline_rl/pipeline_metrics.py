# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Version-aware off-policy diagnostics for PipelineRL.

verl's existing :mod:`verl.trainer.ppo.rollout_corr_helper` already computes
token-level importance-sampling weights and KL/PPL/chi-squared diagnostics from
``old_log_prob`` and ``rollout_log_prob``. PipelineRL extends those diagnostics
with metrics bucketed by *policy-version gap*: how stale each token is relative
to the current trainer version.

The functions here are deliberately additive: they read the version tensor
attached by :mod:`verl.experimental.pipeline_rl.version_tracker` and produce a flat
``dict[str, float]`` suitable for the verl tracking logger. They do not modify
the loss or weights themselves -- token-level IS continues to be computed by the
existing helper.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from verl.experimental.pipeline_rl.version_tracker import (
    TOKEN_POLICY_VERSION_KEY,
    TRAJECTORY_END_VERSION_KEY,
    TRAJECTORY_START_VERSION_KEY,
)

_PAD_VERSION = -1


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def compute_pipeline_version_metrics(
    *,
    token_policy_version: torch.Tensor,
    response_mask: torch.Tensor,
    current_version: int,
    log_ratio: Optional[torch.Tensor] = None,
    trajectory_start_version: Optional[np.ndarray] = None,
    trajectory_end_version: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute version-bucketed off-policy diagnostics.

    Args:
        token_policy_version: ``int`` tensor of shape ``(B, T_resp)`` with the
            policy version that produced each token. Padding cells should be
            ``-1`` (the convention used by
            :func:`verl.experimental.pipeline_rl.version_tracker.attach_versions_to_batch`).
        response_mask: ``(B, T_resp)`` mask of valid response tokens.
        current_version: trainer-side current parameter version.
        log_ratio: optional ``(B, T_resp)`` log ratio
            ``log(pi_train(t) / pi_rollout(t))`` -- when provided, mean log
            ratio is reported per version bucket so collapse can be detected
            independently for fresh vs. stale tokens.
        trajectory_start_version, trajectory_end_version: optional ``(B,)``
            arrays. When both are present, fraction of in-flight-updated
            trajectories is reported.

    Returns:
        Dict of scalars prefixed with ``pipeline_rl/``.
    """
    if token_policy_version.shape != response_mask.shape:
        raise ValueError(
            f"token_policy_version shape {token_policy_version.shape} does not match "
            f"response_mask shape {response_mask.shape}."
        )

    metrics: dict[str, float] = {}

    versions = _to_tensor(token_policy_version).to(torch.int64)
    mask = _to_tensor(response_mask).bool()
    valid = mask & (versions != _PAD_VERSION)

    n_valid = int(valid.sum().item())
    metrics["pipeline_rl/valid_tokens"] = float(n_valid)
    if n_valid == 0:
        return metrics

    valid_versions = versions[valid].to(torch.float64)
    staleness = current_version - valid_versions
    staleness_clamped = staleness.clamp_min(0.0)

    metrics["pipeline_rl/current_version"] = float(current_version)
    metrics["pipeline_rl/token_version_min"] = float(valid_versions.min().item())
    metrics["pipeline_rl/token_version_max"] = float(valid_versions.max().item())
    metrics["pipeline_rl/token_version_mean"] = float(valid_versions.mean().item())
    metrics["pipeline_rl/token_staleness_mean"] = float(staleness_clamped.mean().item())
    metrics["pipeline_rl/token_staleness_max"] = float(staleness_clamped.max().item())

    # Fraction of tokens whose generating version differs from current.
    metrics["pipeline_rl/token_off_policy_fraction"] = float((staleness_clamped > 0).float().mean().item())

    # Per-staleness-bucket mean log-ratio (variance proxy for off-policyness).
    if log_ratio is not None:
        lr = _to_tensor(log_ratio)
        if lr.shape != response_mask.shape:
            raise ValueError(f"log_ratio shape {lr.shape} does not match response_mask shape {response_mask.shape}.")
        lr_valid = lr[valid].to(torch.float64)
        for bucket in (0, 1, 2):
            bucket_mask = staleness_clamped == bucket
            count = int(bucket_mask.sum().item())
            metrics[f"pipeline_rl/staleness_{bucket}/count"] = float(count)
            if count > 0:
                metrics[f"pipeline_rl/staleness_{bucket}/log_ratio_mean"] = float(lr_valid[bucket_mask].mean().item())
                metrics[f"pipeline_rl/staleness_{bucket}/log_ratio_abs_mean"] = float(
                    lr_valid[bucket_mask].abs().mean().item()
                )

        # Aggregate "stale" bucket (anything older than current).
        stale_mask = staleness_clamped >= 1
        stale_count = int(stale_mask.sum().item())
        metrics["pipeline_rl/staleness_ge1/count"] = float(stale_count)
        if stale_count > 0:
            metrics["pipeline_rl/staleness_ge1/log_ratio_mean"] = float(lr_valid[stale_mask].mean().item())
            metrics["pipeline_rl/staleness_ge1/log_ratio_abs_mean"] = float(lr_valid[stale_mask].abs().mean().item())

    # Trajectory-level: fraction of trajectories that experienced a
    # mid-generation weight update.
    if trajectory_start_version is not None and trajectory_end_version is not None:
        starts = np.asarray(trajectory_start_version, dtype=np.int64)
        ends = np.asarray(trajectory_end_version, dtype=np.int64)
        if starts.shape != ends.shape:
            raise ValueError("trajectory_start_version and trajectory_end_version must share shape.")
        if starts.size > 0:
            inflight = ends > starts
            metrics["pipeline_rl/trajectory_inflight_update_fraction"] = float(inflight.mean())
            metrics["pipeline_rl/trajectory_version_span_max"] = float((ends - starts).max())
            metrics["pipeline_rl/trajectory_version_span_mean"] = float((ends - starts).mean())

    return metrics


def compute_pipeline_metrics_from_batch(
    batch,
    *,
    current_version: int,
    response_mask_key: str = "response_mask",
    old_log_prob_key: str = "old_log_probs",
    rollout_log_prob_key: str = "rollout_log_probs",
) -> dict[str, float]:
    """Convenience wrapper that pulls tensors out of a verl ``DataProto``-like batch.

    Returns an empty dict if the batch was not annotated with version info
    (e.g. the recipe is running without inflight weight updates), which makes
    the call safe to insert into the standard fit step.
    """
    if TOKEN_POLICY_VERSION_KEY not in batch.batch:
        return {}

    token_versions = batch.batch[TOKEN_POLICY_VERSION_KEY]
    response_mask = batch.batch[response_mask_key]

    log_ratio: Optional[torch.Tensor] = None
    if old_log_prob_key in batch.batch and rollout_log_prob_key in batch.batch:
        log_ratio = batch.batch[old_log_prob_key] - batch.batch[rollout_log_prob_key]

    starts = batch.non_tensor_batch.get(TRAJECTORY_START_VERSION_KEY)
    ends = batch.non_tensor_batch.get(TRAJECTORY_END_VERSION_KEY)

    return compute_pipeline_version_metrics(
        token_policy_version=token_versions,
        response_mask=response_mask,
        current_version=current_version,
        log_ratio=log_ratio,
        trajectory_start_version=starts,
        trajectory_end_version=ends,
    )
