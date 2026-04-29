# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PipelineRL rollouter.

Specialization of :class:`FullyAsyncRollouter` that maintains a
:class:`VersionClock` and tags every emitted trajectory with the policy
version(s) that produced its tokens.

The rollouter exposes two callbacks, ``on_before_weight_swap`` and
``on_after_weight_swap``, that the trainer's :class:`InflightWeightSync` invokes
around the actual weight broadcast. Between swaps, every produced trajectory is
annotated with ``start_version`` (the clock value at request submission) and
``end_version`` (the clock value at completion). When these differ the
trajectory was sampled across an in-flight weight update, and the
version-aware metrics in :mod:`verl.experimental.pipeline_rl.pipeline_metrics` will report
that fact.

This implementation does not pause generation on weight sync (the underlying
``CheckpointEngineManager.update_weights`` aborts and resumes per request,
which keeps prefixes intact), so the rollouter never blocks on the trainer.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import ray

from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.pipeline_rl.version_tracker import (
    GenerationVersionRecord,
    VersionClock,
    attach_versions_to_batch,
)

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10, max_concurrency=100)
class PipelineRLRollouter(FullyAsyncRollouter):
    """FullyAsyncRollouter with per-token policy-version tagging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._version_clock = VersionClock(initial=0)
        self._pipeline_cfg = self.config.async_training.get("pipeline_rl", {}) or {}
        self._pipeline_enabled = bool(self._pipeline_cfg.get("enabled", True))

    # ----- callbacks invoked by InflightWeightSync on the trainer side -----

    def on_before_weight_swap(self, target_version: int) -> int:
        """Called on every replica just before the trainer broadcasts new weights.

        Returns the clock value seen at the moment of the call (purely for
        tracing / debugging).
        """
        v = self._version_clock.current()
        logger.info(
            "[PipelineRLRollouter] before weight swap: clock=%s target=%s",
            v,
            target_version,
        )
        return v

    def on_after_weight_swap(self, new_version: int) -> int:
        """Called on every replica just after weights have been swapped.

        Bumps the version clock so trajectories emitted from this point on are
        tagged with ``new_version``.
        """
        bumped = self._version_clock.bump(to=int(new_version))
        logger.info("[PipelineRLRollouter] after weight swap: clock=%s", bumped)
        return bumped

    def get_version(self) -> int:
        return self._version_clock.current()

    # ----- per-sample generation override -----

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Generate a single sample and annotate it with the policy version(s)."""
        if not self._pipeline_enabled:
            return await super()._process_single_sample_streaming(rollout_sample)

        start_version = self._version_clock.current()

        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        rollout_sample.full_batch = ret

        end_version = self._version_clock.current()

        # Compute response lengths from the response_mask if present, otherwise from
        # ``responses`` directly. Both code paths exist depending on rollout mode.
        response_lens = _extract_response_lens(ret)

        records = [
            GenerationVersionRecord(
                start_version=start_version,
                end_version=end_version,
                per_token_versions=None,  # exact per-token timestamps require deeper vLLM hooks
                response_len=int(n),
            )
            for n in response_lens
        ]

        try:
            attach_versions_to_batch(ret, records, response_lens=response_lens)
        except Exception:
            logger.exception("[PipelineRLRollouter] failed to attach version tensors")

        # Mirror parent bookkeeping.
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.rollout_status = await self.get_statistics()
        rollout_sample.rollout_status["pipeline_rl/sample_start_version"] = start_version
        rollout_sample.rollout_status["pipeline_rl/sample_end_version"] = end_version

        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1


def _extract_response_lens(batch) -> list[int]:
    """Best-effort extraction of per-row response lengths from a verl batch."""
    rm = batch.batch.get("response_mask") if hasattr(batch, "batch") else None
    if rm is not None:
        return rm.sum(dim=-1).long().tolist()

    responses = batch.batch.get("responses") if hasattr(batch, "batch") else None
    if responses is not None:
        # Fall back to the full response width; metrics still work because
        # padded positions are masked out.
        return [int(responses.shape[-1])] * int(responses.shape[0])

    raise KeyError("Cannot determine response lengths: neither 'response_mask' nor 'responses' present.")


# Optional helper for tests / external triggers.
def make_rollouter_handle(rollouter_actor) -> PipelineRLRollouter:
    """Type-narrowing helper for callsites that received a ray ActorHandle."""
    return rollouter_actor  # type: ignore[return-value]


_ = Optional  # quiet linters in non-runtime module imports
