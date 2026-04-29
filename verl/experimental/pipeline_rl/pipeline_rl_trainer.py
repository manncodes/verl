# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PipelineRL trainer.

Specialization of :class:`FullyAsyncTrainer` that performs eager, in-flight
weight broadcasts and records version-aware off-policy metrics.

Key differences vs. the baseline fully-async trainer:

  * ``trigger_parameter_sync_step`` defaults to 1 -- weights are broadcast every
    actor step (modulo a configurable minimum interval) instead of being
    batched.
  * Weight broadcasts go through :class:`InflightWeightSync`, which calls back
    into the rollouter so the per-token :class:`VersionClock` advances exactly
    once per swap.
  * Per-step metrics include version-bucketed off-policy diagnostics from
    :func:`verl.experimental.pipeline_rl.pipeline_metrics.compute_pipeline_metrics_from_batch`.

The extension is additive: when ``async_training.pipeline_rl.enabled=False`` the
recipe falls back to FullyAsyncTrainer's exact behavior.
"""

from __future__ import annotations

import logging

import ray

from verl import DataProto
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.pipeline_rl.inflight_weight_sync import InflightWeightSync
from verl.experimental.pipeline_rl.pipeline_metrics import compute_pipeline_metrics_from_batch

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10)
class PipelineRLTrainer(FullyAsyncTrainer):
    """FullyAsyncTrainer with eager in-flight weight broadcast and version-aware metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PipelineRL-specific config block (optional; falls back to fully-async defaults).
        self._pipeline_cfg = self.config.async_training.get("pipeline_rl", {}) or {}
        self._pipeline_enabled = bool(self._pipeline_cfg.get("enabled", True))
        # Force eager weight sync when pipeline_rl is enabled: bypass FullyAsyncTrainer's
        # ``local_trigger_step`` gating so updates fire on every actor step.
        if self._pipeline_enabled:
            self.trigger_parameter_sync_step = 1
        # Lazy-built; needs ``self.checkpoint_manager`` and ``self.rollouter`` to exist.
        self._inflight_sync: InflightWeightSync | None = None

    # ----- weight sync overrides -----

    def _build_inflight_sync(self) -> InflightWeightSync:
        async def _before_swap(version: int) -> None:
            # Tell the rollouter to flush the version-record buffer for any
            # in-flight trajectory before the swap, so tokens emitted up to this
            # instant carry the pre-swap version.
            await self._rollouter_call("on_before_weight_swap", version)

        async def _after_swap(version: int) -> None:
            # Bump the rollouter's VersionClock; subsequent tokens are tagged
            # with ``version``.
            await self._rollouter_call("on_after_weight_swap", version)

        return InflightWeightSync(
            checkpoint_manager=self.checkpoint_manager,
            before_swap=_before_swap,
            after_swap=_after_swap,
            non_aborting=bool(self._pipeline_cfg.get("non_aborting_swap", False)),
        )

    async def _rollouter_call(self, method_name: str, *args) -> None:
        """Call an optional method on the rollouter; tolerate older rollouters that lack it."""
        if self.rollouter is None:
            return
        method = getattr(self.rollouter, method_name, None)
        if method is None:
            return
        try:
            import asyncio

            await asyncio.wrap_future(method.remote(*args).future())
        except Exception:
            logger.exception("[PipelineRLTrainer] rollouter callback %s failed", method_name)

    async def _fit_update_weights(self):
        if not self._pipeline_enabled:
            await super()._fit_update_weights()
            return

        # Eager update path: bump version every step; do not gate on
        # ``local_trigger_step``. The ``current_param_version`` numbering still
        # ties checkpoints/metrics to a monotonic counter.
        from verl.utils.debug import marked_timer

        # Advance the parameter version; FullyAsyncTrainer.fit_step calls
        # ``_fit_update_local_step`` *before* this, which already increments
        # ``current_param_version`` when the trigger threshold is met. With
        # ``trigger_parameter_sync_step=1`` the version increments every call.
        with marked_timer("timing_s/param_sync", self.timing_raw):
            if self._inflight_sync is None:
                self._inflight_sync = self._build_inflight_sync()
            result = await self._inflight_sync.update_weights(version=self.current_param_version)

        self.timing_raw["pipeline_rl/inflight_sync_duration"] = result.duration_s
        logger.info(
            "[PipelineRLTrainer] eager weight sync version=%s duration=%.4fs",
            result.version,
            result.duration_s,
        )

        # Reset rollouter staleness; same as FullyAsyncTrainer but unconditional.
        import asyncio

        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(data=timing_raw, step=self.current_param_version)
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    # ----- metrics overrides -----

    def _collect_metrics_from_samples(self, batch: DataProto, metrics: dict) -> None:
        super()._collect_metrics_from_samples(batch, metrics)
        if not self._pipeline_enabled:
            return
        try:
            pipeline_metrics = compute_pipeline_metrics_from_batch(batch, current_version=self.current_param_version)
        except Exception:
            logger.exception("[PipelineRLTrainer] pipeline metrics computation failed; skipping")
            return
        metrics.update(pipeline_metrics)
