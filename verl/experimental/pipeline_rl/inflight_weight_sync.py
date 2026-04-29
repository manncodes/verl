# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""In-flight weight synchronization helper for PipelineRL.

verl's :class:`CheckpointEngineManager.update_weights` already implements
*partial rollout*: it aborts in-flight requests, swaps weights, and resumes the
saved prefixes under the new weights. Conceptually that is exactly PipelineRL's
mid-generation weight update, missing only one piece: the version-bump callback
that lets the rollouter tag tokens emitted after the update with a new policy
version.

:class:`InflightWeightSync` is a thin wrapper that

  * delegates the actual broadcast to ``CheckpointEngineManager.update_weights``,
  * brackets the broadcast with ``before_swap`` / ``after_swap`` callbacks the
    rollouter uses to bump its :class:`VersionClock`, and
  * preserves the per-replica timing so we can report mid-generation overhead.

It does not require any change to the existing checkpoint engine; if a future
backend supports a true *non-aborting* in-place swap, only the
``update_weights`` call below needs to switch to the new path -- the API is
already aligned.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

VersionCallback = Callable[[int], Awaitable[None]]


@dataclass
class InflightSyncResult:
    """Timing/diagnostic record for one in-flight weight sync."""

    version: int
    duration_s: float
    aborted_then_resumed: bool


class InflightWeightSync:
    """Coordinator for mid-generation weight broadcast.

    Args:
        checkpoint_manager: an instance of :class:`CheckpointEngineManager`.
        before_swap: optional async callback invoked just before the actual
            ``update_weights`` call. Receives the *target* version. Used by the
            rollouter to flush per-token version records up to this point so
            the trainer-bound metadata is consistent.
        after_swap: optional async callback invoked just after a successful
            broadcast. Receives the new version. Should bump the rollouter's
            :class:`VersionClock` so subsequently emitted tokens are tagged
            with the new version.
        non_aborting: when True, attempts to skip the abort+resume path for
            backends that support a non-aborting swap. Default ``False``.
            Reserved for future use; currently the default partial-rollout
            path of ``CheckpointEngineManager.update_weights`` is invoked
            either way (it preserves the response prefix).
    """

    def __init__(
        self,
        checkpoint_manager,
        *,
        before_swap: Optional[VersionCallback] = None,
        after_swap: Optional[VersionCallback] = None,
        non_aborting: bool = False,
    ) -> None:
        self._mgr = checkpoint_manager
        self._before_swap = before_swap
        self._after_swap = after_swap
        self._non_aborting = non_aborting
        self._lock = asyncio.Lock()

    async def update_weights(self, version: int) -> InflightSyncResult:
        """Run a single in-flight weight update.

        Concurrent calls are serialized: at most one weight sync proceeds at a
        time, matching the underlying checkpoint engine's invariants.
        """
        async with self._lock:
            t0 = time.time()
            if self._before_swap is not None:
                try:
                    await self._before_swap(version)
                except Exception:
                    logger.exception("[InflightWeightSync] before_swap callback failed")
                    raise

            await self._mgr.update_weights(global_steps=version)

            if self._after_swap is not None:
                try:
                    await self._after_swap(version)
                except Exception:
                    logger.exception("[InflightWeightSync] after_swap callback failed")
                    raise

            duration = time.time() - t0
            logger.info(
                "[InflightWeightSync] version=%s duration=%.4fs non_aborting=%s",
                version,
                duration,
                self._non_aborting,
            )
            return InflightSyncResult(
                version=version,
                duration_s=duration,
                aborted_then_resumed=not self._non_aborting,
            )
