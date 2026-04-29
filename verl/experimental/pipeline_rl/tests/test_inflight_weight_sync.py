# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for InflightWeightSync ordering/serialization invariants.

Run with: ``pytest verl/experimental/pipeline_rl/tests/test_inflight_weight_sync.py``
"""

from __future__ import annotations

import asyncio

import pytest

from verl.experimental.pipeline_rl.inflight_weight_sync import InflightWeightSync


class _FakeMgr:
    def __init__(self):
        self.calls = []

    async def update_weights(self, global_steps=None):
        self.calls.append(("update_weights", global_steps))
        # Simulate some latency.
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_callbacks_fire_in_order():
    mgr = _FakeMgr()
    events: list[tuple[str, int]] = []

    async def before(v):
        events.append(("before", v))

    async def after(v):
        events.append(("after", v))

    sync = InflightWeightSync(mgr, before_swap=before, after_swap=after)
    res = await sync.update_weights(version=7)

    assert res.version == 7
    assert res.duration_s >= 0.0
    # Order: before, broadcast, after.
    assert events[0] == ("before", 7)
    assert events[1] == ("after", 7)
    assert mgr.calls == [("update_weights", 7)]


@pytest.mark.asyncio
async def test_concurrent_updates_serialized():
    mgr = _FakeMgr()
    in_progress = 0
    max_in_progress = 0
    lock = asyncio.Lock()

    async def before(v):
        nonlocal in_progress, max_in_progress
        async with lock:
            in_progress += 1
            max_in_progress = max(max_in_progress, in_progress)

    async def after(v):
        nonlocal in_progress
        async with lock:
            in_progress -= 1

    sync = InflightWeightSync(mgr, before_swap=before, after_swap=after)
    await asyncio.gather(*(sync.update_weights(version=v) for v in range(5)))
    assert max_in_progress == 1, "InflightWeightSync must serialize concurrent calls"


@pytest.mark.asyncio
async def test_callback_exception_propagates():
    mgr = _FakeMgr()

    async def failing(v):
        raise RuntimeError("boom")

    sync = InflightWeightSync(mgr, before_swap=failing)
    with pytest.raises(RuntimeError, match="boom"):
        await sync.update_weights(version=1)
    # Broadcast should not have been issued because before_swap failed.
    assert mgr.calls == []


@pytest.mark.asyncio
async def test_no_callbacks_still_works():
    mgr = _FakeMgr()
    sync = InflightWeightSync(mgr)
    res = await sync.update_weights(version=42)
    assert res.version == 42
    assert mgr.calls == [("update_weights", 42)]
