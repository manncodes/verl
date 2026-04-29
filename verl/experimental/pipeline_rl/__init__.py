# Copyright 2026 The verl-project authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PipelineRL recipe for verl.

PipelineRL is an asynchronous RL training paradigm where rollout generation and
policy training overlap in time and policy weights are pushed to inference engines
*mid-generation*. Tokens belonging to the same trajectory may therefore be sampled
under different policy versions.

This recipe extends :class:`FullyAsyncTrainer` with three additions that together
implement the PipelineRL paradigm on top of verl's existing async infrastructure:

1. ``inflight_weight_sync`` -- weight broadcast that does not pause running rollout
   replicas, exposed as a configurable hook on ``CheckpointEngineManager``.
2. ``version_tracker`` -- per-token policy-version annotation, attached to each
   trajectory via ``meta_info`` and ``non_tensor_batch`` so the trainer can identify
   which weight version produced each emitted token.
3. ``pipeline_metrics`` -- version-aware off-policy diagnostics that complement the
   existing token-level IS in :mod:`verl.trainer.ppo.rollout_corr_helper`.

See ``README.md`` for design notes and known limitations.
"""
