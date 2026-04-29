# PipelineRL recipe for verl

This recipe implements the **PipelineRL** asynchronous-RL paradigm on top of
verl's existing async infrastructure. PipelineRL pushes new policy weights to
the rollout engines *while* trajectories are still being generated, so a single
trajectory may contain tokens sampled under multiple policy versions. With
proper per-token importance sampling this keeps the inference GPUs maximally
utilised while limiting off-policy bias.

This is a **scaffolding-quality** implementation: it composes verl's existing
parts (FullyAsyncTrainer + partial rollout in `CheckpointEngineManager` +
token-level IS in `rollout_corr_helper`) with three new ingredients required
to make PipelineRL semantics concrete. It is intended as the basis for further
work, not a fully validated training recipe — see "Status" below.

## What's here

```
verl/experimental/pipeline_rl/
  version_tracker.py        # VersionClock + per-token version annotation
  pipeline_metrics.py       # Version-bucketed off-policy diagnostics
  inflight_weight_sync.py   # Trainer-side wrapper around CheckpointEngineManager
  pipeline_rl_trainer.py    # Subclass of FullyAsyncTrainer with eager weight sync
  pipeline_rl_rollouter.py  # Subclass of FullyAsyncRollouter with version tagging
  main.py                   # Hydra entrypoint (mirrors fully_async_main.py)
  config/pipeline_rl_trainer.yaml
  tests/                    # Unit tests for the new modules
```

## How it relates to existing verl pieces

Most of the heavy lifting was already in verl:

| Need                           | Existing verl piece                                           | What this recipe adds                                  |
| ------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------ |
| Decoupled actor / rollout      | `verl.experimental.fully_async_policy.FullyAsyncTrainer`      | Subclass with eager weight sync                        |
| Async generation               | `verl.experimental.agent_loop.AgentLoopManager`               | (reused as-is)                                         |
| Pause-free weight sync         | `CheckpointEngineManager.update_weights` (abort + resume)     | `InflightWeightSync` callback shim                     |
| Per-token IS / TIS / IcePop    | `verl.trainer.ppo.rollout_corr_helper`                        | (reused via `bypass_mode + use_rollout_log_probs`)     |
| Per-token policy version       | —                                                             | `VersionClock`, `GenerationVersionRecord`, `attach_*`  |
| Version-aware diagnostics      | —                                                             | `compute_pipeline_version_metrics`                     |

The two key insights that let us reuse the existing pieces:

1. `CheckpointEngineManager.update_weights` already aborts in-flight requests,
   broadcasts new weights, and **resumes generation from the saved prefix**.
   That is exactly the partial-rollout semantics PipelineRL wants. This recipe
   wraps that call in `InflightWeightSync` only to add `before_swap` /
   `after_swap` callbacks that the rollouter uses to bump its `VersionClock`.
2. With `bypass_mode + use_rollout_log_probs + calculate_log_probs=True`
   (already available in fully-async), every token carries the rollout-time
   log probability of the policy version that *actually sampled it*. Plugging
   that into the existing token-level IS in `rollout_corr_helper` yields a
   correct per-token importance correction even when one trajectory spans
   multiple versions — no new IS code is required for correctness.

What `version_tracker` adds is a way to **observe** which version sampled which
token, so `pipeline_metrics` can break down KL/log-ratio/staleness by version
gap. That is essential for debugging PipelineRL runs (e.g. confirming that
older tokens really do have higher variance).

## Configuration knobs

`config/pipeline_rl_trainer.yaml` exposes:

```yaml
async_training:
  staleness_threshold: 0           # 0 = strict pipeline (no stale batches)
  trigger_parameter_sync_step: 1   # weight broadcast every actor step
  partial_rollout: True            # required (preserves prefix across swaps)
  pipeline_rl:
    enabled: True                  # off => behaves exactly like FullyAsync
    non_aborting_swap: False       # reserved for future engine support
algorithm:
  rollout_correction:
    bypass_mode: True              # required: old_log_probs := rollout_log_probs
    rollout_is: token              # token-level IS for mixed-version trajectories
    rollout_is_threshold: 2.0
```

## Running

```bash
python -m verl.experimental.pipeline_rl.main \
  actor_rollout_ref.model.path=<model_path> \
  data.train_files=<train.parquet> \
  data.val_files=<val.parquet> \
  trainer.n_gpus_per_node=4 \
  rollout.n_gpus_per_node=4
```

Any override accepted by the standard `ppo_trainer` config works because the
recipe inherits from it via the Hydra `defaults` list.

## Tests

```bash
pytest verl/experimental/pipeline_rl/tests/ -v
```

The unit tests cover the modules that have no Ray / GPU dependency:
`version_tracker`, `pipeline_metrics`, `inflight_weight_sync`. The trainer and
rollouter classes are thin Ray actors; their behavior is exercised by an
end-to-end run.

## Status and known limitations

* **Not yet end-to-end tested with a real model.** The recipe is wired up but
  has not been smoke-tested against a live LLM. Expect to find integration
  bugs (Ray serialization, config defaults, etc.) on the first run.
* **Per-token version is interpolated, not measured.** Recording exact per-token
  timestamps requires hooks inside vLLM/SGLang's token loop. Until that lands,
  `GenerationVersionRecord.per_token_versions` is `None` and tokens are bucketed
  evenly between `start_version` and `end_version`. This is correct on average
  and good enough for diagnostics; it is not exact for IS.
* **Non-aborting swap is reserved.** Today the swap path still aborts and
  resumes via the standard checkpoint engine. The `non_aborting_swap` flag is a
  placeholder for backends that grow true in-place updates.
* **Trainer-side validation (`use_trainer_do_validate=True`) is not supported.**
  The base `FullyAsyncTrainer` path mutates checkpoint backends mid-flight in a
  way that is incompatible with eager weight broadcast; this recipe asserts off.
* **CLAUDE.md compliance.** AI assistance was used to produce this recipe. A
  human submitter must review every line and run the relevant tests before
  opening any PR.

## References

* Piché et al., *PipelineRL* (ServiceNow Research) — original paper / blog.
* `verl.experimental.fully_async_policy` — async PPO baseline this recipe
  builds on.
* `verl.trainer.ppo.rollout_corr_helper` — token-level IS / RS implementation
  reused unchanged.
