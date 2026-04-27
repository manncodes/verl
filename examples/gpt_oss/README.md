# gpt-oss launch scripts

End-to-end recipe for training **openai/gpt-oss-20b** with verl and verifying
the forward/backward pass before kicking off a real run.

## Files

| file | purpose |
| --- | --- |
| `install.sh` | One-shot uv install (torch + verl + sglang + flash-attn) |
| `Dockerfile` | gpt-oss image: sglang base + verl + Mxfp4-capable transformers |
| `build_with_colima.sh` | Clone verl + start colima + build the image |
| `prepare_model.py` | Dequantize the HF MXFP4 release to bf16 (one-time) |
| `test_attention_sinks.py` | Verify gpt-oss sinks are wired through the actor's attention path |
| `test_router_replay_capability.py` | R3 capability check (sglang recorder, hf_config, router determinism) |
| `test_rollout_e2e.py` | Heavyweight: boot sglang/vllm, generate, then HF backward through the response |
| `check_gpt_oss_fwd_bwd.py` | Standalone forward/backward correctness check |
| `run_check.sh` | Wrapper around the check that also dequantizes if needed |
| `launch_train_gpt_oss_20b.sh` | GRPO training on GSM8K with FSDP + sglang |
| `wandb_ray_metrics.py` | Sidecar: forward Ray's per-node Prometheus metrics to wandb |
| `sonic_moe_patch.py` | Scaffolding to swap HF's GptOssMoE with Dao-AILab/sonic-moe (experimental) |
| `test_sonic_moe.py` | Forward probe: sonic-moe vs gpt-oss clamped GLU, numerics + wall-clock |
| `test_sonic_moe_fwd_bwd.py` | Forward + backward parity test (vanilla SwiGLU on both sides) |
| `flex_attention_sinks.py` | Sinks-aware attention via PyTorch flex_attention; self-test included |
| `benchmark_speedups.py` | Microbenchmark: attention impls × MoE impls, correctness + stability + timing |

## One-shot flow

A single command does dependency check -> dequantize -> preprocess gsm8k ->
forward/backward correctness check -> launch GRPO training:

```bash
# 0. (one-time) install verl + sglang + transformers in a uv-managed venv
bash examples/gpt_oss/install.sh
source .venv/bin/activate

# 1. run everything
bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

If `flash-attn` fails to build (no CUDA toolchain, or no prebuilt wheel for
your torch/cuda combo), re-run with:

```bash
SKIP_FLASH_ATTN=1 bash examples/gpt_oss/install.sh
```

verl runs fine without flash-attn — gpt-oss uses `attn_implementation=eager`
by default in this recipe.

### Or: containerised build via colima

Prefer a sealed environment over installing into the host? Use the colima
wrapper to clone the repo, bring up colima, and build the image in one shot:

```bash
# from anywhere; clones https://github.com/manncodes/verl into ./verl
bash <(curl -sL https://raw.githubusercontent.com/manncodes/verl/main/examples/gpt_oss/build_with_colima.sh)

# or from an existing checkout
bash examples/gpt_oss/build_with_colima.sh
```

Knobs (env vars): `VERL_REPO`, `VERL_REF`, `IMAGE_TAG`, `COLIMA_CPU`,
`COLIMA_MEMORY`, `COLIMA_DISK`, `COLIMA_ARCH`. The defaults pin
`COLIMA_ARCH=x86_64` because the sglang + CUDA base image is amd64-only,
which matters on Apple Silicon.

Note: colima on macOS has no GPU passthrough — the image builds on macOS,
but training itself still needs to run on a Linux box with CUDA-capable
GPUs. Push the image to a registry and pull it on the GPU host.

Each stage is idempotent: re-running skips work that's already done. To run
just the correctness check (no training):

```bash
SKIP_TRAIN=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

To skip the correctness check (e.g. on a re-launch after a crash):

```bash
SKIP_CHECK=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

Override any other knob via env vars, e.g.:

```bash
N_GPUS_PER_NODE=4 ROLLOUT_TP_SIZE=2 \
TRAIN_BATCH_SIZE=128 PPO_MICRO_BATCH_SIZE_PER_GPU=16 \
bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

`run_check.sh` is also available as a standalone wrapper around the
correctness check if you only want that piece.

## Why these defaults

### Loading / runtime

- **MXFP4 -> bf16 once**: the FSDP/Megatron training paths cannot ingest the
  shipped MXFP4 weights; we dequantize via `Mxfp4Config(dequantize=True)` and
  stamp `attn_implementation=eager` on the saved config so training picks the
  supported attention path.
- **`rollout.name=sglang` + `attention_backend=triton` + `mode=async`**: the
  combination explicitly supported for gpt-oss in the upstream example.
- **`load_format=safetensors`**: required after dequantization so the rollout
  weight transfer path can ingest the bf16 shards.
- **`use_remove_padding=True`** plus FSDP: the FSDP path supports the THD
  packed format. Megatron currently needs BSHD (PR #4323) — switch to the
  Megatron recipe under `examples/grpo_trainer/run_qwen3moe-30b_megatron_*.sh`
  as a starting template if you need the Megatron backend.

### MoE training stability

| flag (env var → hydra override) | default | why it matters for gpt-oss |
| --- | --- | --- |
| `model.override_config.attn_implementation=eager` | `eager` | verl's HFModelConfig defaults to `flash_attention_2`; FA2 silently bypasses gpt-oss attention sinks (and 2.8.x has an ABI mismatch with torch 2.9.1 → ImportError on `c10_cuda_check_implementation`). Forcing `eager` matches what `prepare_model.py` saved into `config.json` and what the precheck verifies. |
| `TRAIN_BATCH_SIZE_PER_NODE` × `NNODES` (`data.train_batch_size`) | `256 * NNODES` | Global batch scales linearly with `NNODES`. `TRAIN_BATCH_SIZE` overrides if set explicitly. |
| `TRAIN_BATCH_SIZE == PPO_MINI_BATCH_SIZE` (`actor.ppo_mini_batch_size`) | equal | MoE training diverges quickly when the two differ; upstream example keeps them equal. |
| `ENABLE_TIS=1` (`algorithm.rollout_correction.rollout_is`, `rollout.calculate_log_probs`) | on, token-level | Issue #3894 reports `rollout_actor_probs_pearson_corr ~ 0.5` from training/rollout drift — TIS is the supported mitigation. Set `TIS_LEVEL=sequence` for higher-variance unbiased weights. |
| `TIS_THRESHOLD` (`algorithm.rollout_correction.rollout_is_threshold`) | `2.0` | Per the upstream guide: 1.5–5.0 for token-TIS, 2.0–10.0 for sequence-TIS. |
| `USE_DYNAMIC_BSZ=False` (`actor.use_dynamic_bsz`) | off | Required by the gpt-oss megatron path (PR #4323) and safer for FSDP MoE since dynamic packing changes routing per step. |
| `USE_TORCH_COMPILE=False` (`actor.use_torch_compile`, `ref.use_torch_compile`) | off | torch.compile + MoE has been a recurring source of breakage; several fsdp/sglang examples hard-code it off. |
| `actor.use_kl_loss=True`, `kl_loss_type=low_var_kl`, `kl_loss_coef=0.001` | on | GRPO low-variance KL penalty; recommended by the upstream gpt-oss recipe. |
| `actor.entropy_coeff=0` | 0 | Extra entropy on top of GRPO can destabilise the router. |
| `actor.model.enable_gradient_checkpointing=True` | on | Memory pressure dominates with 32 experts × ~3.6B params each. |

Set `ENABLE_TIS=0` to drop back to vanilla GRPO (matches the existing
`examples/grpo_trainer/run_gptoss_20b.sh` baseline).

### Scaling across nodes

The launcher follows a strict per-node convention so multi-node runs are a
single env-var change. `TRAIN_BATCH_SIZE_PER_NODE` is the only batch knob
you set; the launcher multiplies it by `NNODES` for the global batch.

| knob | scaling | reason |
| --- | --- | --- |
| `TRAIN_BATCH_SIZE` | `TRAIN_BATCH_SIZE_PER_NODE * NNODES` | Linear weak scaling — keeps per-GPU work constant. |
| `PPO_MINI_BATCH_SIZE` | `= TRAIN_BATCH_SIZE` | MoE stability requirement. |
| `PPO_MICRO_BATCH_SIZE_PER_GPU` | constant | Fixed by GPU memory, not topology. |
| `actor.rollout.log_prob_micro_batch_size_per_gpu` | constant | Same as above. |
| `ROLLOUT_TP_SIZE` | constant | Picked by model size; gpt-oss-20b fits at TP=2. |
| `ROLLOUT_N` | constant | Algorithmic — generations per prompt. |

Examples:

```bash
# 1 H100 node (default): 256 prompts/step
bash examples/gpt_oss/launch_train_gpt_oss_20b.sh

# 4 H100 nodes: auto-derives TRAIN_BATCH_SIZE=1024
NNODES=4 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh

# Smaller per-node batch on a memory-tight cluster
TRAIN_BATCH_SIZE_PER_NODE=128 NNODES=2 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh

# Override the auto-scaled global batch directly
TRAIN_BATCH_SIZE=384 NNODES=3 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

The launcher prints the effective topology + batch shape just before training
starts, plus warnings if the global batch is not divisible by total GPUs or
if `train_batch_size` and `ppo_mini_batch_size` differ.

### Attention sinks correctness

gpt-oss attention layers carry learnable per-head **sink scores** that are
added to the softmax denominator. Most attention backends silently drop them
and the model trains on subtly wrong logits. Only `eager` (HF default for
gpt-oss), FlashAttention 3, and TRTLLM honour sinks.

`test_attention_sinks.py` runs as the first preflight in the launcher and
asserts:

1. Every layer exposes a `sinks` parameter of shape `[num_heads]`.
2. The sinks tensor is non-zero (i.e. the bf16 checkpoint actually loaded
   the trained values).
3. `model.config._attn_implementation == "eager"`.
4. **Forward logits change when sinks are zeroed** — proves the attention
   kernel is using them, catches the silent SDPA / FA2 / FlashInfer bypass.
5. Backward through the loss accumulates non-zero gradient on every sinks
   parameter — proves they're in the autograd graph.
6. **Cross-backend probe** (default on; `--no-compare-backends` to skip):
   reload the model with each of `sdpa` and `flash_attention_2`, rerun the
   sink-effect probe on each. Backends that load successfully but produce
   bit-identical logits with and without sinks are silently bypassing them
   — that's the regression class, and it fails the test loudly. Backends
   that fail to load (e.g. flash-attn ABI mismatch with the active torch)
   are reported and skipped — that's a different failure mode and doesn't
   indicate a correctness bug, just an install issue.

Run standalone:

```bash
python examples/gpt_oss/test_attention_sinks.py --model-dir /model/Huggingface/openai/gpt-oss-20b-bf16
```

Skip the auto-run: `SKIP_SINKS_TEST=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh`.

References for the bug class:
[unsloth #3142](https://github.com/unslothai/unsloth/issues/3142)
(SDPA bypass);
[vllm #22331](https://github.com/vllm-project/vllm/issues/22331),
[#22279](https://github.com/vllm-project/vllm/issues/22279) (FA2 errors);
[vllm #30919](https://github.com/vllm-project/vllm/issues/30919) (FlashInfer);
[NVIDIA/TE #2070](https://github.com/NVIDIA/TransformerEngine/issues/2070).

### R3 routing replay

`test_router_replay_capability.py` runs after the sinks test and checks
the recording side of router replay is wired:

1. `verl.workers.config.RouterReplayConfig` exists; the rollout YAML has
   `enable_rollout_routing_replay`.
2. The hf_config exposes `num_hidden_layers`, `num_experts_per_tok`,
   `num_local_experts`.
3. sglang has `extract_routed_experts_from_meta_info` (the patched build
   from sgl-project/sglang commit `bed301a5`); warns (not fails) if not.
4. The HF router is **deterministic** under fixed inputs — two forward
   passes produce bit-identical top-k expert ids per layer. Without
   this, replaying recorded routes is meaningless.

The actor-side replay is still megatron-only
(`verl/workers/engine_workers.py:477`); on FSDP this test is mainly a
discovery + correctness check for a future megatron switch.

Skip with `SKIP_R3_TEST=1`.

### Rollout end-to-end (opt-in)

`test_rollout_e2e.py` is the heavyweight precheck — opt in with
`RUN_ROLLOUT_TEST=1`. It actually boots the rollout engine on the bf16
checkpoint, generates ~16 tokens, then feeds (prompt + generated tokens)
through the HF actor for forward+backward. This catches the bug class
that the static tests can't:

- vLLM gpt-oss FA2/SDPA sinks bypass on Hopper / earlier ([vllm #22331](https://github.com/vllm-project/vllm/issues/22331), [#22279](https://github.com/vllm-project/vllm/issues/22279), [#30919](https://github.com/vllm-project/vllm/issues/30919))
- vLLM 0.12+ harmony encoding pre-warm requirement
- sglang attention-backend defaults
- tokenizer drift between rollout and training stacks

Both sglang and vLLM are tested if installed; skip individually with
`--skip-sglang` / `--skip-vllm` when invoking the script directly. Each
engine load is ~30s and grabs ~40GB of GPU memory.

```bash
# from the launcher (off by default)
RUN_ROLLOUT_TEST=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh

# standalone
python examples/gpt_oss/test_rollout_e2e.py --model-dir /model/Huggingface/openai/gpt-oss-20b-bf16 --tensor-parallel-size 2
```

### Not enabled (Megatron-only)

- **Router replay (R2/R3)** is wired only for the Megatron actor today
  (`verl/workers/engine_workers.py:477` gates on `actor.strategy=="megatron"`).
  If you switch backends, see `examples/router_replay/` for the recipe.
- **Expert parallel (EP/ETP)** lives under `actor.megatron.*`; n/a for FSDP.

## Speedup options

The 17-min/step baseline (with default offloads on) is dominated by
`update_actor` at 87.7% of step time. Most of the speedups land there.

### `FAST_PRESET=1` — bundled known-safe speedups

```bash
FAST_PRESET=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

Sets, in one go:

- **Right-sized sequence budget**: `MAX_PROMPT_LENGTH=256`,
  `MAX_RESPONSE_LENGTH=1024`. gsm8k under `reasoning_effort=medium` has
  mean prompt ≈ 137 tok and mean response ≈ 350 tok; the previous 512 /
  2048 budget left token utilisation under 6%. Eager attention is
  O(seq²), so wasted budget burns both compute and activation memory.
- **Packed micro-batches**: `PPO_MICRO_BATCH_SIZE_PER_GPU=8` (up from
  2). With the tighter 1280-token per-seq budget, 8 sequences pack to
  ~10k tokens per GPU per micro-step — well under the
  `ppo_max_token_len_per_gpu=16384` default. Drops accumulation steps
  from 16 to 4 per mini-batch on 8 GPUs.
- **No FSDP offloads**: `PARAM_OFFLOAD=False`, `OPTIMIZER_OFFLOAD=False`,
  `ACTIVATION_OFFLOAD=False` (5-10× on `update_actor`, costs HBM that's
  now freed by the smaller seq budget and the lower KV reserve).
- **Ref policy on GPU**: `REF_PARAM_OFFLOAD=False` (saves ~half the 63 s
  ref forward).
- **Ulysses SP=2**: eager-attention compute drops ~4×. Only set if
  `N_GPUS_PER_NODE` is even.
- **Lower KV reservation**: `ROLLOUT_GPU_MEM_UTIL=0.5` (smaller responses
  need less KV cache, freeing HBM for the now-resident actor).
- **bypass_mode**: `ENABLE_BYPASS_MODE=1` skips the third actor forward
  each step (~3% on the measured profile, free).

Each of these is overridable individually after the preset:
`FAST_PRESET=1 PPO_MICRO_BATCH_SIZE_PER_GPU=4 bash …` if you OOM at 8;
`FAST_PRESET=1 MAX_RESPONSE_LENGTH=2048 bash …` if your dataset has
longer responses than gsm8k. Watch the trainer's truncation rate; if it
climbs, bump `MAX_RESPONSE_LENGTH` back up.

Expected on 8×H100 80GB at the previous 1027 s baseline (gsm8k,
reasoning_effort=medium):

| phase | offloads-on baseline | FAST_PRESET=1 (estimated) |
| --- | --- | --- |
| `update_actor` | 900 s | 80-130 s |
| `ref` | 63 s | 25-30 s |
| `old_log_prob` | 31 s | 0 s (bypass) |
| `gen` | 26 s | 25 s |
| total step | ~1027 s | ~150-200 s |

### Flex Attention with sinks (experimental, NOT wired into training yet)

`flex_attention_sinks.py` implements the sinks-aware attention via
PyTorch's `flex_attention`, using the "sink as extra K position" trick:
append one extra key/value position with `V=0`, set the score there to
the per-head learned `sinks[h]` via `score_mod`. This is mathematically
equivalent to gpt-oss's `softmax([scores | sinks])[..., :-1] @ V` and
runs a Triton-compiled kernel that supports backward (unlike FA3, which
the community has confirmed broken for sinks backward).

Validate before integrating:

```bash
# fp32 strict parity probe (small tensors)
python examples/gpt_oss/flex_attention_sinks.py --self-test

# bf16 broader benchmark including sdpa-bypass detection
python examples/gpt_oss/benchmark_speedups.py
python examples/gpt_oss/benchmark_speedups.py --sliding-window 128
```

Wiring this into the verl actor requires registering a custom
`attn_implementation="flex"` in transformers' attention dispatch. That is
NOT in this repo yet — once the self-test passes at fp32 with tight
tolerance and the benchmark shows a speedup at bf16, the next step is to
either monkey-patch `transformers.models.gpt_oss.modeling_gpt_oss` or
contribute the impl upstream.

### Combined microbenchmark

```bash
python examples/gpt_oss/benchmark_speedups.py            # attention only (fast)
python examples/gpt_oss/benchmark_speedups.py --moe      # also exercise MoE
python examples/gpt_oss/benchmark_speedups.py --dtype float32  # tighter correctness floor
```

Reports, for each variant:

- **correctness**: max grad diff vs the gpt-oss-correct eager reference
- **stability**: peak GPU memory and a finite-output check
- **timing**: forward + backward wall-clock per pass

The `sdpa (NO sinks)` variant is included as the reference for what
silently dropping sinks looks like — its `out Δ` column quantifies how
wrong "training on SDPA" actually is on gpt-oss.

## sonic-moe integration (experimental)

[Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe) ships
grouped-GEMM MoE kernels for Hopper and Blackwell. Replacing HF's
`GptOssMoE` with it could give a real wall-clock win on the actor's
training step (where eager attention dominates today, but the MoE block
is the next-largest cost). It is **not** wired up yet — gpt-oss uses a
clamped, GELU-approximating SwiGLU with a `(up + 1)` shift that no entry
in `sonicmoe.enums.ActivationType` matches, so a naive drop-in changes
training numerics.

What's in this directory:

| file | role |
| --- | --- |
| `sonic_moe_patch.py` | Reference implementation of gpt-oss's GLU + a stub `apply_sonic_moe_to_model()` that documents the integration plan and currently raises |
| `test_sonic_moe.py` | Forward-only probe: builds a sonic-moe MoE at gpt-oss-20b shapes, computes the activation gap vs the gpt-oss reference, benchmarks wall-clock |

### Step 1: install (opt-in)

```bash
INSTALL_SONIC_MOE=1 bash examples/gpt_oss/install.sh
```

Adds `sonic-moe` to the venv. Hopper or Blackwell only; CUDA 12.9+ and
torch ≥ 2.7 (both already pulled in by `verl[sglang]`).

### Step 2: run the parity test (forward + backward)

```bash
python examples/gpt_oss/test_sonic_moe_fwd_bwd.py
python examples/gpt_oss/test_sonic_moe_fwd_bwd.py --tokens 8192   # bigger workload
python examples/gpt_oss/test_sonic_moe_fwd_bwd.py --skip-backward # forward only
```

This is the test that decides whether the integration is feasible at
all. It runs sonic-moe's grouped-GEMM kernel and a pure-pytorch reference
on the *same* expert weights using vanilla SwiGLU on both sides, then
checks:

- **Forward parity**: max |sonic_out - ref_out| within `--atol-fwd`
  (default 5e-2 for bf16). If this fails, the kernel and reference are
  computing different things and the layout remap is wrong.
- **Backward parity**: max |Δ grad| on `gate_up_proj`, `down_proj`,
  `router_weight`, and the input `x`, all within `--atol-bwd` (default
  1e-1 for bf16 gradient accumulation). If this fails, sonic-moe's
  autograd path is incomplete and we can't use it for training.
- **Activation gap** (informational): max |vanilla SwiGLU - gpt-oss
  clamped GLU| on the same weights. Quantifies what the adapter has to
  reconstruct.

If both forward and backward PASS, the kernel is sound and the only
remaining work is the activation swap (Step 3 below). If either FAILs,
read the parameter dump it prints — the most likely cause is a layout
the script didn't try to remap.

### Step 2b: optional benchmark probe

```bash
RUN_SONIC_MOE_PROBE=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
# or standalone
python examples/gpt_oss/test_sonic_moe.py --tokens 8192
```

The benchmark probe reports:

- The **numerical gap** between sonic-moe's vanilla SwiGLU output and
  gpt-oss's clamped + `(up+1)` GLU at the same expert weights. If the
  max abs diff exceeds the `--atol` (default 5e-2 for bf16), the
  adapter must replace the activation before this can be used in
  training. Expected outcome: gap is large, because the activations
  really are different.
- A **wall-clock comparison** against a naive eager reference. The
  reference is slow (per-expert Python loop), so the speedup ratio it
  reports is an upper bound, not a real-vs-real number — useful only
  to sanity-check that sonic-moe runs at all on this hardware.

### Step 3 (TODO): finish the adapter

`USE_SONIC_MOE=1` in the launcher currently hard-fails. The integration
plan lives in the `sonic_moe_patch.py` module docstring. Summary:

1. Bypass `sonicmoe.MoE` (which bakes in SwiGLU) and compose
   `sonicmoe.functional._up_projection_forward` (with a no-op
   activation, if sonic-moe exposes one) + the gpt-oss clamped GLU in
   PyTorch + `_down_projection_forward`.
2. Map HF state-dict tensors (`gate_up_proj`, `down_proj`,
   `router.weight`, `router.bias`) into sonic-moe's expected layout.
   Note: gpt-oss-20b's `intermediate_size=2880` is not a power of two
   — the chosen layout has to handle that or pad.
3. Walk `model.model.layers[i].mlp` and replace the experts before
   FSDP wrap (post-wrap is much harder).
4. Verify forward parity at `atol=1e-2` against HF `GptOssExperts`,
   then verify backward by comparing parameter gradients on the same
   loss.

Until step 3 lands, the launcher refuses `USE_SONIC_MOE=1` so nobody
silently corrupts a training run.

## Per-node cluster metrics in wandb

verl's wandb run only logs system metrics from rank 0. To get GPU/CPU/mem
per node across the whole Ray cluster (matching what Ray dashboard shows),
run the sidecar in a second shell **after** the trainer has started Ray:

```bash
python examples/gpt_oss/wandb_ray_metrics.py \
    --project verl_gpt_oss_20b \
    --run-name gpt_oss_20b_grpo_gsm8k \
    --interval 15
```

It scrapes `/tmp/ray/session_latest/metrics/prometheus/prom_metrics_service_discovery.json`
for the cluster's per-node `/metrics` endpoints and forwards them to a
parallel wandb run named `<run-name>-cluster`. Stop with Ctrl-C when
training finishes; safe to start/stop repeatedly. No verl changes needed.

## References

- Upstream example: `examples/grpo_trainer/run_gptoss_20b.sh`
- v0.7 blog: `docs/blog/v0.7.md` (notes Megatron gpt-oss support)
- Issue #2930 (initial OpenAI OSS support request)
- Issue #3794 (agentic RL on gpt-oss)
- PR #3865 (tool-agent fix for gpt-oss)
- Issue #3894 (high actor/rollout pearson correlation)
- PR #4323 (Megatron gpt-oss support, BSHD requirement)
- PR #4750 (MFU compute support)
- PR #5131 (vllm gpt-oss encoding)
