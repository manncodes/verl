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
| `check_gpt_oss_fwd_bwd.py` | Standalone forward/backward correctness check |
| `run_check.sh` | Wrapper around the check that also dequantizes if needed |
| `launch_train_gpt_oss_20b.sh` | GRPO training on GSM8K with FSDP + sglang |

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

Run standalone:

```bash
python examples/gpt_oss/test_attention_sinks.py --model-dir ~/models/gpt-oss-20b-bf16
```

Skip the auto-run: `SKIP_SINKS_TEST=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh`.

References for the bug class:
[unsloth #3142](https://github.com/unslothai/unsloth/issues/3142)
(SDPA bypass);
[vllm #22331](https://github.com/vllm-project/vllm/issues/22331),
[#22279](https://github.com/vllm-project/vllm/issues/22279) (FA2 errors);
[vllm #30919](https://github.com/vllm-project/vllm/issues/30919) (FlashInfer);
[NVIDIA/TE #2070](https://github.com/NVIDIA/TransformerEngine/issues/2070).

### Not enabled (Megatron-only)

- **Router replay (R2/R3)** is wired only for the Megatron actor today
  (`verl/workers/engine_workers.py:477` gates on `actor.strategy=="megatron"`).
  If you switch backends, see `examples/router_replay/` for the recipe.
- **Expert parallel (EP/ETP)** lives under `actor.megatron.*`; n/a for FSDP.

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
