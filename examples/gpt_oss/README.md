# gpt-oss launch scripts

End-to-end recipe for training **openai/gpt-oss-20b** with verl and verifying
the forward/backward pass before kicking off a real run.

## Files

| file | purpose |
| --- | --- |
| `prepare_model.py` | Dequantize the HF MXFP4 release to bf16 (one-time) |
| `check_gpt_oss_fwd_bwd.py` | Standalone forward/backward correctness check |
| `run_check.sh` | Wrapper around the check that also dequantizes if needed |
| `launch_train_gpt_oss_20b.sh` | GRPO training on GSM8K with FSDP + sglang |

## One-shot flow

A single command does dependency check -> dequantize -> preprocess gsm8k ->
forward/backward correctness check -> launch GRPO training:

```bash
bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

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
| `TRAIN_BATCH_SIZE == PPO_MINI_BATCH_SIZE` (`data.train_batch_size`, `actor.ppo_mini_batch_size`) | equal | MoE training diverges quickly when the two differ; upstream example keeps them equal. |
| `ENABLE_TIS=1` (`algorithm.rollout_correction.rollout_is`, `rollout.calculate_log_probs`) | on, token-level | Issue #3894 reports `rollout_actor_probs_pearson_corr ~ 0.5` from training/rollout drift — TIS is the supported mitigation. Set `TIS_LEVEL=sequence` for higher-variance unbiased weights. |
| `TIS_THRESHOLD` (`algorithm.rollout_correction.rollout_is_threshold`) | `2.0` | Per the upstream guide: 1.5–5.0 for token-TIS, 2.0–10.0 for sequence-TIS. |
| `USE_DYNAMIC_BSZ=False` (`actor.use_dynamic_bsz`) | off | Required by the gpt-oss megatron path (PR #4323) and safer for FSDP MoE since dynamic packing changes routing per step. |
| `USE_TORCH_COMPILE=False` (`actor.use_torch_compile`, `ref.use_torch_compile`) | off | torch.compile + MoE has been a recurring source of breakage; several fsdp/sglang examples hard-code it off. |
| `actor.use_kl_loss=True`, `kl_loss_type=low_var_kl`, `kl_loss_coef=0.001` | on | GRPO low-variance KL penalty; recommended by the upstream gpt-oss recipe. |
| `actor.entropy_coeff=0` | 0 | Extra entropy on top of GRPO can destabilise the router. |
| `actor.model.enable_gradient_checkpointing=True` | on | Memory pressure dominates with 32 experts × ~3.6B params each. |

Set `ENABLE_TIS=0` to drop back to vanilla GRPO (matches the existing
`examples/grpo_trainer/run_gptoss_20b.sh` baseline).

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
