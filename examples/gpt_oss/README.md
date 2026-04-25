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

## Typical flow

```bash
# 1. sanity check the model end-to-end (forward + backward, finite grads,
#    every expected submodule receives gradient).
bash examples/gpt_oss/run_check.sh

# 2. start training
bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

Override knobs via env vars, e.g.:

```bash
N_GPUS_PER_NODE=4 ROLLOUT_TP_SIZE=2 \
TRAIN_BATCH_SIZE=128 PPO_MICRO_BATCH_SIZE_PER_GPU=16 \
bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
```

## Why these defaults

- **MXFP4 -> bf16 once**: the FSDP/Megatron training paths cannot ingest the
  shipped MXFP4 weights; we dequantize via `Mxfp4Config(dequantize=True)` and
  stamp `attn_implementation=eager` on the saved config so training picks the
  supported attention path.
- **`train_batch_size == ppo_mini_batch_size`**: MoE training is unstable
  when these differ; the upstream example and issue #3894 keep them equal.
- **`rollout.name=sglang` + `attention_backend=triton` + `mode=async`**: the
  combination explicitly supported for gpt-oss in the upstream example.
- **`load_format=safetensors`**: required after dequantization so the rollout
  weight transfer path can ingest the bf16 shards.
- **`use_remove_padding=True`** plus FSDP: the FSDP path supports the THD
  packed format. Megatron currently needs BSHD (PR #4323) — switch to the
  Megatron recipe under `examples/grpo_trainer/run_qwen3moe-30b_megatron_*.sh`
  as a starting template if you need the Megatron backend.

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
