#!/bin/bash
# One-shot launcher: dequantize -> preprocess -> forward/backward check -> train.
#
# Run from the repo root:
#     bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
#
# Override any tunable via environment variables, e.g.:
#     N_GPUS_PER_NODE=4 TRAIN_BATCH_SIZE=128 \
#         bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
#
# Skip / opt-in stages with:
#     SKIP_SINKS_TEST=1   bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # no sinks test
#     SKIP_R3_TEST=1      bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # no R3 capability check
#     RUN_ROLLOUT_TEST=1  bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # heavy: launches sglang/vllm
#     RUN_SONIC_MOE_PROBE=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh # benchmark sonic-moe vs gpt-oss GLU
#     SKIP_CHECK=1        bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # no fwd/bwd check
#     SKIP_TRAIN=1        bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # checks only
#
# Speedup presets:
#     FAST_PRESET=1       bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # all known-safe speedups on
#     ENABLE_BYPASS_MODE=1 ...                                                # skip the third actor forward (~3%)
#     ULYSSES_SP_SIZE=2   ...                                                 # 4x attention compute reduction
#     PARAM_OFFLOAD=False OPTIMIZER_OFFLOAD=False ACTIVATION_OFFLOAD=False ...  # 5-10x update_actor
#
# Recipe consolidated from examples/grpo_trainer/run_gptoss_20b.sh and the
# upstream issues/PRs noted in examples/gpt_oss/README.md. Caveats baked in:
#   * gpt-oss ships in MXFP4 -> we dequantize once to bf16.
#   * gpt-oss kernels assume eager attention; saved config carries that flag.
#   * Keep train_batch_size == ppo_mini_batch_size for MoE training stability.
#   * sglang + triton attention backend is the supported rollout combination.
#   * load_format=safetensors is required after the mxfp4->bf16 dequantization.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${REPO_ROOT}"

log() { printf '[launch] %s\n' "$*"; }

# ---- model / data --------------------------------------------------------
MODEL_ID=${MODEL_ID:-openai/gpt-oss-20b}
MODEL_DIR=${MODEL_DIR:-/model/Huggingface/openai/gpt-oss-20b-bf16}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
PROJECT_NAME=${PROJECT_NAME:-verl_gpt_oss_20b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gpt_oss_20b_grpo_gsm8k}

# ---- topology ------------------------------------------------------------
# Defaults assume one 8 x H100 80GB node. Set NNODES=N to scale out.
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TOTAL_GPUS=$((NNODES * N_GPUS_PER_NODE))

# ---- batch-size scaling rule ---------------------------------------------
# What scales linearly with NNODES (set per-node, multiplied here):
#     - TRAIN_BATCH_SIZE        (global #prompts per training step)
#     - PPO_MINI_BATCH_SIZE     (= TRAIN_BATCH_SIZE for MoE stability)
# What stays per-GPU (no scaling):
#     - PPO_MICRO_BATCH_SIZE_PER_GPU
#     - log_prob_micro_batch_size_per_gpu (actor + ref)
# What is independent of NNODES:
#     - ROLLOUT_TP_SIZE         (decided by model size, not node count)
#     - ROLLOUT_N               (algorithmic, # generations per prompt)
#     - max_prompt/response_length, KL/loss settings
#
# 1 H100 node default: 32 prompts/GPU * 8 GPUs = 256 train_batch_size.
# 2 nodes:             32 prompts/GPU * 16 GPUs = 512.
TRAIN_BATCH_SIZE_PER_NODE=${TRAIN_BATCH_SIZE_PER_NODE:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE_PER_NODE * NNODES))}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
# gpt-oss requires eager attention (only path that honours sinks). Eager
# materialises [bs, heads, seq, seq] in bf16: a single attention forward at
# seq=2560 with 64 heads costs ~3.4 GB per micro-batch element. The actor's
# backward pass also recomputes attention scores (gradient checkpointing),
# adding another peak. With all three FSDP offloads ON, GPU peaks at ~50 GB
# during backward (plus 26 GB sglang resident) on micro_batch=4 — leaving
# only 4 GB free, which fragments and OOMs at step 3. micro_batch=2 cuts
# the attention memory in half and reliably runs end-to-end on 8 x H100.
# Bump back to 4 if you have HBM headroom (e.g. ROLLOUT_GPU_MEM_UTIL=0.4).
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}

# ---- rollout / generation ------------------------------------------------
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
# Hybrid-engine memory budget. sglang reserves this fraction for KV cache,
# but the FSDP actor still occupies ~20 GB/GPU (params+grads+optimizer even
# with offload, eager attention activations). 0.7 was the upstream default
# and OOMs at sglang's resume_memory_occupation when the actor's training
# step finishes. 0.55 leaves enough headroom for the handoff on 8x H100.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
# gsm8k responses are typically <500 tokens. 8192 was the upstream default but
# wastes a lot of compute on padding and blows up eager attention's seq^2
# memory cost. 2048 leaves >4x headroom and matches the natural answer length.
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
REASONING_EFFORT=${REASONING_EFFORT:-medium}

# ---- training schedule ---------------------------------------------------
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-'["console","wandb"]'}

# ---- topology sanity checks ----------------------------------------------
if (( N_GPUS_PER_NODE % ROLLOUT_TP_SIZE != 0 )); then
    echo "[launch] ERROR: N_GPUS_PER_NODE=${N_GPUS_PER_NODE} not divisible by ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}" >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE % TOTAL_GPUS != 0 )); then
    echo "[launch] WARNING: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} not divisible by TOTAL_GPUS=${TOTAL_GPUS}; verl may pad or error" >&2
fi
if (( PPO_MINI_BATCH_SIZE != TRAIN_BATCH_SIZE )); then
    echo "[launch] WARNING: PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} != TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; MoE training is unstable when these differ (see issue #3894)" >&2
fi

# ---- MoE stability knobs --------------------------------------------------
# Background:
#   * Issue #3894 reports rollout_actor_probs_pearson_corr ~ 0.5 on gpt-oss-20B,
#     i.e. severe drift between training (FSDP bf16) and rollout (sglang) policies.
#     Truncated Importance Sampling (TIS) via algorithm.rollout_correction is
#     the supported mitigation (docs/algo/rollout_corr.md).
#   * MoE training is unstable when train_batch_size != ppo_mini_batch_size; we
#     keep them equal by default.
#   * use_dynamic_bsz=False is required for gpt-oss megatron (PR #4323) and is
#     safer for FSDP+MoE too because dynamic packing changes routing per step.
#   * torch.compile + MoE has a long history of breakage (see qwen3-fsdp NPU
#     examples). Default to off for both actor and ref.
#   * router_replay (R2/R3) is currently wired only for the megatron actor
#     (verl/workers/engine_workers.py:477); it is intentionally NOT enabled here.
#     Use examples/router_replay/* if you switch to the megatron backend.
# Tunables — all opt-out:
ENABLE_TIS=${ENABLE_TIS:-1}                     # set 0 to disable TIS
TIS_LEVEL=${TIS_LEVEL:-token}                   # token | sequence
TIS_THRESHOLD=${TIS_THRESHOLD:-2.0}             # 1.5–5.0 typical for token; 2.0–10.0 for sequence
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}   # actor + ref
# True is the better-perf default. The verl trainer calls left_right_2_no_padding
# unconditionally in _compute_old_log_prob (regardless of this flag), so its
# flash_attn.bert_padding dependency is not bypassable by flipping this off —
# we instead added a pure-torch fallback in verl/utils/attention_utils.py that
# kicks in when the flash-attn .so has an ABI mismatch with torch.
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

# ---- speedup preset (opt-in bundle) --------------------------------------
# FAST_PRESET=1 flips on the speedups that have been benchmarked to be safe
# on this stack:
#   * all three FSDP offloads OFF (5-10x update_actor speedup, costs HBM)
#   * ref policy param offload OFF (saves ~half the 63s ref forward)
#   * ULYSSES_SP_SIZE=2 (cuts eager-attention compute ~4x; needs even gpus)
#   * MAX_PROMPT_LENGTH=256, MAX_RESPONSE_LENGTH=1024 — gsm8k mean is ~137
#     prompt / ~350 response tokens with reasoning_effort=medium. The 512 /
#     2048 defaults were ~7-8x too big on the actual distribution; eager
#     attention is O(seq^2) so the wasted budget shows up as both wasted
#     compute AND wasted activation memory. Right-sizing lets us pack many
#     more sequences per micro-step.
#   * PPO_MICRO_BATCH_SIZE_PER_GPU bumped to 8 (up from 2/4): with the
#     1280-token per-seq budget above, 8 sequences pack to ~10k tokens per
#     GPU, well under verl's default ppo_max_token_len_per_gpu of 16k. This
#     drops accumulation steps from 16 to 4 per mini-batch on 8 GPUs.
#   * ROLLOUT_GPU_MEM_UTIL dropped to 0.5 (smaller responses need less KV
#     cache, freeing HBM for the now-resident actor).
#   * ENABLE_BYPASS_MODE=1 (skips the third actor forward each step; ~3% on
#     the profile we measured but free)
# Each of these is still overridable individually after the preset block, so
# you can FAST_PRESET=1 PPO_MICRO_BATCH_SIZE_PER_GPU=4 MAX_RESPONSE_LENGTH=2048
# if you OOM or expect longer responses. Watch the trainer's truncation rate;
# if it climbs, bump MAX_RESPONSE_LENGTH back up.
FAST_PRESET=${FAST_PRESET:-0}
if [ "${FAST_PRESET}" = "1" ]; then
    export PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
    export OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
    export ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-False}
    export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
    export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
    export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
    export ENABLE_BYPASS_MODE=${ENABLE_BYPASS_MODE:-1}
    if (( N_GPUS_PER_NODE % 2 == 0 )); then
        export ULYSSES_SP_SIZE=${ULYSSES_SP_SIZE:-2}
    fi
    echo "[launch] FAST_PRESET=1: offloads off, ulysses_sp=${ULYSSES_SP_SIZE:-1}, "\
"micro=${PPO_MICRO_BATCH_SIZE_PER_GPU}, prompt=${MAX_PROMPT_LENGTH}, "\
"response=${MAX_RESPONSE_LENGTH}, gpu_mem=${ROLLOUT_GPU_MEM_UTIL}, "\
"bypass_mode=${ENABLE_BYPASS_MODE}"
fi

# ---- offload knobs (perf/memory trade-off) -------------------------------
# Defaults are conservative because sglang doesn't fully release its memory
# when "asleep" — it keeps ~26 GB resident on every GPU (model weights +
# workspace) even between rollout steps. With all three offloads ON, GPU
# is already at 50/80 GB during the actor backward (the other 26 is sglang),
# so disabling any of them risks OOM in update_actor's backward pass.
#
# Trade-off: with offloads ON, update_actor is 8-14 min/step (MFU ~0.4%).
# With offloads OFF (and a less-resident sglang), update_actor drops to
# ~60-90s (MFU ~5-10%). On a 8x H100 80GB box with hybrid-engine sglang,
# OFF defaults OOM at step 3. If you have spare HBM (e.g. lower
# ROLLOUT_GPU_MEM_UTIL, or only one engine on the GPU), flip these to
# False via env to get a 5-10x speedup.
#
# CONSTRAINT: PARAM_OFFLOAD=False + OPTIMIZER_OFFLOAD=True is invalid.
# verl's FSDP engine init (workers/engine/fsdp/transformer_impl.py:187)
# calls `to(model=param_offload, optimizer=optimizer_offload, grad=param_offload)`
# and the base `to()` (workers/engine/base.py:180) asserts that if
# model=False then optimizer and grad must also be False. Practically this
# means: keep optimizer offload at most as aggressive as param offload.
# The recommended fast config is BOTH False (along with ACTIVATION_OFFLOAD=False).
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-True}

# Enforce the coupling above before we get a 5-min-into-init crash.
if [ "${PARAM_OFFLOAD}" = "False" ] && [ "${OPTIMIZER_OFFLOAD}" = "True" ]; then
    echo "[launch] ERROR: PARAM_OFFLOAD=False with OPTIMIZER_OFFLOAD=True is unsupported." >&2
    echo "        verl's FSDP engine cannot move just the optimizer to CPU while keeping" >&2
    echo "        params on GPU (workers/engine/base.py:180 asserts on this combo)." >&2
    echo "        Set OPTIMIZER_OFFLOAD=False as well, or leave PARAM_OFFLOAD=True." >&2
    exit 1
fi
# Ulysses sequence parallelism: shards the seq dim across N GPUs, making
# eager attention's seq^2 memory/compute scale as (seq/N)^2. Set to 2 if
# you have spare GPUs and want a ~4x attention speedup. Leave at 1 for
# the default 8x DP setup.
ULYSSES_SP_SIZE=${ULYSSES_SP_SIZE:-1}

# Ref policy: saves ~half the ref forward (63s on the measured profile)
# when its params live on GPU. Costs ~5GB / GPU. Default True keeps the
# previous behaviour; FAST_PRESET=1 flips it to False.
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}

# Bypass mode for rollout correction: reuses rollout log_probs as
# old_log_prob, skipping the third actor forward each step. Requires
# `calculate_log_probs=True` in rollout (which ENABLE_TIS already sets).
# Independent of the FAST_PRESET so you can opt in alone.
ENABLE_BYPASS_MODE=${ENABLE_BYPASS_MODE:-0}

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# sglang's torch_memory_saver (the thing that releases KV cache between
# rollout and training) is incompatible with expandable_segments and will
# crash sglang init with "TorchMemorySaver is disabled for the current
# process because expandable_segments is not supported yet". With
# PPO_MICRO_BATCH_SIZE_PER_GPU=2 the per-allocation size is small enough
# that fragmentation is rarely the OOM trigger; if you still hit one,
# lower micro_batch to 1 instead of touching the allocator config.

SKIP_SINKS_TEST=${SKIP_SINKS_TEST:-0}
SKIP_R3_TEST=${SKIP_R3_TEST:-0}
RUN_ROLLOUT_TEST=${RUN_ROLLOUT_TEST:-0}    # heavyweight; opt-in. Boots sglang/vllm.
RUN_SONIC_MOE_PROBE=${RUN_SONIC_MOE_PROBE:-0}  # benchmark + numeric gap vs gpt-oss GLU
USE_SONIC_MOE=${USE_SONIC_MOE:-0}          # placeholder — see sonic_moe_patch.py
SKIP_CHECK=${SKIP_CHECK:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
CHECK_SEQ_LEN=${CHECK_SEQ_LEN:-64}
CHECK_BATCH_SIZE=${CHECK_BATCH_SIZE:-1}

PYTHON=${PYTHON:-python3}

# Auto-pick the local .venv if the caller forgot to activate it. install.sh
# creates ${REPO_ROOT}/.venv by default; without this, the preflight runs
# against the system python and reports every dep as missing. Also prepend
# .venv/bin to PATH so subprocesses (notably flashinfer's JIT, which spawns
# `ninja`) find the venv-local binaries — without this, sglang crashes
# during CUDA graph capture with FileNotFoundError on ninja.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
    export VIRTUAL_ENV="${REPO_ROOT}/.venv"
    log "auto-using ${PYTHON} (run 'source .venv/bin/activate' to make it sticky)"
fi

# ---- preflight: required python packages ---------------------------------
log "checking python dependencies"
"${PYTHON}" - <<'PY'
import importlib, sys

required = {
    "torch": "torch",
    "transformers": "transformers (>=4.46 with Mxfp4Config)",
    "datasets": "datasets",
    "verl": "verl (pip install -e .)",
}
missing = []
for mod, hint in required.items():
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"  - {mod}: {hint} ({exc})")
if missing:
    print("Missing required packages:\n" + "\n".join(missing), file=sys.stderr)
    print("\nFix with: bash examples/gpt_oss/install.sh", file=sys.stderr)
    sys.exit(1)

# Mxfp4Config landed in transformers 4.46; bail early if older.
try:
    from transformers import Mxfp4Config  # noqa: F401
except Exception as exc:
    print(f"transformers is missing Mxfp4Config: {exc}", file=sys.stderr)
    print("Fix with: bash examples/gpt_oss/install.sh", file=sys.stderr)
    sys.exit(1)
PY

# ---- 0. dequantize weights -----------------------------------------------
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    log "dequantizing ${MODEL_ID} -> ${MODEL_DIR} (one-time, ~40GB download)"
    "${PYTHON}" "${HERE}/prepare_model.py" \
        --model-id "${MODEL_ID}" \
        --output-dir "${MODEL_DIR}"
else
    log "reusing existing bf16 checkpoint at ${MODEL_DIR}"
fi

# ---- 1. preprocess gsm8k --------------------------------------------------
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    log "preprocessing gsm8k -> ${DATA_DIR}"
    mkdir -p "${DATA_DIR}"
    "${PYTHON}" examples/data_preprocess/gsm8k.py --local_save_dir "${DATA_DIR}"
else
    log "reusing existing gsm8k parquet at ${DATA_DIR}"
fi

# ---- 2. attention sinks correctness test (default: on) -------------------
# gpt-oss attention layers carry learnable per-head sink scores. Most attention
# backends (SDPA, FA2, FlashInfer, TE) silently drop them — only eager / FA3 /
# TRTLLM honour them. This test fails fast if a config drift puts the actor on
# a sink-blind backend, which would silently corrupt training.
if [ "${SKIP_SINKS_TEST}" != "1" ]; then
    log "running attention sinks correctness test"
    "${PYTHON}" "${HERE}/test_attention_sinks.py" \
        --model-dir "${MODEL_DIR}" \
        --seq-len "${CHECK_SEQ_LEN}"
else
    log "SKIP_SINKS_TEST=1, skipping attention sinks test"
fi

# ---- 3. R3 (router replay) capability check (default: on) ----------------
# Verifies the routing-replay recording stack is present and the HF gpt-oss
# router is deterministic — preconditions for R3 to work end-to-end. Cheap
# (one HF forward pass), so we run it eagerly.
if [ "${SKIP_R3_TEST}" != "1" ]; then
    log "running R3 (router replay) capability check"
    "${PYTHON}" "${HERE}/test_router_replay_capability.py" \
        --model-dir "${MODEL_DIR}" \
        --seq-len "${CHECK_SEQ_LEN}"
else
    log "SKIP_R3_TEST=1, skipping R3 capability check"
fi

# ---- 4. rollout end-to-end test (opt-in, heavyweight) ---------------------
# Actually boots sglang (and vLLM if installed), generates ~16 tokens, then
# runs forward+backward on the (prompt + response) sequence via HF. Catches
# FA2/sinks bypass, harmony pre-warm regressions, tokenizer drift between
# rollout and actor. Off by default — each engine load is ~30s + ~40GB GPU.
if [ "${RUN_ROLLOUT_TEST}" = "1" ]; then
    log "running rollout end-to-end test (sglang + vLLM + backward)"
    "${PYTHON}" "${HERE}/test_rollout_e2e.py" \
        --model-dir "${MODEL_DIR}" \
        --tensor-parallel-size "${ROLLOUT_TP_SIZE}"
fi

# ---- 4b. sonic-moe probe (opt-in) -----------------------------------------
# Forward-only benchmark + activation-parity check at gpt-oss-20b shapes
# against vanilla SwiGLU. Cheap (one MoE forward, no full model). Sets up
# the data needed to decide whether finishing the sonic-moe adapter in
# examples/gpt_oss/sonic_moe_patch.py is worth the work.
if [ "${RUN_SONIC_MOE_PROBE}" = "1" ]; then
    log "running sonic-moe forward probe (Hopper/Blackwell only)"
    "${PYTHON}" "${HERE}/test_sonic_moe.py"
fi

if [ "${USE_SONIC_MOE}" = "1" ]; then
    # The adapter is intentionally a stub: gpt-oss's clamped + (up+1) GLU
    # has no parity in sonic-moe's baked-in SwiGLU, so flipping this on
    # without finishing the integration would silently corrupt training.
    # Fail fast here so nobody burns GPU-hours on the misconfiguration.
    echo "[launch] ERROR: USE_SONIC_MOE=1 is not yet supported." >&2
    echo "        See examples/gpt_oss/sonic_moe_patch.py for the integration plan and" >&2
    echo "        run RUN_SONIC_MOE_PROBE=1 first to confirm the win is worth the work." >&2
    exit 1
fi

# ---- 5. forward/backward correctness check (default: on) ------------------
if [ "${SKIP_CHECK}" != "1" ]; then
    log "running forward/backward correctness check"
    "${PYTHON}" "${HERE}/check_gpt_oss_fwd_bwd.py" \
        --model-dir "${MODEL_DIR}" \
        --seq-len "${CHECK_SEQ_LEN}" \
        --batch-size "${CHECK_BATCH_SIZE}"
else
    log "SKIP_CHECK=1, skipping correctness check"
fi

# ---- 3. launch training ---------------------------------------------------
if [ "${SKIP_TRAIN}" = "1" ]; then
    log "SKIP_TRAIN=1, exiting before training"
    exit 0
fi

log "topology: NNODES=${NNODES}  N_GPUS_PER_NODE=${N_GPUS_PER_NODE}  TOTAL_GPUS=${TOTAL_GPUS}"
log "batch:    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} (= ${TRAIN_BATCH_SIZE_PER_NODE} per-node x ${NNODES} nodes)"
log "          PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}  PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
log "rollout:  TP=${ROLLOUT_TP_SIZE}  DP=$((TOTAL_GPUS / ROLLOUT_TP_SIZE))  N=${ROLLOUT_N}  GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL}"
log "seq:      MAX_PROMPT=${MAX_PROMPT_LENGTH}  MAX_RESPONSE=${MAX_RESPONSE_LENGTH}  total_per_seq=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
log "          tokens_per_micro_gpu=$((PPO_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))) (worst-case, padded)"
log "          accum_steps_per_minibatch=$((PPO_MINI_BATCH_SIZE / (TOTAL_GPUS * PPO_MICRO_BATCH_SIZE_PER_GPU)))"
log "offload:  PARAM=${PARAM_OFFLOAD}  OPTIMIZER=${OPTIMIZER_OFFLOAD}  ACTIVATION=${ACTIVATION_OFFLOAD}  REF=${REF_PARAM_OFFLOAD}  ULYSSES_SP=${ULYSSES_SP_SIZE}"
log "speedups: FAST_PRESET=${FAST_PRESET}  ENABLE_BYPASS_MODE=${ENABLE_BYPASS_MODE}  ENABLE_TIS=${ENABLE_TIS}"
log "launching GRPO training"

# Build optional MoE-stability args (TIS via rollout correction).
TIS_ARGS=()
CALC_LOGP=False
if [ "${ENABLE_TIS}" = "1" ]; then
    log "enabling truncated importance sampling: level=${TIS_LEVEL} threshold=${TIS_THRESHOLD}"
    TIS_ARGS=(
        algorithm.rollout_correction.rollout_is="${TIS_LEVEL}"
        algorithm.rollout_correction.rollout_is_threshold="${TIS_THRESHOLD}"
    )
    CALC_LOGP=True
fi

# Optional: bypass-mode shortcut for the actor's old_log_prob phase.
# Reuses the rollout's log-probs (already calculated when CALC_LOGP=True)
# instead of running a third actor forward each PPO step.
BYPASS_ARGS=()
if [ "${ENABLE_BYPASS_MODE}" = "1" ]; then
    if [ "${CALC_LOGP}" != "True" ]; then
        echo "[launch] WARNING: ENABLE_BYPASS_MODE=1 but CALC_LOGP=False;" >&2
        echo "        bypass_mode requires calculate_log_probs=True in rollout." >&2
        echo "        Either set ENABLE_TIS=1 (which turns it on) or unset bypass." >&2
    else
        log "enabling rollout-correction bypass_mode (skips one actor forward per step)"
        BYPASS_ARGS=(
            algorithm.rollout_correction.bypass_mode=True
            algorithm.rollout_correction.loss_type=ppo_clip
        )
    fi
fi

"${PYTHON}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "${TIS_ARGS[@]}" \
    "${BYPASS_ARGS[@]}" \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.reasoning_effort="${REASONING_EFFORT}" \
    actor_rollout_ref.model.path="${MODEL_DIR}" \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload="${ACTIVATION_OFFLOAD}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.calculate_log_probs="${CALC_LOGP}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" "$@"
