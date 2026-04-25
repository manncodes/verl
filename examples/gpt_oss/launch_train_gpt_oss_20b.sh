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
# Skip stages with:
#     SKIP_CHECK=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # no fwd/bwd check
#     SKIP_TRAIN=1 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh   # check only
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

# ---- knobs ----------------------------------------------------------------
MODEL_ID=${MODEL_ID:-openai/gpt-oss-20b}
MODEL_DIR=${MODEL_DIR:-$HOME/models/gpt-oss-20b-bf16}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
PROJECT_NAME=${PROJECT_NAME:-verl_gpt_oss_20b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gpt_oss_20b_grpo_gsm8k}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-'["console","wandb"]'}

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

SKIP_CHECK=${SKIP_CHECK:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
CHECK_SEQ_LEN=${CHECK_SEQ_LEN:-64}
CHECK_BATCH_SIZE=${CHECK_BATCH_SIZE:-1}

PYTHON=${PYTHON:-python3}

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

# ---- 2. forward/backward correctness check (default: on) ------------------
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

"${PYTHON}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "${TIS_ARGS[@]}" \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.reasoning_effort="${REASONING_EFFORT}" \
    actor_rollout_ref.model.path="${MODEL_DIR}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.calculate_log_probs="${CALC_LOGP}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
