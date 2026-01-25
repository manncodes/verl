#!/bin/bash
# =============================================================================
# GSPO Fully Async Training Script - 2x+ Speedup
# =============================================================================
#
# This script uses the fully_async_policy recipe which can provide 2x-2.7x
# speedup by completely decoupling the Trainer and Rollouter.
#
# KEY BENEFITS:
# - Overlapping generation and training (no pipeline bubbles)
# - Multi-step asynchronous training with freshness control
# - Partial rollout support (interrupt/resume during param sync)
# - Streaming sample production
#
# TRADE-OFF:
# - Slightly off-policy training (controlled via staleness_threshold)
# - Requires separate resource allocation for trainer and rollouter
#
# Usage:
#   bash train_gspo_fully_async.sh [/path/to/model/dir] [additional_overrides...]
#
# Key Environment Variables:
#   TRAINER_NNODES    - Nodes for trainer (default: half of total)
#   ROLLOUT_NNODES    - Nodes for rollouter (default: half of total)
#   STALENESS         - Max stale sample ratio (default: 0.5)
#
# =============================================================================

export LC_ALL=C
export LANG=C

set -euo pipefail

# =============================================================================
# IMPORTANT: CHANGE TO VERL ROOT DIRECTORY
# =============================================================================
# The fully_async_policy recipe uses relative hydra config paths that require
# running from the verl root directory. Auto-detect and change to it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find verl root directory (contains verl/ and recipe/ directories)
find_verl_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/verl" && -d "$dir/recipe" && -f "$dir/pyproject.toml" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

VERL_ROOT="${VERL_ROOT:-$(find_verl_root "$SCRIPT_DIR")}"
if [[ -z "$VERL_ROOT" || ! -d "$VERL_ROOT/recipe/fully_async_policy" ]]; then
    echo "[ERROR] Cannot find verl root directory. Set VERL_ROOT environment variable."
    exit 1
fi

echo "[INFO] Changing to verl root directory: $VERL_ROOT"
cd "$VERL_ROOT"

# =============================================================================
# WANDB SETUP
# =============================================================================

export WANDB_PROJECT="${WANDB_PROJECT:-math-only-async}"
export WEAVE_DISABLED=true

bash /exp/qpn744/wandb/wandb.sh

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

validate_model_path() {
    local model_path="$1"
    if [[ -z "$model_path" ]]; then
        log_error "MODEL_PATH not provided!"
        echo "Usage: bash $0 /path/to/model/dir [additional_overrides...]"
        exit 1
    fi
    if [[ ! -d "$model_path" ]]; then
        log_error "Model directory does not exist: $model_path"
        exit 1
    fi
    if [[ ! -f "$model_path/config.json" ]]; then
        log_error "config.json not found in $model_path"
        exit 1
    fi
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

DEFAULT_MODEL_PATH="/fsxp2/qpn744/rl/checkpoints//qpn744-rmrisk4-grpo-if/hf/global_step_250"
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"
validate_model_path "$MODEL_PATH"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_DIR="${DATA_DIR:-/fsxp2/qpn744/data/dolci_hints_4_curriculum}"
TRAIN_FILES="[${DATA_DIR}/math/train_easy.parquet,${DATA_DIR}/math/train_medium.parquet,${DATA_DIR}/math/train_hard.parquet]"
VAL_FILES=(
    "/fsxp2/qpn744/data/aime2025/train_ready_for_validation.parquet"
)

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=32000

# =============================================================================
# DISTRIBUTED CONFIGURATION - SPLIT BETWEEN TRAINER AND ROLLOUTER
# =============================================================================

N_GPUS="${N_GPUS:-8}"
TOTAL_NNODES="${NNODES:-32}"

# Split resources between trainer and rollouter
# Recommendation: Adjust based on idle_ratio metrics after initial runs
TRAINER_NNODES="${TRAINER_NNODES:-$((TOTAL_NNODES / 2))}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-$((TOTAL_NNODES / 2))}"

# Parallelism settings
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
SP_SIZE="${SP_SIZE:-2}"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

LEARNING_RATE="${LEARNING_RATE:-5e-7}"
GRAD_CLIP=1.0

# Batch sizes for fully async
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

# =============================================================================
# GSPO-SPECIFIC CONFIGURATION
# =============================================================================

N_RESPONSES="${N_RESPONSES:-8}"
KL_COEF="${KL_COEF:-0.0}"
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_MODE="gspo"
LOSS_AGG_MODE="token-mean"

# =============================================================================
# FULLY ASYNC CONFIGURATION
# =============================================================================

# Total rollout steps (equivalent to train_batch_size * steps in sync mode)
# Calculate: For 5 epochs with ~1000 prompts = 5000 total, with batch_size 4 = 1250 steps
# So total_rollout_steps = 4 * 1250 = 5000
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-5000}"

# How many ppo_mini_batch_size batches to fetch at once
# Higher = more stable training, lower = more streaming
REQUIRE_BATCHES="${REQUIRE_BATCHES:-4}"

# How many local updates before parameter sync
# Higher = more throughput but more off-policy
TRIGGER_PARAM_SYNC_STEP="${TRIGGER_PARAM_SYNC_STEP:-4}"

# Staleness threshold: controls max proportion of stale samples
# 0 = synchronous (no speedup)
# 0.5 = recommended balance of speed and accuracy
# 1.0 = equivalent to one_step_off_policy
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.5}"

# Enable partial rollout (interrupt/resume during param sync)
# Reduces waiting time for in-progress generations
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"

# Use log_probs from rollout (required for async correctness)
USE_ROLLOUT_LOG_PROBS="${USE_ROLLOUT_LOG_PROBS:-True}"

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
LIGER="${LIGER:-True}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"

# Token budget calculations
ACTOR_MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
INFER_MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))
MAX_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1024))

# FSDP configuration
FSDP_DTYPE="float16"
FSDP_PARAM_OFFLOAD=True
FSDP_OPTIMIZER_OFFLOAD=True

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

TRAIN_TEMPERATURE=1.0
TRAIN_TOP_P=1.0
TRAIN_TOP_K=-1
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.7
VAL_TOP_K=-1

# =============================================================================
# LOGGING & CHECKPOINTING
# =============================================================================

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${WANDB_PROJECT}_async_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/exp/qpn744/${WANDB_PROJECT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/exp/qpn744/rl/checkpoints/${WANDB_PROJECT}}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-20}"  # Higher for async to avoid slowing rollout
LOG_VAL_GENERATIONS=100

# =============================================================================
# EXTERNAL SERVICES
# =============================================================================

SANDBOX_FUSION_URL="http://sandbox-fusion-code-rl-service.llm-pretraining.svc.cluster.local:8080/run_code"
REWARD_FUNCTION_PATH="/fsxp2/qpn744/rl/verl/verl/utils/reward_score/dolci_think_rl_v2.py"
REWARD_FUNCTION_NAME="compute_score"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

export MODEL_PATH
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export VERL_ENABLE_TRACKER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_V1=1

# =============================================================================
# BUILD VALIDATION FILES ARRAY
# =============================================================================

VAL_FILES_STR=$(IFS=,; echo "[${VAL_FILES[*]}]")

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

log_info "Starting GSPO FULLY ASYNC training"
log_info "  Experiment: $EXPERIMENT_NAME"
log_info "  Model: $MODEL_PATH"
log_info "  Trainer nodes: $TRAINER_NNODES x $N_GPUS GPUs"
log_info "  Rollout nodes: $ROLLOUT_NNODES x $N_GPUS GPUs"
log_info "  Staleness threshold: $STALENESS_THRESHOLD"
log_info "  Partial rollout: $PARTIAL_ROLLOUT"
log_info "  Total rollout steps: $TOTAL_ROLLOUT_STEPS"

# Use the fully_async_policy recipe
python3 -m recipe.fully_async_policy.fully_async_main \
    \
    `# === Algorithm Configuration ===` \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    algorithm.use_kl_in_reward=False \
    \
    `# === Data Configuration ===` \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES_STR" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.return_raw_chat=True \
    \
    `# === Model Configuration ===` \
    actor_rollout_ref.model.path="${MODEL_PATH//=/\\=}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger="$LIGER" \
    \
    `# === Actor (Policy) Configuration ===` \
    actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.grad_clip="$GRAD_CLIP" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
    critic.strategy=fsdp2 \
    \
    `# === Actor FSDP Configuration ===` \
    actor_rollout_ref.actor.fsdp_config.dtype="$FSDP_DTYPE" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIMIZER_OFFLOAD" \
    \
    `# === Hybrid Engine (disabled for fully async) ===` \
    actor_rollout_ref.hybrid_engine=False \
    \
    `# === Rollout Configuration ===` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="$N_RESPONSES" \
    actor_rollout_ref.rollout.temperature="$TRAIN_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TRAIN_TOP_P" \
    actor_rollout_ref.rollout.top_k="$TRAIN_TOP_K" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    \
    `# === Reference Model Configuration ===` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    \
    `# === Sequence Parallelism ===` \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    \
    `# === Reward Model Configuration ===` \
    reward_model.sandbox_fusion.url="$SANDBOX_FUSION_URL" \
    reward_model.reward_manager=naive \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name="$REWARD_FUNCTION_NAME" \
    \
    `# === Trainer Configuration (separate resources) ===` \
    trainer.nnodes="$TRAINER_NNODES" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.logger='["console", "wandb", "tensorboard"]' \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.rollout_data_dir="$LOG_DIR" \
    \
    `# === Rollout Resource Configuration (separate resources) ===` \
    rollout.nnodes="$ROLLOUT_NNODES" \
    rollout.n_gpus_per_node="$N_GPUS" \
    rollout.total_rollout_steps="$TOTAL_ROLLOUT_STEPS" \
    rollout.test_freq="$TEST_FREQ" \
    \
    `# === Async Training Configuration ===` \
    async_training.require_batches="$REQUIRE_BATCHES" \
    async_training.trigger_parameter_sync_step="$TRIGGER_PARAM_SYNC_STEP" \
    async_training.staleness_threshold="$STALENESS_THRESHOLD" \
    async_training.partial_rollout="$PARTIAL_ROLLOUT" \
    async_training.use_rollout_log_probs="$USE_ROLLOUT_LOG_PROBS" \
    \
    `# === Ray Runtime Environment (propagate env vars to workers) ===` \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.HYDRA_FULL_ERROR="1"' \
    \
    `# === Additional Hydra Overrides ===` \
    "${@:2}"

# =============================================================================
# COMPLETION
# =============================================================================

log_info "Training completed!"
log_info "Checkpoints saved to: $CHECKPOINT_DIR"
log_info ""
log_info "Performance tips:"
log_info "  - Check trainer/idle_ratio and rollouter/idle_ratio in wandb"
log_info "  - If trainer idle is high, reduce TRAINER_NNODES and increase ROLLOUT_NNODES"
log_info "  - If rollouter idle is high, do the opposite"
log_info "  - Increase STALENESS_THRESHOLD for more speed (but more off-policy)"
