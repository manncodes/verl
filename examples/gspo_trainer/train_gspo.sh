#!/bin/bash
# =============================================================================
# GSPO Training Script for Custom Split LLaMA on IFEval
# =============================================================================
#
# PERFORMANCE OPTIMIZATIONS APPLIED:
# 1. Increased GPU memory utilization (0.6 -> 0.85)
# 2. Disabled enforce_eager to enable CUDA graphs
# 3. Enabled prefix caching for shared prompt prefixes
# 4. Switched to async rollout mode (from sync)
# 5. Enabled free_cache_engine to free KV cache between phases
# 6. Increased max_num_batched_tokens for better throughput
# 7. Added torch compile for actor
# 8. Optimized token length calculations
#
# Usage:
#   bash train_gspo.sh [/path/to/model/dir] [additional_hydra_overrides...]
#
# Environment Variables (all optional, with defaults):
#   DATA_DIR          - Path to training data directory
#   N_GPUS            - Number of GPUs per node (default: 8)
#   NNODES            - Number of nodes (default: 1)
#   LEARNING_RATE     - Learning rate (default: 5e-7)
#   BASE_BATCH_SIZE   - Base batch size, scales with NNODES (default: 16)
#   BATCH_SIZE        - Override scaled batch size (default: BASE_BATCH_SIZE * NNODES)
#   MICRO_BATCH_SIZE  - Per-GPU micro batch size, constant (default: 8)
#   EPOCHS            - Total training epochs (default: 100)
#   WANDB_PROJECT     - Weights & Biases project name
#   CHECKPOINT_DIR    - Directory for saving checkpoints
#   ENABLE_FULLY_ASYNC - Set to "true" for 2x+ speedup with fully async training
#
# =============================================================================

# =============================================================================
# SHELL CONFIGURATION
# =============================================================================

export LC_ALL=C
export LANG=C

set -euo pipefail

# =============================================================================
# WANDB SETUP
# =============================================================================

export WANDB_PROJECT="${WANDB_PROJECT:-math-only}"
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
        echo "Usage: bash $0 /path/to/model/dir [additional_hydra_overrides...]"
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
log_info "Using model from: $MODEL_PATH"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_DIR="${DATA_DIR:-/fsxp2/qpn744/data/dolci_hints_4_curriculum}"

TRAIN_FILES="[${DATA_DIR}/math/train_easy.parquet,${DATA_DIR}/math/train_medium.parquet,${DATA_DIR}/math/train_hard.parquet]"

VAL_FILES=(
    "/fsxp2/qpn744/data/aime2025/train_ready_for_validation.parquet"
)

# Sequence length limits
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=32000

# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================

N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"

# Parallelism settings
TP_SIZE="${TP_SIZE:-4}"      # Tensor parallel size
PP_SIZE="${PP_SIZE:-1}"      # Pipeline parallel size
SP_SIZE="${SP_SIZE:-2}"      # Sequence parallel size (Ulysses)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

LEARNING_RATE="${LEARNING_RATE:-5e-7}"
EPOCHS="${EPOCHS:-5}"
GRAD_CLIP=1.0

# -----------------------------------------------------------------------------
# Batch Size Configuration (scales with NNODES)
# -----------------------------------------------------------------------------
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-4}"
BASE_MINI_PPO_BATCH_SIZE="${BASE_MINI_PPO_BATCH_SIZE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

compute_scaled_batch_sizes() {
    local nnodes="$1"
    BATCH_SIZE="${BATCH_SIZE:-$((BASE_BATCH_SIZE * nnodes))}"
    MINI_PPO_BATCH_SIZE="${MINI_PPO_BATCH_SIZE:-$((BASE_MINI_PPO_BATCH_SIZE * nnodes))}"
}

compute_scaled_batch_sizes "$NNODES"

log_info "Batch sizes (NNODES=$NNODES): BATCH_SIZE=$BATCH_SIZE, MINI_PPO=$MINI_PPO_BATCH_SIZE, MICRO=$MICRO_BATCH_SIZE"

# =============================================================================
# GSPO-SPECIFIC CONFIGURATION
# =============================================================================

N_RESPONSES="${N_RESPONSES:-8}"
KL_COEF="${KL_COEF:-0.0}"

# GSPO clip ratios (tighter than standard GRPO)
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# Loss configuration
LOSS_MODE="gspo"
LOSS_AGG_MODE="token-mean"

# =============================================================================
# MEMORY & PERFORMANCE CONFIGURATION - OPTIMIZED
# =============================================================================

LIGER="${LIGER:-True}"

# OPTIMIZATION 1: Increase GPU memory utilization from 0.6 to 0.85
# Best practices recommend 0.8-0.9 with offloading enabled
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# OPTIMIZATION 2: Use async rollout mode instead of sync (default)
# Async mode uses AsyncLLM with non-blocking generation via ZeroMQ
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"

# Token budget calculations
# Use 2x multiplier for actor (training), 3x for inference (more headroom)
ACTOR_MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
INFER_MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))

# OPTIMIZATION 3: Increase max_num_batched_tokens
# Rule of thumb: max(8192, max_prompt_length + max_response_length, max_model_len)
MAX_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1024))

# FSDP configuration
FSDP_DTYPE="float16"
FSDP_PARAM_OFFLOAD=True
FSDP_OPTIMIZER_OFFLOAD=True

# OPTIMIZATION 4: Enable torch compile for actor (20-30% speedup)
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

# Training rollout sampling
TRAIN_TEMPERATURE=1.0
TRAIN_TOP_P=1.0
TRAIN_TOP_K=-1

# Validation sampling (more deterministic)
VAL_TEMPERATURE=1.0
VAL_TOP_P=0.7
VAL_TOP_K=-1

# =============================================================================
# TIS SAMPLING CONFIGURATION
# =============================================================================

ROLLOUT_IS="${ROLLOUT_IS:-sequence}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
ROLLOUT_IS_BATCH_NORMALIZE="${ROLLOUT_IS_BATCH_NORMALIZE:-true}"

# =============================================================================
# LOGGING & CHECKPOINTING
# =============================================================================

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${WANDB_PROJECT}_$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="${LOG_DIR:-/exp/qpn744/${WANDB_PROJECT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/exp/qpn744/rl/checkpoints/${WANDB_PROJECT}}"

SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-3}"
LOG_VAL_GENERATIONS=100

# =============================================================================
# EXTERNAL SERVICES
# =============================================================================

SANDBOX_FUSION_URL="http://sandbox-fusion-code-rl-service.llm-pretraining.svc.cluster.local:8080/run_code"
REWARD_FUNCTION_PATH="/fsxp2/qpn744/rl/verl/verl/utils/reward_score/dolci_think_rl_v2.py"
REWARD_FUNCTION_NAME="compute_score_batch"

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

log_info "Starting GSPO training (OPTIMIZED)"
log_info "  Experiment: $EXPERIMENT_NAME"
log_info "  Batch size: $BATCH_SIZE"
log_info "  N responses: $N_RESPONSES"
log_info "  GPUs: $N_GPUS x $NNODES nodes"
log_info "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
log_info "  Rollout Mode: $ROLLOUT_MODE"
log_info "  Torch Compile: $USE_TORCH_COMPILE"

python3 -m verl.trainer.main_ppo \
    \
    `# === Algorithm Configuration ===` \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    algorithm.use_kl_in_reward=False \
    \
    `# === Data Configuration ===` \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES_STR" \
    data.train_batch_size="$BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
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
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_PPO_BATCH_SIZE" \
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
    actor_rollout_ref.actor.checkpoint.save_contents="['model']" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
    \
    `# === Actor FSDP Configuration ===` \
    actor_rollout_ref.actor.fsdp_config.dtype="$FSDP_DTYPE" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIMIZER_OFFLOAD" \
    \
    `# === Rollout Configuration - OPTIMIZED ===` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="$ROLLOUT_MODE" \
    +rollout.nnodes="$NNODES" \
    +rollout.n_gpus_per_node="$N_GPUS" \
    actor_rollout_ref.rollout.n="$N_RESPONSES" \
    actor_rollout_ref.rollout.temperature="$TRAIN_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TRAIN_TOP_P" \
    actor_rollout_ref.rollout.top_k="$TRAIN_TOP_K" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.pipeline_model_parallel_size="$PP_SIZE" \
    actor_rollout_ref.rollout.trace.backend=dummy \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    \
    `# === TIS Sampling Configuration ===` \
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
    reward_model.reward_manager=batch \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name="$REWARD_FUNCTION_NAME" \
    \
    `# === Trainer Configuration ===` \
    trainer.val_before_train=False \
    trainer.logger='["console", "wandb", "tensorboard"]' \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$EPOCHS" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.rollout_data_dir="$LOG_DIR" \
    \
    `# === Debug Configuration ===` \
    +ray_init.timeline_file="/fsxp/qpn744/rl/tmp/ray_timeline.json" \
    \
    `# === Additional Hydra Overrides ===` \
    "${@:2}"

# =============================================================================
# COMPLETION
# =============================================================================

log_info "Training completed!"
log_info "Checkpoints saved to: $CHECKPOINT_DIR"
