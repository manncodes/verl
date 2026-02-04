#!/bin/bash
# =============================================================================
# GRPO Training for Structured Output (JSON Schema Compliance)
# =============================================================================
#
# Reward: Fine-grained JSON schema validation
#   - JSON Parsability (0.2): can the output be parsed as valid JSON?
#   - Schema Validity (0.3): does the JSON conform to the target schema?
#   - Field Coverage  (0.3): what fraction of required fields are correct type?
#   - Content Score   (0.2): does the content match expected values (if available)?
#
# Key insight: Schema validation provides a FINE-GRAINED reward, not binary.
#   Missing one field out of twelve should score higher than producing garbage.
#   This gradient signal is what makes GRPO converge for structural compliance.
#
# Dataset: nvidia/Nemotron-RL-instruction_following-structured_outputs
#   - 9.4K train / 512 validation examples
#   - Varying schema complexity (5-12 required fields)
#   - Document extraction + JSON schema formatting tasks
#
# References:
#   - Schema RL (SRL):          arxiv:2502.18878
#   - CRANE:                    arxiv:2502.09061
#   - Think Inside the JSON:    arxiv:2502.14905
#   - RL-Struct:                arxiv:2512.00319
#
# Prerequisites:
#   1. Prepare dataset:
#      python -m recipe.structured_output.prepare_data \
#          --local_dir ~/data/structured_output --train_repeat 3
#
#   2. Install jsonschema for full validation:
#      pip install jsonschema
#
# Usage:
#   bash recipe/structured_output/run_structured_output_grpo.sh [MODEL_PATH]
#
# For CRANE-style (reasoning + constrained output):
#   REWARD_MODE=crane bash recipe/structured_output/run_structured_output_grpo.sh
#
# =============================================================================

export LC_ALL=C
export LANG=C
set -euo pipefail

log_info()  { echo "[INFO]  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_warn()  { echo "[WARN]  $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

DEFAULT_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

log_info "Model: $MODEL_PATH"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_DIR="${DATA_DIR:-$HOME/data/structured_output}"

TRAIN_FILES="[${DATA_DIR}/structured_output_train.parquet]"
VAL_FILES=("${DATA_DIR}/structured_output_val.parquet")
VAL_FILES_STR=$(IFS=,; echo "[${VAL_FILES[*]}]")

# Structured output prompts include document text + schema instructions.
# Schemas can be large (up to 18K chars), so we need generous prompt length.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

# =============================================================================
# DISTRIBUTED CONFIGURATION
# =============================================================================

N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
TOTAL_GPUS=$((N_GPUS * NNODES))

TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
SP_SIZE="${SP_SIZE:-1}"

# =============================================================================
# CORE HYPERPARAMETERS
# =============================================================================

LEARNING_RATE="${LEARNING_RATE:-5e-7}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EPOCHS="${EPOCHS:-30}"

# =============================================================================
# BATCH SIZE CONFIGURATION
# =============================================================================

BASE_TRAIN_BATCH_SIZE="${BASE_TRAIN_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((BASE_TRAIN_BATCH_SIZE * NNODES))}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-$((8 * NNODES))}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"

# Schema compliance has high variance across different schema complexities
# (5-field vs 12-field schemas behave very differently).
# More samples per prompt -> better advantage estimation for heterogeneous schemas.
N_RESPONSES="${N_RESPONSES:-8}"

log_info "Batch: train=$TRAIN_BATCH_SIZE, mini=$MINI_BATCH_SIZE, micro=$MICRO_BATCH_SIZE, n=$N_RESPONSES"

# =============================================================================
# LOSS CONFIGURATION
# =============================================================================
# seq-mean-token-mean is appropriate here because:
#   1. Structured output length is driven by schema complexity, not verbosity
#   2. A 12-field JSON is naturally longer than a 5-field JSON
#   3. We don't want to bias toward shorter outputs that omit required fields
#   4. The field_coverage reward already penalizes missing fields

LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
LOSS_MODE="${LOSS_MODE:-gspo}"

# =============================================================================
# CLIPPING CONFIGURATION
# =============================================================================
# Symmetric clipping. Schema rewards are continuous [0, 1] with fine granularity
# (e.g., 7/12 fields correct = 0.583). Slightly wider clip than binary rewards
# to let the model learn from nuanced differences.

CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.18}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.18}"

log_info "Clipping: symmetric [$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH]"

# =============================================================================
# KL DIVERGENCE CONFIGURATION
# =============================================================================
# Moderate KL: we want the model to learn structural patterns from RL
# but not forget instruction-following ability from SFT.
# Too low = overfits to one schema pattern, too high = never learns structure.

USE_KL_LOSS="${USE_KL_LOSS:-True}"
KL_COEF="${KL_COEF:-0.01}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"

# =============================================================================
# ENTROPY REGULARIZATION
# =============================================================================
# Low but nonzero: JSON has limited structural diversity (unlike free-form text)
# but field values should remain diverse. We need enough exploration to discover
# correct field orderings, nesting patterns, and array structures.

ENTROPY_COEF="${ENTROPY_COEF:-0.001}"

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================
# Moderate temperature: need diversity for meaningful group comparison
# but structured output is more constrained than free-form generation.
# Lower than rubric/reasoning tasks because the format is rigid.

TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.8}"
TRAIN_TOP_P="${TRAIN_TOP_P:-0.95}"
TRAIN_TOP_K="${TRAIN_TOP_K:--1}"

# Validation: lower temp for consistent schema compliance measurement
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.3}"
VAL_TOP_P="${VAL_TOP_P:-0.9}"
VAL_TOP_K="${VAL_TOP_K:--1}"

# =============================================================================
# STRUCTURED OUTPUT REWARD CONFIGURATION
# =============================================================================
# Reward mode: "fine_grained" (default), "binary", or "crane"
#   - fine_grained: weighted sum of parsability + validity + coverage + content
#   - binary: 0 or 1 based on full schema compliance (stricter)
#   - crane: CRANE-style with reasoning section bonus
#
# The reward function is loaded as a custom reward function and used by the
# structured_output reward manager, which handles decoding, schema lookup,
# and CRANE-style output splitting.

REWARD_MODE="${REWARD_MODE:-fine_grained}"

# =============================================================================
# GUIDED DECODING (OPTIONAL)
# =============================================================================
# When enabled, uses vLLM's guided decoding to constrain output format.
# Options: "xgrammar" (fastest), "outlines", "lm-format-enforcer"
#
# WARNING: Pure constrained decoding during RL training can hurt exploration.
# The model needs to see what "wrong" JSON looks like to learn from it.
# Consider using guided decoding only for validation or final deployment,
# or use CRANE-style where reasoning is unconstrained but output is constrained.

GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-}"  # empty = disabled

# CRANE-style: reasoning + constrained output (only when REWARD_MODE=crane)
CRANE_REASONING_DELIMITER="${CRANE_REASONING_DELIMITER:-<answer>}"
CRANE_REASONING_END_DELIMITER="${CRANE_REASONING_END_DELIMITER:-</answer>}"
CRANE_REASONING_BONUS="${CRANE_REASONING_BONUS:-0.1}"

# =============================================================================
# MEMORY OPTIMIZATION
# =============================================================================

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
LIGER="${LIGER:-True}"

MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_BATCHED_TOKENS=$((MAX_TOKEN_LEN + 100))

FSDP_DTYPE="${FSDP_DTYPE:-bfloat16}"
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-True}"
FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-True}"

# =============================================================================
# LOGGING & CHECKPOINTING
# =============================================================================

WANDB_PROJECT="${WANDB_PROJECT:-structured-output-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-so_grpo_${REWARD_MODE}_lr${LEARNING_RATE}_kl${KL_COEF}_n${N_RESPONSES}_$(date +%m%d_%H%M)}"

LOG_DIR="${LOG_DIR:-/tmp/verl_logs/${WANDB_PROJECT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${WANDB_PROJECT}}"

SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-50}"

# =============================================================================
# ENVIRONMENT
# =============================================================================

export WANDB_PROJECT
export WEAVE_DISABLED=true
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export VERL_ENABLE_TRACKER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_V1=1

# =============================================================================
# LAUNCH
# =============================================================================

log_info "=========================================="
log_info "Starting Structured Output GRPO Training"
log_info "  Reward:       JSON Schema Validation (${REWARD_MODE})"
log_info "    parse=0.2, valid=0.3, coverage=0.3, content=0.2"
log_info "  Model:        $MODEL_PATH"
log_info "  Experiment:   $EXPERIMENT_NAME"
log_info "  GPUs:         $TOTAL_GPUS ($N_GPUS x $NNODES)"
log_info "  LR:           $LEARNING_RATE"
log_info "  KL coef:      $KL_COEF"
log_info "  Entropy:      $ENTROPY_COEF"
log_info "  Clip:         [$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH]"
log_info "  Loss agg:     $LOSS_AGG_MODE"
log_info "  Batch:        $TRAIN_BATCH_SIZE x $N_RESPONSES responses"
if [[ -n "$GUIDED_DECODING_BACKEND" ]]; then
    log_info "  Guided:       $GUIDED_DECODING_BACKEND"
fi
if [[ "$REWARD_MODE" == "crane" ]]; then
    log_info "  CRANE:        reasoning ${CRANE_REASONING_DELIMITER}...${CRANE_REASONING_END_DELIMITER} (bonus=${CRANE_REASONING_BONUS})"
fi
log_info "=========================================="

# Build extra args for guided decoding and CRANE mode
EXTRA_ARGS=()
if [[ -n "$GUIDED_DECODING_BACKEND" ]]; then
    EXTRA_ARGS+=(
        "actor_rollout_ref.rollout.guided_decoding.backend=$GUIDED_DECODING_BACKEND"
    )
fi
if [[ "$REWARD_MODE" == "crane" ]]; then
    EXTRA_ARGS+=(
        "reward_model.reward_kwargs.reasoning_delimiter=$CRANE_REASONING_DELIMITER"
        "reward_model.reward_kwargs.reasoning_end_delimiter=$CRANE_REASONING_END_DELIMITER"
        "reward_model.reward_kwargs.reasoning_bonus=$CRANE_REASONING_BONUS"
        "actor_rollout_ref.rollout.guided_decoding.enable_reasoning=True"
        "actor_rollout_ref.rollout.guided_decoding.reasoning_delimiter=$CRANE_REASONING_DELIMITER"
        "actor_rollout_ref.rollout.guided_decoding.reasoning_end_delimiter=$CRANE_REASONING_END_DELIMITER"
    )
fi

python3 -m recipe.structured_output.main_structured_output \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES_STR" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH//=/\\=}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_liger="$LIGER" \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="$LR_WARMUP_STEPS" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN" \
    actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
    actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEF" \
    actor_rollout_ref.actor.grad_clip="$GRAD_CLIP" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.checkpoint.save_contents="['model']" \
    \
    actor_rollout_ref.actor.fsdp_config.dtype="$FSDP_DTYPE" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.pipeline_model_parallel_size="$PP_SIZE" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.kl_ctrl.kl_coef="$KL_COEF" \
    algorithm.use_kl_in_reward="$USE_KL_IN_REWARD" \
    \
    reward_model.reward_manager=structured_output \
    reward_model.reward_kwargs.reward_mode="$REWARD_MODE" \
    \
    trainer.val_before_train=True \
    trainer.logger='["console", "wandb"]' \
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
    "${EXTRA_ARGS[@]}" \
    "${@:2}"

log_info "Training completed. Checkpoints: $CHECKPOINT_DIR"
