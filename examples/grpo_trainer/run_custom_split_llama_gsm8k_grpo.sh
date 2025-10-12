#!/bin/bash
# Training script for Custom Split LLaMA on GSM8K with GRPO
#
# Usage:
#   bash run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir
#
# The model directory should contain:
#   - config.json (with Custom Split LLaMA configuration)
#   - References to 8B and 70B checkpoints in config.json
#
# Example config.json:
#   {
#     "architectures": ["CustomSplitLLamaForCausalLM"],
#     "path8b": "/path/to/llama-8b-model",
#     "path70b": "/path/to/llama-70b-model",
#     "num_layers_8": 32,
#     "num_layers_70": 8,
#     "mlp": false,
#     "vocab_size": 128256
#   }

set -e  # Exit on error
set -x  # Print commands

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model path (required - provide as first argument)
MODEL_PATH=${1:-""}

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH not provided!"
    echo "Usage: bash $0 /path/to/model/dir"
    echo ""
    echo "The model directory should contain config.json with Custom Split LLaMA configuration."
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_PATH"
    exit 1
fi

# Check if config.json exists
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Error: config.json not found in $MODEL_PATH"
    exit 1
fi

echo "Using model from: $MODEL_PATH"

# Data directory (default: ~/data/gsm8k)
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k"}

# Check if GSM8K data is preprocessed
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "Warning: GSM8K data not found in $DATA_DIR"
    echo "Preprocessing GSM8K dataset..."

    # Preprocess GSM8K data
    python3 ../data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"

    echo "GSM8K data preprocessed successfully!"
fi

# GPU configuration
N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-1e-6}
BATCH_SIZE=${BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-20}

# Parallelism configuration
TP_SIZE=${TP_SIZE:-4}  # Tensor parallelism size
PP_SIZE=${PP_SIZE:-1}  # Pipeline parallelism size

# GRPO configuration
N_RESPONSES=${N_RESPONSES:-5}  # Number of responses per prompt
KL_COEF=${KL_COEF:-0.001}

# Checkpointing
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-5}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints/custom_split_llama_gsm8k_grpo"}

# Logging
WANDB_PROJECT=${WANDB_PROJECT:-"custom_split_llama_grpo_gsm8k"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"custom_split_llama_grpo_$(date +%Y%m%d_%H%M%S)"}

# ============================================================================
# RUN TRAINING
# ============================================================================

export MODEL_PATH

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$EPOCHS \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    $@

echo ""
echo "============================================================================"
echo "Training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "============================================================================"
