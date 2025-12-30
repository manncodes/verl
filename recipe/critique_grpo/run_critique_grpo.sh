#!/bin/bash
# Critique-GRPO Training Script
# Based on: https://arxiv.org/abs/2506.03106
#
# Usage:
#   bash recipe/critique_grpo/run_critique_grpo.sh
#
# This script trains a model using Critique-GRPO, which combines
# natural language feedback (critiques) with numerical rewards.

set -x

# Stop any existing Ray cluster
ray stop

# Set up paths
ROOT=$(dirname $(dirname $(dirname $(realpath $0))))
export PYTHONPATH=$ROOT:$PYTHONPATH

# Weights & Biases configuration (optional)
export PROJECT_NAME=critique-grpo
export WANDB_PROJECT="${WANDB_PROJECT:-critique_grpo_training}"
export WANDB_NAME="${WANDB_NAME:-critique_grpo_run}"
# Uncomment and set your API key to enable wandb logging
# export WANDB_API_KEY="your_api_key_here"

# Ray configuration
export RAY_BACKEND_LOG_LEVEL=debug
export HYDRA_FULL_ERROR=1

# Data configuration
# Set these to your dataset paths
DATA_DIR="${DATA_DIR:-/path/to/your/data}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"

# Model configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"

# Training hyperparameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6144}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

# PPO hyperparameters
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"

# Rollout configuration
ROLLOUT_N="${ROLLOUT_N:-8}"
N_PREFIX="${N_PREFIX:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"

# Critique configuration
# Options: "simple", "simple_gt", "text"
CRITIQUE_TYPE="${CRITIQUE_TYPE:-simple_gt}"

# Training configuration
TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
SAVE_FREQ="${SAVE_FREQ:-300}"
TEST_FREQ="${TEST_FREQ:-50}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# Checkpoint directory
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/${WANDB_NAME}}"
LOG_FILE="${LOG_FILE:-${WANDB_NAME}.log}"

# Print configuration
echo "=========================================="
echo "Critique-GRPO Training Configuration"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data Directory: $DATA_DIR"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "Learning Rate: $LEARNING_RATE"
echo "Rollout N: $ROLLOUT_N"
echo "N Prefix (refinements): $N_PREFIX"
echo "Critique Type: $CRITIQUE_TYPE"
echo "Temperature: $TEMPERATURE"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Save Frequency: $SAVE_FREQ"
echo "Test Frequency: $TEST_FREQ"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "=========================================="

# Compute max_num_batched_tokens
MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1000))

# Set visible GPUs (adjust as needed)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Run training
python3 -m recipe.critique_grpo.main_critique_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.critique_type=$CRITIQUE_TYPE \
    actor_rollout_ref.rollout.n_prefix=$N_PREFIX \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$WANDB_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE

echo "Training complete!"
echo "Logs saved to: $LOG_FILE"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
