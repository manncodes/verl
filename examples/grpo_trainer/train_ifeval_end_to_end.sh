#!/bin/bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-End IFEval GRPO Training Script
# This script handles everything: data download, preprocessing, and training
#
# Usage:
#   bash examples/grpo_trainer/train_ifeval_end_to_end.sh
#
# Optional environment variables:
#   DATA_DIR: Directory to store processed data (default: ~/data/ifeval)
#   MODEL_PATH: Base model to fine-tune (default: Qwen/Qwen2-7B-Instruct)
#   NUM_GPUS: Number of GPUs to use (default: 8)
#   NUM_EPOCHS: Number of training epochs (default: 10)

set -e  # Exit on error
set -x  # Print commands

# Configuration
DATA_DIR=${DATA_DIR:-"$HOME/data/ifeval"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-7B-Instruct"}
NUM_GPUS=${NUM_GPUS:-8}
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-512}
GROUP_SIZE=${GROUP_SIZE:-8}

echo "=================================================="
echo "IFEval GRPO Training - End-to-End Pipeline"
echo "=================================================="
echo "Data directory: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Group size: $GROUP_SIZE"
echo "=================================================="
echo ""

# Step 1: Install optional dependencies
echo "Step 1/4: Checking dependencies..."
echo "=================================================="

# Check if datasets library is installed
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing datasets library..."
    pip install datasets
fi

# Optional: Install official IFEval library for better accuracy
echo "Do you want to install the official Google IFEval library? (recommended)"
echo "This provides more accurate instruction verification."
read -p "Install? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing official IFEval library..."
    pip install git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval || {
        echo "Warning: Failed to install official library. Will use built-in verification."
    }
else
    echo "Skipping official library installation. Using built-in verification."
fi

echo ""

# Step 2: Download and preprocess data
echo "Step 2/4: Downloading and preprocessing IFEval dataset..."
echo "=================================================="

if [ -f "$DATA_DIR/train.parquet" ] && [ -f "$DATA_DIR/test.parquet" ]; then
    echo "Dataset already exists at $DATA_DIR"
    read -p "Re-download and process? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping data preprocessing."
    else
        echo "Downloading and preprocessing IFEval dataset..."
        python3 examples/data_preprocess/ifeval.py \
            --local_save_dir "$DATA_DIR" \
            --add_instruction_prompt
    fi
else
    echo "Downloading and preprocessing IFEval dataset..."
    mkdir -p "$DATA_DIR"
    python3 examples/data_preprocess/ifeval.py \
        --local_save_dir "$DATA_DIR" \
        --add_instruction_prompt
fi

# Verify data was created
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "Error: Failed to create dataset files!"
    exit 1
fi

echo "Dataset ready!"
echo "  Train: $DATA_DIR/train.parquet"
echo "  Test: $DATA_DIR/test.parquet"
echo ""

# Step 3: Verify GPU availability
echo "Step 3/4: Checking GPU availability..."
echo "=================================================="

if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Cannot verify GPU availability."
else
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Available GPUs: $AVAILABLE_GPUS"

    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
        echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available."
        echo "Adjusting NUM_GPUS to $AVAILABLE_GPUS"
        NUM_GPUS=$AVAILABLE_GPUS
    fi
fi

echo ""

# Step 4: Run GRPO training
echo "Step 4/4: Starting GRPO training..."
echo "=================================================="
echo "Training will begin in 5 seconds. Press Ctrl+C to cancel."
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_ifeval' \
    trainer.experiment_name="ifeval_$(date +%Y%m%d_%H%M%S)" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=2 \
    trainer.total_epochs=$NUM_EPOCHS "$@"

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
echo "Check your wandb dashboard for training metrics."
echo "Model checkpoints are saved in the output directory."
