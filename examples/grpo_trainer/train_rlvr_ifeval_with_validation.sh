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

# RLVR-IFeval Training with Google IFEval Validation
#
# This script demonstrates the recommended approach for instruction following training:
# - Train on: RLVR-IFeval (allenai/RLVR-IFeval) - synthetic training data
# - Validate on: Google IFEval (google/IFEval) - original benchmark test set
#
# This prevents data leakage and provides accurate validation against the real benchmark.
#
# Usage:
#   bash examples/grpo_trainer/train_rlvr_ifeval_with_validation.sh
#
# Optional environment variables:
#   TRAIN_DATA_DIR: Directory for RLVR-IFeval training data (default: ~/data/rlvr_ifeval)
#   VAL_DATA_DIR: Directory for Google IFEval validation data (default: ~/data/ifeval_test)
#   MODEL_PATH: Base model to fine-tune (default: Qwen/Qwen2-7B-Instruct)
#   NUM_GPUS: Number of GPUs to use (default: 8)
#   NUM_EPOCHS: Number of training epochs (default: 20)

set -e  # Exit on error
set -x  # Print commands

# Configuration
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-"$HOME/data/rlvr_ifeval"}
VAL_DATA_DIR=${VAL_DATA_DIR:-"$HOME/data/ifeval_test"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-7B-Instruct"}
NUM_GPUS=${NUM_GPUS:-8}
NUM_EPOCHS=${NUM_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-512}
GROUP_SIZE=${GROUP_SIZE:-8}

echo "=========================================================="
echo "RLVR-IFeval Training with Google IFEval Validation"
echo "=========================================================="
echo "Training data: RLVR-IFeval → $TRAIN_DATA_DIR"
echo "Validation data: Google IFEval → $VAL_DATA_DIR"
echo "Model: $MODEL_PATH"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Group size (n): $GROUP_SIZE"
echo "=========================================================="
echo ""

# Step 1: Install dependencies
echo "Step 1/5: Checking dependencies..."
echo "=========================================================="

# Check if datasets library is installed
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing datasets library..."
    pip install datasets
fi

# Optional: Install official IFEval library for validation
echo "Installing official Google IFEval library for accurate validation..."
pip install git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval || {
    echo "Warning: Failed to install official library. Will use built-in verification."
}

echo ""

# Step 2: Preprocess RLVR-IFeval training data
echo "Step 2/5: Preprocessing RLVR-IFeval training dataset..."
echo "=========================================================="

if [ -f "$TRAIN_DATA_DIR/train.parquet" ] && [ -f "$TRAIN_DATA_DIR/val.parquet" ]; then
    echo "RLVR-IFeval dataset already exists at $TRAIN_DATA_DIR"
    read -p "Re-download and process? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping RLVR-IFeval preprocessing."
    else
        echo "Downloading and preprocessing RLVR-IFeval dataset..."
        python3 examples/data_preprocess/rlvr_ifeval.py \
            --local_save_dir "$TRAIN_DATA_DIR" \
            --add_instruction_prompt \
            --train_split_ratio 0.95
    fi
else
    echo "Downloading and preprocessing RLVR-IFeval dataset..."
    mkdir -p "$TRAIN_DATA_DIR"
    python3 examples/data_preprocess/rlvr_ifeval.py \
        --local_save_dir "$TRAIN_DATA_DIR" \
        --add_instruction_prompt \
        --train_split_ratio 0.95
fi

# Verify training data was created
if [ ! -f "$TRAIN_DATA_DIR/train.parquet" ]; then
    echo "Error: Failed to create training dataset!"
    exit 1
fi

echo "Training dataset ready: $TRAIN_DATA_DIR/train.parquet"
echo ""

# Step 3: Preprocess Google IFEval validation/test data
echo "Step 3/5: Preprocessing Google IFEval validation dataset..."
echo "=========================================================="

if [ -f "$VAL_DATA_DIR/test.parquet" ]; then
    echo "Google IFEval dataset already exists at $VAL_DATA_DIR"
    read -p "Re-download and process? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping Google IFEval preprocessing."
    else
        echo "Downloading and preprocessing Google IFEval dataset..."
        python3 examples/data_preprocess/ifeval.py \
            --local_save_dir "$VAL_DATA_DIR" \
            --add_instruction_prompt
    fi
else
    echo "Downloading and preprocessing Google IFEval dataset..."
    mkdir -p "$VAL_DATA_DIR"
    python3 examples/data_preprocess/ifeval.py \
        --local_save_dir "$VAL_DATA_DIR" \
        --add_instruction_prompt
fi

# Verify validation data was created
if [ ! -f "$VAL_DATA_DIR/test.parquet" ]; then
    echo "Error: Failed to create validation dataset!"
    exit 1
fi

echo "Validation dataset ready: $VAL_DATA_DIR/test.parquet"
echo ""

# Step 4: Verify GPU availability
echo "Step 4/5: Checking GPU availability..."
echo "=========================================================="

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

# Step 5: Run GRPO training
echo "Step 5/5: Starting GRPO training..."
echo "=========================================================="
echo "Training on: RLVR-IFeval"
echo "Validating on: Google IFEval (original benchmark)"
echo ""
echo "Training will begin in 5 seconds. Press Ctrl+C to cancel."
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA_DIR/train.parquet" \
    data.val_files="$VAL_DATA_DIR/test.parquet" \
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
    trainer.project_name='verl_grpo_rlvr_ifeval' \
    trainer.experiment_name="rlvr_ifeval_$(date +%Y%m%d_%H%M%S)" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=2 \
    trainer.total_epochs=$NUM_EPOCHS "$@"

echo ""
echo "=========================================================="
echo "Training completed!"
echo "=========================================================="
echo "Trained on: RLVR-IFeval (allenai synthetic data)"
echo "Validated on: Google IFEval (original benchmark)"
echo ""
echo "Check your wandb dashboard for:"
echo "  - Training metrics (on RLVR-IFeval)"
echo "  - Validation metrics (on Google IFEval test set)"
echo ""
echo "Model checkpoints are saved in the output directory."
echo "=========================================================="
