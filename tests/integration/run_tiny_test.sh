#!/bin/bash

set -e

echo "=============================================="
echo "  End-to-End Test: Tiny CustomSplitLLama"
echo "=============================================="
echo ""

# Configuration
MODEL_8B="${MODEL_8B:-/model-zoo/meta-llama_Llama-3.1-8B-Instruct/}"
MODEL_70B="${MODEL_70B:-/model-zoo/meta-llama_Llama-3.3-70B-Instruct/}"
TINY_MODEL_PATH="${TINY_MODEL_PATH:-/model-zoo/tiny-custom-split-llama-test}"
VERL_DIR="${VERL_DIR:-/home/jovyan/rl/verl}"

echo "Configuration:"
echo "  8B Model: $MODEL_8B"
echo "  70B Model: $MODEL_70B"
echo "  Tiny Model Output: $TINY_MODEL_PATH"
echo "  veRL Directory: $VERL_DIR"
echo ""

# Step 1: Create tiny model
echo "=============================================="
echo "Step 1: Creating Tiny CustomSplitLLama"
echo "=============================================="
echo ""

if [ -d "$TINY_MODEL_PATH" ]; then
    echo "⚠  Tiny model already exists at $TINY_MODEL_PATH"
    read -p "Recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TINY_MODEL_PATH"
    else
        echo "Using existing model"
    fi
fi

if [ ! -d "$TINY_MODEL_PATH" ]; then
    python3 create_tiny_custom_split.py \
        --path_8b "$MODEL_8B" \
        --path_70b "$MODEL_70B" \
        --output "$TINY_MODEL_PATH" \
        --num_layers_8 1 \
        --num_layers_70 1 \
        --mlp_adapter

    if [ $? -ne 0 ]; then
        echo "✗ Failed to create tiny model"
        exit 1
    fi
fi

echo ""
echo "✓ Tiny model ready at: $TINY_MODEL_PATH"
echo ""

# Step 2: Test model loading
echo "=============================================="
echo "Step 2: Testing Model Loading"
echo "=============================================="
echo ""

python3 test_tiny_model.py "$TINY_MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "✗ Model test failed"
    exit 1
fi

echo ""
echo "✓ Model tests passed"
echo ""

# Step 3: Copy model file to verl if needed
echo "=============================================="
echo "Step 3: Setting up veRL"
echo "=============================================="
echo ""

if [ ! -d "$VERL_DIR" ]; then
    echo "✗ veRL not found at $VERL_DIR"
    echo "  Please clone veRL first:"
    echo "  git clone https://github.com/manncodes/verl.git $VERL_DIR"
    exit 1
fi

# Check if custom_split_llama.py exists
CUSTOM_MODEL_FILE="$VERL_DIR/verl/models/transformers/custom_split_llama.py"
if [ ! -f "$CUSTOM_MODEL_FILE" ]; then
    echo "⚠  custom_split_llama.py not found in veRL"
    echo "  Please copy it to: $CUSTOM_MODEL_FILE"
    exit 1
fi

echo "✓ veRL setup verified"
echo ""

# Step 4: Prepare GSM8K data
echo "=============================================="
echo "Step 4: Preparing GSM8K Data"
echo "=============================================="
echo ""

DATA_DIR="/home/jovyan/data/gsm8k"
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating GSM8K data directory..."
    mkdir -p "$DATA_DIR"
fi

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
    echo "Downloading GSM8K data..."
    cd "$VERL_DIR"
    python3 -c "
from datasets import load_dataset
import os

# Load GSM8K dataset
dataset = load_dataset('openai/gsm8k', 'main')

# Save to parquet
os.makedirs('$DATA_DIR', exist_ok=True)
dataset['train'].to_parquet('$DATA_DIR/train.parquet')
dataset['test'].to_parquet('$DATA_DIR/test.parquet')

print('✓ GSM8K data downloaded')
"
fi

echo "✓ GSM8K data ready"
echo ""

# Step 5: Run training
echo "=============================================="
echo "Step 5: Running GRPO Training (TEST RUN)"
echo "=============================================="
echo ""

echo "Starting training with tiny model..."
echo "This is a short test run (2 epochs) to verify everything works"
echo ""

cd "$VERL_DIR"

# Set WandB mode
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-tiny-custom-split-test}"
export WANDB_NAME="${WANDB_NAME:-tiny-test-$(date +%Y%m%d_%H%M%S)}"

echo "WandB Configuration:"
echo "  Mode: $WANDB_MODE"
echo "  Project: $WANDB_PROJECT"
echo "  Run Name: $WANDB_NAME"
echo ""

# Run training with minimal configuration for testing
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="$TINY_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$WANDB_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="./checkpoints/tiny_test"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "=============================================="
    echo ""
    echo "Check WandB for results:"
    echo "  Project: $WANDB_PROJECT"
    echo "  Run: $WANDB_NAME"
    echo ""
    echo "If you see metrics logged, the test passed!"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "=============================================="
    exit $EXIT_CODE
fi
