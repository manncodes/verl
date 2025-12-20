#!/bin/bash
# ==============================================================================
# Cascaded RL Training Script
# ==============================================================================
# This script runs multi-task domain-wise RL training following Nemotron-Cascade.
#
# Usage:
#   ./run_cascaded_rl.sh [OPTIONS]
#
# Options:
#   --model PATH      Base model path (default: Qwen/Qwen3-8B)
#   --data_dir DIR    Directory containing processed data (default: ./data)
#   --output_dir DIR  Output directory for checkpoints (default: ./output)
#   --stages STAGES   Comma-separated list of stages to run (default: all)
#   --gpus NUM        Number of GPUs to use (default: 8)
#   --resume STAGE    Resume from a specific stage
#
# Examples:
#   # Run all stages
#   ./run_cascaded_rl.sh --model Qwen/Qwen3-8B --data_dir ./processed_data
#
#   # Run only math and code stages
#   ./run_cascaded_rl.sh --stages math_rl,code_rl
#
#   # Resume from code_rl stage
#   ./run_cascaded_rl.sh --resume code_rl
# ==============================================================================

set -e

# Default values
MODEL_PATH="Qwen/Qwen3-8B"
DATA_DIR="./data"
OUTPUT_DIR="./output"
STAGES="all"
NUM_GPUS=8
RESUME_STAGE=""
CONFIG_FILE="config/cascaded_rl_config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --stages)
            STAGES="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME_STAGE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Cascaded RL Training"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Stages: $STAGES"
echo "GPUs: $NUM_GPUS"
echo "Resume from: ${RESUME_STAGE:-none}"
echo "============================================================"

# Build stage configuration overrides
STAGE_OVERRIDES=""
if [[ "$STAGES" != "all" ]]; then
    # Filter stages
    IFS=',' read -ra STAGE_LIST <<< "$STAGES"
    echo "Running selected stages: ${STAGE_LIST[*]}"
fi

# Build resume override
RESUME_OVERRIDE=""
if [[ -n "$RESUME_STAGE" ]]; then
    RESUME_OVERRIDE="cascade.resume_from_stage=$RESUME_STAGE"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=cascaded_rl

# Navigate to recipe directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run training
echo "Starting Cascaded RL training..."
python main_cascaded_rl.py \
    --config-path=config \
    --config-name=cascaded_rl_config \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    cascade.checkpoint_dir="$OUTPUT_DIR/checkpoints" \
    trainer.default_local_dir="$OUTPUT_DIR/trainer_checkpoints" \
    trainer.project_name=cascaded_rl \
    trainer.experiment_name="${MODEL_PATH##*/}_cascade" \
    resource_pool.spec.default="[$NUM_GPUS]" \
    ${RESUME_OVERRIDE:+$RESUME_OVERRIDE} \
    "$@"

echo "============================================================"
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "============================================================"
