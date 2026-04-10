#!/bin/bash
# Off-policy prompt distillation using verl's FSDP SFT trainer.
#
# This script trains a student model on teacher-generated data (from prepare_data.py)
# using supervised fine-tuning. The student learns to replicate the teacher's behavior
# WITHOUT the system prompt.
#
# Usage:
#   bash run_sft.sh <nproc_per_node> <train_data_path> <model_path> <save_path> [other_configs...]
#
# Example:
#   bash run_sft.sh 4 \
#       ~/data/distill/train.parquet \
#       Qwen/Qwen2.5-7B-Instruct \
#       checkpoints/prompt_distill_sft \
#       model.lora_rank=32 \
#       optim.lr=1e-4 \
#       trainer.total_epochs=3

set -x

if [ "$#" -lt 4 ]; then
    echo "Usage: run_sft.sh <nproc_per_node> <train_data_path> <model_path> <save_path> [other_configs...]"
    echo ""
    echo "Arguments:"
    echo "  nproc_per_node   Number of GPUs per node"
    echo "  train_data_path  Path to teacher-generated training data (Parquet from prepare_data.py)"
    echo "  model_path       Path or HuggingFace model name for the student model"
    echo "  save_path        Directory to save checkpoints"
    echo ""
    echo "Example:"
    echo "  bash run_sft.sh 4 ~/data/distill/train.parquet Qwen/Qwen2.5-7B-Instruct checkpoints/distill"
    exit 1
fi

nproc_per_node=$1
train_data_path=$2
model_path=$3
save_path=$4

# Shift past the required args so $@ contains optional overrides
shift 4

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data_path \
    data.prompt_key=question \
    data.response_key=response \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    model.partial_pretrain=$model_path \
    model.lora_rank=32 \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-4 \
    optim.lr_scheduler=cosine \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=prompt-distillation \
    trainer.experiment_name=prompt-distill-sft \
    trainer.logger=console \
    trainer.total_epochs=3 \
    "$@"
