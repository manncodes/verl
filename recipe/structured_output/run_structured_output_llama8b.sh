#!/bin/bash
# Structured Output RL Training with GRPO
#
# This script trains a Llama-3.1-8B-Instruct model to produce valid
# structured outputs (JSON) using GRPO with schema validation rewards.
#
# Prerequisites:
#   1. Prepare the dataset:
#      python -m recipe.structured_output.prepare_data --local_dir ~/data/structured_output
#
#   2. Install jsonschema for full schema validation:
#      pip install jsonschema
#
# Usage:
#   bash recipe/structured_output/run_structured_output_llama8b.sh
#
# For CRANE-style training (reasoning + constrained output):
#   Add these overrides:
#     structured_output.reward_mode=crane
#     actor_rollout_ref.rollout.guided_decoding.enable_reasoning=true

set -euo pipefail

# Data paths - update these to match your prepared data
DATA_DIR="${DATA_DIR:-$HOME/data/structured_output}"
TRAIN_FILE="${DATA_DIR}/structured_output_train.parquet"
VAL_FILE="${DATA_DIR}/structured_output_val.parquet"

# Model
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

python3 -m recipe.structured_output.main_structured_output \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.project_name=verl-structured-output \
    trainer.experiment_name=grpo_llama8b_json \
    trainer.total_epochs=10 \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    "$@"
