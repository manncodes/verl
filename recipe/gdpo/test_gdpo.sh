#!/bin/bash
# Test script for GDPO

# This script tests GDPO with a simple GSM8K setup
# GDPO should work as a drop-in replacement for GRPO

set -e

# Run with GDPO advantage estimator
python -m verl.trainer.main_ppo \
    --config-name gdpo_trainer \
    --config-path recipe/gdpo/config \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.adv_estimator=gdpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name=gdpo_test \
    trainer.experiment_name=gsm8k_gdpo \
    trainer.n_gpus_per_node=1 \
    trainer.logger='["console"]' \
    $@
