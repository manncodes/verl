#!/usr/bin/env bash
# Train a 7B model on how2everything with GRPO + rule-based heuristic reward.
#
# This is an ablation setup that does NOT require the How2Judge model.
# Useful for validating the data pipeline end-to-end and for comparison
# against the full GenRM setup (run_grpo_7b.sh).
#
# Prerequisites:
#   1. Preprocess data: python recipe/how2everything/data_preprocess.py
#   2. Set MODEL_PATH environment variable
#
# Reference: https://github.com/lilakk/how2everything
set -xeuo pipefail

project_name='How2Everything'
exp_name='GRPO-7B-RuleReward'

# Algorithm: GRPO (no critic)
adv_estimator=grpo

# No KL penalty
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# Sequence lengths
max_prompt_length=1024
max_response_length=2048

# GRPO group sampling
train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=32

# Sampling
temperature=0.8
top_p=0.95
top_k=-1

# Performance
offload=True
gen_tp=1

# Paths
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/how2everything/train.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/how2everything/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/verl/ckpts/how2everything/${exp_name}"}

# Ray cluster
NNODES=${NNODES:-1}

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/how2everything/config"

python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_PATH" \
    --config-name genrm_config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.enable=False \
    custom_reward_function.path=recipe/how2everything/reward_fn.py \
    custom_reward_function.name=compute_score_how2_rule \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=150 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
