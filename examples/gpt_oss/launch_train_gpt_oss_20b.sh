#!/bin/bash
# Launch GRPO training on openai/gpt-oss-20b using verl's FSDP actor + sglang rollout.
#
# This recipe is consolidated from the existing example at
# examples/grpo_trainer/run_gptoss_20b.sh and the discussion in upstream issues
# #2930, #3794, #3865, #3894 and PRs #4323, #4750, #5131.
#
# Caveats baked in:
#   * gpt-oss ships in MXFP4 -> we dequantize once to bf16 (see prepare_model.py).
#   * gpt-oss kernels assume eager attention; verl reads attn_implementation from
#     the saved config so we stamp it before saving.
#   * MoE training is unstable when train_batch_size != ppo_mini_batch_size;
#     keep them equal (issue #3894 noted high actor/rollout pearson_corr but the
#     equal-batch recipe is the recommended starting point).
#   * sglang + triton attention backend is the supported rollout combination.
#   * load_format=safetensors is required so weight transfer works after the
#     mxfp4->bf16 dequantization.
#
# Usage:
#   bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
#
# Override any tunable via environment variables, e.g.:
#   N_GPUS_PER_NODE=4 TRAIN_BATCH_SIZE=128 bash examples/gpt_oss/launch_train_gpt_oss_20b.sh
set -euxo pipefail

# ---- knobs ----------------------------------------------------------------
MODEL_ID=${MODEL_ID:-openai/gpt-oss-20b}
MODEL_DIR=${MODEL_DIR:-$HOME/models/gpt-oss-20b-bf16}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
PROJECT_NAME=${PROJECT_NAME:-verl_gpt_oss_20b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gpt_oss_20b_grpo_gsm8k}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-'["console","wandb"]'}

# ---- 0. dequantize weights -----------------------------------------------
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "[launch] dequantizing ${MODEL_ID} -> ${MODEL_DIR}"
    python3 "$(dirname "$0")/prepare_model.py" \
        --model-id "${MODEL_ID}" \
        --output-dir "${MODEL_DIR}"
fi

# ---- 1. preprocess gsm8k --------------------------------------------------
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "[launch] preprocessing gsm8k -> ${DATA_DIR}"
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "${DATA_DIR}"
fi

# ---- 2. launch training ---------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.reasoning_effort="${REASONING_EFFORT}" \
    actor_rollout_ref.model.path="${MODEL_DIR}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" "$@"
