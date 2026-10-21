#!/usr/bin/env bash
# Example: matched ablation of calibrated importance sampling (CIS, arXiv 2609.32444)
# One arm per launch; run every arm with the same SEED values:
#
#   ARM=baseline SEED=1 bash examples/rollout_correction/run_qwen1_5_moe_a2_7b_cis_fsdp.sh
#   ARM=cis      SEED=1 bash examples/rollout_correction/run_qwen1_5_moe_a2_7b_cis_fsdp.sh
#
# The arms differ only in the detached weight on the PPO token loss, with k = pi_old / pi_rollout:
#   baseline  no weight (the train-rollout mismatch is still logged under rollout_corr/)
#   tis       min(k, 2)
#   cis       min(k, 1 + lambda * max(1 - pi_old, kappa)), pi_old recomputed by the trainer
# Checkpoint, data and order, seeds, token budget, optimizer and PPO settings are shared.
# The recipe follows the paper's GSM8K setup where verl allows it.
#
# Data:
#   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
#   python3 examples/data_preprocess/math_dataset.py --local_save_dir ~/data/math
#
# References:
#   - Rollout Correction Docs: https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md
#   - CIS reference implementation: https://github.com/kzhao5/CIS-RL

set -xeuo pipefail

# ==============================================================================
# Ablation arm
# ==============================================================================

ARM=${ARM:-cis}
SEED=${SEED:-1}
CIS_LAMBDA=${CIS_LAMBDA:-2.3}
CIS_KAPPA=${CIS_KAPPA:-0.005}

case "${ARM}" in
    baseline)
        CORRECTION=(algorithm.rollout_correction.rollout_is=null) ;;
    tis)
        CORRECTION=(algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=2.0) ;;
    cis)
        CORRECTION=(
            algorithm.rollout_correction.rollout_is=token
            algorithm.rollout_correction.rollout_is_cis_lambda=${CIS_LAMBDA}
            algorithm.rollout_correction.rollout_is_cis_kappa=${CIS_KAPPA}
        ) ;;
    *)
        echo "ARM must be baseline, tis or cis, got '${ARM}'" >&2
        exit 1 ;;
esac

# ==============================================================================
# Model and Data Configuration
# ==============================================================================

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen1.5-MoE-A2.7B-Chat"}
DATA_DIR=${DATA_DIR:-"${HOME}/data"}
TRAIN_FILES=${TRAIN_FILES:-"['${DATA_DIR}/gsm8k/train.parquet']"}
# One data source per held-out domain, reported as val-core/<data_source>/...
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/gsm8k/test.parquet','${DATA_DIR}/math/test.parquet']"}
OUTPUT_DIR=${OUTPUT_DIR:-"${HOME}/cis_ablation/${ARM}_seed${SEED}"}

max_prompt_length=1024
max_response_length=1024

# ==============================================================================
# Training Configuration (paper: 3 epochs of GSM8K = 87 optimizer steps)
# ==============================================================================

train_batch_size=256
ppo_mini_batch_size=256
total_epochs=3
learning_rate=3e-6
weight_decay=0.01

# ==============================================================================
# Launch Training
# ==============================================================================
########################### parameter arrays ###########################

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_batch_size}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.seed=${SEED}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "${CORRECTION[@]}"
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${learning_rate}
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.data_loader_seed=${SEED}
)

ROLLOUT=(
    # rollout_log_probs are recorded in every arm; the baseline only logs them
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.name=${INFER_BACKEND:-vllm}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.seed=${SEED}
    # CIS assumes rollout and trainer log-probs describe the same distribution: no top-p / top-k truncation
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="rollout_corr_cis_ablation"
    trainer.experiment_name="qwen1_5_moe_a2_7b_${ARM}_seed${SEED}"
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}
    trainer.nnodes=${NNODES:-1}
    trainer.total_epochs=${total_epochs}
    trainer.val_before_train=True
    trainer.test_freq=10
    trainer.save_freq=-1
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints"
    trainer.validation_data_dir="${OUTPUT_DIR}/validation"
)

EXTRA=(
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × fsdp extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"

echo "Training completed: arm=${ARM} seed=${SEED}"
echo "Compare arms on val-core/<data_source>/reward/mean@1 and the rollout_corr/ metrics,"
echo "see docs/algo/rollout_corr.md, section 'Calibrated importance sampling'."
