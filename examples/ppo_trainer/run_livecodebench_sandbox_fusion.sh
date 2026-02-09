set -x

# LiveCodeBench + SandboxFusion PPO Training Example
#
# This script trains an LLM on LiveCodeBench competitive programming problems
# using SandboxFusion for secure, sandboxed code execution during reward computation.
#
# Prerequisites:
#   1. Preprocess the dataset:
#        python examples/data_preprocess/livecodebench.py \
#            --local_dir ~/data/livecodebench \
#            --start_date 2024-08-01 \
#            --end_date 2025-01-01
#
#   2. Deploy a SandboxFusion service:
#        See https://github.com/bytedance/SandboxFusion for setup instructions.
#        Update the sandbox_fusion.url below with your endpoint.

SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-"http://localhost:8080/run_code"}
DATA_DIR=${DATA_DIR:-"$HOME/data/livecodebench"}
MODEL=${MODEL:-"Qwen/Qwen2.5-Coder-7B-Instruct"}

python3 -m verl.trainer.main_ppo \
    reward_model.sandbox_fusion.url="$SANDBOX_FUSION_URL" \
    reward_model.sandbox_fusion.max_concurrent=128 \
    reward_model.sandbox_fusion.memory_limit_mb=1024 \
    algorithm.adv_estimator=gae \
    data.train_files=$DATA_DIR/test.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_livecodebench' \
    trainer.experiment_name='livecodebench_sandbox_fusion' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=1 \
    trainer.total_epochs=15 \
    reward_manager.name=prime $@
