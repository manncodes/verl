# Custom Split LLaMA Training Guide

Complete guide for training Custom Split LLaMA on GSM8K with GRPO (Group Relative Policy Optimization).

## Quick Start

```bash
# 1. Preprocess GSM8K data (if not already done)
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# 2. Run training
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir
```

## Prerequisites

### 1. Model Directory Setup

Your model directory should contain a `config.json` with the Custom Split LLaMA configuration:

```json
{
  "architectures": ["CustomSplitLLamaForCausalLM"],
  "path8b": "/path/to/llama-8b-model",
  "path70b": "/path/to/llama-70b-model",
  "num_layers_8": 32,
  "num_layers_70": 8,
  "mlp": false,
  "vocab_size": 128256
}
```

**Important:** The `path8b` and `path70b` should point to your 8B and 70B LLaMA model checkpoints.

### 2. GSM8K Data Preprocessing

The GSM8K dataset needs to be preprocessed into parquet format:

```bash
python3 examples/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k
```

This creates:
- `~/data/gsm8k/train.parquet` - Training set
- `~/data/gsm8k/test.parquet` - Test set

### 3. GPU Requirements

Recommended setup:
- **Minimum:** 4 x A100 (40GB) or 4 x A100 (80GB)
- **Recommended:** 8 x A100 (80GB)
- **Tensor Parallelism:** 4-8 (adjust based on your GPU count)

## Training Options

### Option 1: Using the Bash Script (Recommended)

```bash
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir
```

**Environment Variables:**

```bash
# Data directory (default: ~/data/gsm8k)
export DATA_DIR=~/data/gsm8k

# GPU configuration
export N_GPUS=8
export NNODES=1

# Hyperparameters
export LEARNING_RATE=1e-6
export BATCH_SIZE=256
export MICRO_BATCH_SIZE=16
export EPOCHS=20

# Parallelism
export TP_SIZE=4  # Tensor parallelism
export PP_SIZE=1  # Pipeline parallelism

# GRPO configuration
export N_RESPONSES=5  # Number of responses per prompt
export KL_COEF=0.001

# Checkpointing
export SAVE_FREQ=10
export TEST_FREQ=5
export CHECKPOINT_DIR=./checkpoints/custom_split_llama_gsm8k_grpo

# Logging
export WANDB_PROJECT=custom_split_llama_grpo_gsm8k
export EXPERIMENT_NAME=my_experiment

# Run training
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir
```

### Option 2: Using the YAML Config

```bash
# Edit the config file to set your model path
vim examples/grpo_trainer/custom_split_llama_gsm8k_grpo.yaml

# Run with config file
python3 -m verl.trainer.main_ppo \
    --config-path examples/grpo_trainer \
    --config-name custom_split_llama_gsm8k_grpo \
    actor_rollout_ref.model.path=/path/to/model/dir
```

### Option 3: Direct Python Command

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet \
    actor_rollout_ref.model.path=/path/to/model/dir \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=20
```

## Configuration Details

### Model Configuration

```yaml
actor_rollout_ref:
  model:
    path: /path/to/model/dir  # Your Custom Split LLaMA model
    use_remove_padding: True  # Enable packed inputs
    enable_gradient_checkpointing: True  # Save memory
```

### Actor (Policy) Configuration

```yaml
actor:
  optim:
    lr: 1e-6  # Learning rate
  ppo_mini_batch_size: 256
  ppo_micro_batch_size_per_gpu: 16
  use_kl_loss: True
  kl_loss_coef: 0.001
```

### Rollout (Generation) Configuration

```yaml
rollout:
  name: vllm  # Use vLLM for fast generation
  tensor_model_parallel_size: 4  # TP size
  gpu_memory_utilization: 0.85
  n: 5  # Number of responses for GRPO
```

### GRPO Algorithm Configuration

```yaml
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: False
  grpo:
    group_size: 5  # Should match rollout.n
```

## Training with LoRA

To enable LoRA fine-tuning, uncomment these lines in the YAML config:

```yaml
actor_rollout_ref:
  model:
    lora_rank: 64
    lora_alpha: 32
    lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

Or add to bash script:

```bash
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32
```

## Monitoring Training

### Weights & Biases (WandB)

Training metrics are logged to WandB by default:

```python
trainer.logger='["console","wandb"]'
trainer.project_name='custom_split_llama_grpo_gsm8k'
trainer.experiment_name='your_experiment_name'
```

View your runs at: https://wandb.ai

### Console Output

Training progress is printed to console:

```
Epoch 1/20
Step 10: loss=2.345, reward=0.678
Step 20: loss=2.123, reward=0.712
...
```

## Checkpointing

Checkpoints are saved every `SAVE_FREQ` epochs to `CHECKPOINT_DIR`:

```
checkpoints/custom_split_llama_gsm8k_grpo/
├── epoch_10/
│   ├── actor/
│   ├── critic/
│   └── optimizer/
├── epoch_20/
└── latest -> epoch_20/
```

## Resume Training

To resume from a checkpoint:

```bash
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir \
    trainer.resume_from=/path/to/checkpoint
```

## Memory Optimization Tips

### 1. Gradient Checkpointing

Always enabled by default:
```yaml
actor_rollout_ref.model.enable_gradient_checkpointing: True
```

### 2. FSDP Offloading

For limited GPU memory:
```yaml
actor_rollout_ref.actor.fsdp_config.param_offload: True
actor_rollout_ref.actor.fsdp_config.optimizer_offload: True
```

### 3. Reduce Batch Size

```bash
export BATCH_SIZE=128
export MICRO_BATCH_SIZE=8
```

### 4. Increase Tensor Parallelism

```bash
export TP_SIZE=8  # Use more GPUs for model parallelism
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `MICRO_BATCH_SIZE=8`
2. Enable offloading: `fsdp_config.param_offload=True`
3. Increase TP size: `TP_SIZE=8`
4. Reduce GPU memory utilization: `gpu_memory_utilization=0.75`

### Slow Training

1. Increase batch size: `MICRO_BATCH_SIZE=32`
2. Disable gradient checkpointing (if memory allows)
3. Use more GPUs: `N_GPUS=8`
4. Optimize data loading: `data.num_workers=8`

### Model Not Found

Ensure your model directory contains:
- `config.json` with Custom Split LLaMA configuration
- Valid paths to 8B and 70B checkpoints in `path8b` and `path70b`

### Data Not Found

Preprocess GSM8K data first:
```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

## Advanced Usage

### Multi-Node Training

```bash
# On master node (rank 0)
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export NNODES=4
export NODE_RANK=0

bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir

# On worker nodes (rank 1, 2, 3, ...)
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export NNODES=4
export NODE_RANK=1  # Change for each node

bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir
```

### Custom Hyperparameters

```bash
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.rollout.temperature=1.2 \
    algorithm.grpo.group_size=10 \
    actor_rollout_ref.rollout.n=10
```

## Expected Results

Training on GSM8K with GRPO typically achieves:

- **Initial Reward:** ~0.3-0.4
- **After 10 epochs:** ~0.6-0.7
- **After 20 epochs:** ~0.75-0.85

Test accuracy on GSM8K test set:
- **Baseline (no RL):** ~40-50%
- **After GRPO training:** ~70-80%

## References

- [veRL Documentation](https://deepwiki.com/volcengine/verl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [Custom Split LLaMA Integration](CUSTOM_SPLIT_LLAMA_INTEGRATION.md)

## Support

For issues or questions:
1. Check the [veRL GitHub Issues](https://github.com/volcengine/verl/issues)
2. Review the [Custom Split LLaMA Integration Guide](CUSTOM_SPLIT_LLAMA_INTEGRATION.md)
3. See the [Optimization Summary](OPTIMIZATION_SUMMARY.md) for efficiency tips
