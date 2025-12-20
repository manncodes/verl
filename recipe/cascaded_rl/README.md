# Cascaded RL Training for Multi-Task Domain-Wise Reinforcement Learning

This recipe implements **Cascaded RL** for training general-purpose reasoning models, based on the [Nemotron-Cascade paper](https://arxiv.org/abs/2512.13607) from NVIDIA.

## Overview

Cascaded RL orchestrates **sequential, domain-wise RL training** instead of blending heterogeneous prompts from different domains. This approach:

- Reduces engineering complexity
- Enables domain-specific hyperparameter tuning
- Rarely degrades prior domain performance
- Delivers state-of-the-art performance across benchmarks

### Training Pipeline

```
SFT → RLHF → IF-RL → Math-RL → Code-RL → SWE-RL
```

| Stage | Domain | Reward Function | Notes |
|-------|--------|-----------------|-------|
| RLHF | General alignment | Reward model (72B) | Boosts overall reasoning |
| IF-RL | Instruction following | Rule-based + RLHF | IFEval verification |
| Math-RL | Mathematical reasoning | Rule-based verifiers | Symbolic answer checking |
| Code-RL | Code generation | Execution-based | Unit test verification |
| SWE-RL | Software engineering | Hybrid lexical-semantic | Patch quality scoring |

## Key Features

- **GRPO Algorithm**: Group Relative Policy Optimization with on-policy training
- **No KL Penalty**: Simplified REINFORCE with group-normalized rewards
- **Domain-Specific Rewards**: Each stage has its own verifier/reward function
- **Checkpoint Transfer**: Seamless model transfer between stages
- **Dolci Dataset Support**: Compatible with AllenAI Dolci-Think-RL format

## Quick Start

### 1. Preprocess Data

```bash
# Download and preprocess Dolci-Think-RL dataset
python preprocess_dolci_dataset.py \
    --input_path allenai/Dolci-Think-RL-32B \
    --output_dir ./data \
    --split_by_domain
```

### 2. Run Training

```bash
# Run all stages
./run_cascaded_rl.sh \
    --model Qwen/Qwen3-8B \
    --data_dir ./data \
    --output_dir ./output \
    --gpus 8

# Run specific stages
./run_cascaded_rl.sh --stages math_rl,code_rl

# Resume from a stage
./run_cascaded_rl.sh --resume code_rl
```

### 3. Python API

```python
from recipe.cascaded_rl import (
    CascadedRLTrainer,
    CascadeRLConfig,
    StageConfig,
    create_default_cascade_stages,
)

# Create stages
stages = create_default_cascade_stages()

# Configure cascade
config = CascadeRLConfig(
    stages=stages,
    cascade_checkpoint_dir="./checkpoints",
)

# Create trainer
trainer = CascadedRLTrainer(
    cascade_config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_mapping,
    resource_pool_manager=resource_manager,
)

# Run training
trainer.fit()
```

## Configuration

### Stage Configuration

Each stage can be customized with:

```yaml
stages:
  - name: math_rl
    domain: math
    train_files: [data/math_train.parquet]
    val_files: [data/math_val.parquet]
    total_training_steps: 500
    train_batch_size: 128
    rollout_n: 8
    learning_rate: 2e-6
    temperature: 0.6
    top_p: 0.95
    max_new_tokens: 16384
```

### GRPO Settings (Nemotron-Cascade defaults)

```yaml
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
  norm_adv_by_std_in_grpo: true

actor_rollout_ref:
  actor:
    use_kl_loss: false
    kl_loss_coef: 0.0
  rollout:
    n: 8  # Group size for GRPO
    temperature: 0.6
    top_p: 0.95
```

## Data Format

The expected data format (parquet):

```python
{
    "data_source": str,        # Domain identifier
    "prompt": list[dict],      # Chat format: [{"role": "user", "content": "..."}]
    "ability": str,            # Task category: math, code, instruction_following, etc.
    "reward_model": {
        "style": "rule",       # "rule" or "model"
        "ground_truth": str,   # Expected answer for verification
    },
    "extra_info": dict,        # Additional metadata
}
```

## Reward Functions

### Math Reward
- Extracts answers from `\boxed{}` or "The answer is" patterns
- Symbolic comparison with normalization
- Partial credit for close answers

### Code Reward
- Extracts code from markdown blocks
- Executes with provided test cases
- Sandbox execution with timeout

### IF Reward
- Checks word/sentence count constraints
- Validates format requirements (JSON, markdown)
- Verifies inclusion/exclusion patterns

### SWE Reward
- Hybrid lexical-semantic patch scoring
- Execution-free verification
- Compares predicted vs target patches

## Hyperparameters

Following Nemotron-Cascade defaults:

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Rollouts per prompt | 8 |
| Learning rate | 2e-6 |
| Optimizer | AdamW |
| Temperature | 0.6 |
| Top-p | 0.95 |
| KL coefficient | 0 |
| Entropy coefficient | 0 |
| Training steps (RLHF) | ~800 |
| Max generation length | 64K tokens |

## References

- [Nemotron-Cascade Paper](https://arxiv.org/abs/2512.13607)
- [NVIDIA Nemotron-Cascade HuggingFace](https://huggingface.co/collections/nvidia/nemotron-cascade)
- [AllenAI Dolci-Think-RL](https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B)
- [AceReason-Nemotron](https://research.nvidia.com/labs/adlr/acemath_rl/)

## Citation

```bibtex
@article{wang2025nemotron,
  title={Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models},
  author={Wang, Boxin and Lee, Chankyu and Lee, Nayeon and others},
  journal={arXiv preprint arXiv:2512.13607},
  year={2025}
}
```
