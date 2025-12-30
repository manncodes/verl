# Critique-GRPO Recipe

This recipe implements **Critique-GRPO** from the paper ["Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback"](https://arxiv.org/abs/2506.03106).

## Overview

Critique-GRPO enhances standard GRPO by combining:
1. **Numerical rewards** (accuracy scores)
2. **Natural language feedback** (critiques)

This dual-signal approach helps models learn from both what went wrong (via critiques) and how to improve (via refined solutions).

## How It Works

```
┌─────────────┐     ┌────────────┐     ┌───────────────┐     ┌─────────────┐
│   Problem   │ ──► │  Initial   │ ──► │   Critique    │ ──► │ Refinement  │
│   Prompt    │     │  Response  │     │  Generation   │     │  Generation │
└─────────────┘     └────────────┘     └───────────────┘     └─────────────┘
                          │                   │                     │
                          ▼                   ▼                     ▼
                    [Reward: 0/1]       [Natural Lang]        [Reward: 0/1]
                          │                   │                     │
                          └───────────────────┴─────────────────────┘
                                              ▼
                                    ┌─────────────────┐
                                    │  GRPO Training  │
                                    │  (Off-policy +  │
                                    │   On-policy)    │
                                    └─────────────────┘
```

### Training Pipeline

1. **Initial Generation**: Generate N responses per prompt
2. **Scoring**: Compute accuracy for each response
3. **Critique Generation**: Create natural language feedback
4. **Refinement**: Generate improved solutions using critique
5. **Selection**: Pick best refinements for off-policy training
6. **Training**: Update model using both on-policy (initial) and off-policy (refined) data

## Files

| File | Description |
|------|-------------|
| `critique_prompts.py` | Critique generation (3 types: simple, simple_gt, text) |
| `refinement_prompts.py` | Refinement prompt creation |
| `reward_function.py` | Math verification and reward computation |
| `critique_ray_trainer.py` | Main Ray-based trainer |
| `critique_vllm_rollout.py` | vLLM rollout with critique/refinement |
| `main_critique_grpo.py` | Entry point |
| `config/critique_grpo.yaml` | Default configuration |
| `run_critique_grpo.sh` | Training script |

## Critique Types

### 1. Simple (`simple`)
```
The generated solution is correct.
```
or
```
The generated solution is incorrect.
```

### 2. Simple with Ground Truth (`simple_gt`) - Recommended
```
The generated solution is incorrect, the ground truth is 42.
```

### 3. Text (Full LLM Critique) (`text`)
Uses an external LLM (e.g., GPT-4) to generate detailed step-by-step critiques.
Requires API configuration.

## Usage

### Quick Start

```bash
# Set your data and model paths
export DATA_DIR=/path/to/your/data
export MODEL_PATH=Qwen/Qwen2.5-7B

# Run training
bash recipe/critique_grpo/run_critique_grpo.sh
```

### Custom Configuration

```bash
# Configure training parameters
export TRAIN_BATCH_SIZE=128
export ROLLOUT_N=8
export N_PREFIX=1  # Number of refinements per prompt
export CRITIQUE_TYPE=simple_gt
export LEARNING_RATE=1e-6

# Run with custom settings
bash recipe/critique_grpo/run_critique_grpo.sh
```

### Using Python Directly

```bash
python -m recipe.critique_grpo.main_critique_grpo \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.rollout.critique_type=simple_gt \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_prefix=1
```

## Key Parameters

### Critique Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `critique_type` | Type of critique (simple, simple_gt, text) | `simple_gt` |
| `n_prefix` | Number of refinements per prompt | `1` |
| `max_refinement_length` | Max tokens for refinement | `6144` |

### Training Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `rollout.n` | Number of responses per prompt | `8` |
| `use_off_policy_loss` | Enable off-policy learning | `True` |
| `off_policy_reshape` | Off-policy weight reshape | `p_div_p_0.1` |
| `norm_adv_by_std_in_grpo` | Normalize advantages | `False` |

## Data Format

Your training data should be in Parquet format with columns:
- `prompt`: The question/problem
- `ground_truth`: The correct answer
- Additional metadata as needed

Example:
```python
import pandas as pd

data = {
    "prompt": ["What is 2 + 2?", "What is 3 * 4?"],
    "ground_truth": ["4", "12"]
}
df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

## Expected Results

Based on the paper:
- **+4.4%** pass@1 improvement on Qwen2.5-7B-Base
- **+16.7%** pass@1 on AIME 2024 vs standard GRPO
- Effective self-improvement through self-critiquing

## Customization

### Custom Reward Function

```python
from recipe.critique_grpo.reward_function import CritiqueRewardFunction

reward_fn = CritiqueRewardFunction(
    use_math_verify=True,
    format_weight=0.1,  # Weight for format checking
    correct_reward=1.0,
    incorrect_reward=0.0
)

score = reward_fn(solution="\\boxed{42}", ground_truth="42")
```

### Custom Critique Generation

```python
from recipe.critique_grpo.critique_prompts import generate_critique

sample = {
    "question": "What is 2 + 2?",
    "response": "2 + 2 = 5",
    "gt": "4",
    "score": 0.0
}

# Generate critique
result = generate_critique(sample, critique_type="simple_gt")
print(result["critique"])
# Output: "The generated solution is incorrect, the ground truth is 4."
```

## Citation

```bibtex
@article{zhang2025critique,
  title={Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback},
  author={Zhang, Xiaoying and others},
  journal={arXiv preprint arXiv:2506.03106},
  year={2025}
}
```

## References

- Paper: [arXiv:2506.03106](https://arxiv.org/abs/2506.03106)
- Reference Implementation: [github.com/zhangxy-2019/critique-GRPO](https://github.com/zhangxy-2019/critique-GRPO)
