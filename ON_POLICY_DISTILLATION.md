# On-Policy Distillation (GKD) for VERL

## Overview

On-Policy Distillation, also known as Generalized Knowledge Distillation (GKD), is a training method that combines the benefits of both distillation and reinforcement learning. It was introduced in the paper "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (https://arxiv.org/abs/2306.13649).

## Key Concepts

### The Problem: Train-Test Mismatch

Traditional distillation trains the student model on teacher-generated sequences (off-policy), but at inference time, the student generates its own sequences. This creates a distribution mismatch that can hurt performance.

### The Solution: On-Policy Distillation

On-policy distillation trains the student on **its own generated sequences** while using the teacher to provide feedback on those sequences. This eliminates the train-test mismatch and provides much denser supervision than RL.

### Comparison with RL

| Method | Information Density | Speed | Supervision Type |
|--------|-------------------|-------|------------------|
| **RL (GRPO/PPO)** | O(1) bits per episode | Baseline | Sparse (reward at end) |
| **On-Policy Distillation** | O(N) bits per episode | **7-10x faster** | Dense (every token) |

On-policy distillation reaches teacher-level performance in ~10 gradient steps vs ~70 steps for RL!

## Implementation in VERL

### Core Loss Functions

We implement three types of KL divergence losses:

#### 1. Reverse KL (Recommended)
```python
Loss = KL(student || teacher)
```
- **Mode-seeking**: Student focuses on teacher's modes
- **Best for distillation**: Avoids low-quality generations
- Set `distillation_type="reverse_kl"` or `distillation_beta=1.0`

#### 2. Forward KL (Standard Distillation)
```python
Loss = KL(teacher || student)
```
- **Mean-seeking**: Student tries to cover all teacher modes
- **More diverse** but may include low-quality outputs
- Set `distillation_type="forward_kl"` or `distillation_beta=0.0`

#### 3. Generalized JSD (Flexible)
```python
Loss = beta * KL(teacher || mixture) + (1-beta) * KL(student || mixture)
```
- **Interpolates** between forward and reverse KL
- Set `distillation_type="gkd"` and `distillation_beta` ∈ [0, 1]

### Configuration Parameters

```python
config = {
    # Distillation type
    "distillation_type": "reverse_kl",  # or "forward_kl", "gkd"

    # Beta parameter (for GKD)
    # 0.0 = forward KL, 1.0 = reverse KL
    "distillation_beta": 1.0,

    # Temperature for softmax (lower = sharper)
    "distillation_temperature": 1.0,

    # Hybrid mode weights
    "hybrid_lambda_rl": 1.0,      # Weight for RL loss
    "hybrid_lambda_distill": 1.0,  # Weight for distillation loss
}
```

## Usage

### Option 1: Pure On-Policy Distillation

Use `compute_on_policy_distillation_loss` as your policy loss function:

```python
from verl.trainer.ppo.on_policy_distillation import compute_on_policy_distillation_loss

# In your training config
algo_config = {
    "policy_loss": "on_policy_distillation",
    "distillation_type": "reverse_kl",
    "distillation_temperature": 1.0,
}
```

### Option 2: Hybrid RL + Distillation

Combine RL rewards with teacher distillation:

```python
algo_config = {
    "policy_loss": "hybrid_rl_distillation",
    "distillation_type": "reverse_kl",
    "hybrid_lambda_rl": 0.5,       # Weight for RL loss
    "hybrid_lambda_distill": 0.5,  # Weight for distillation loss
}
```

### Training Flow

```
1. Student generates sequences (on-policy)
   ├─> prompt + student_response

2. Get feedback from both sources:
   ├─> Teacher provides logits for student's sequences
   └─> Reward model scores the sequences (for hybrid mode)

3. Compute loss:
   ├─> Distillation: KL(student || teacher) on student's generations
   └─> RL (hybrid): Policy gradient with advantages

4. Update student with combined loss
```

## Key Advantages

### 1. Faster Convergence
- **7-10x faster** than pure RL methods
- Reaches teacher performance in ~10 steps vs ~70 for RL
- Dense supervision at every token

### 2. No Distribution Mismatch
- Trains on student's own generations
- Matches inference-time distribution
- Better generalization

### 3. Simpler than RL
- No advantage estimation needed (for pure distillation)
- No value functions or critics
- Just KL divergence between distributions

### 4. Flexible Hybridization
- Can combine with RL for best of both worlds
- Distillation provides dense signal
- RL provides task-specific optimization

## Example: Training with On-Policy Distillation

```python
import torch
from verl.trainer.ppo.on_policy_distillation import compute_on_policy_distillation_loss

# Student generates sequences
student_sequences = student_model.generate(prompts)
student_logits = student_model(student_sequences)

# Teacher evaluates student's sequences
teacher_logits = teacher_model(student_sequences)

# Compute distillation loss
loss, _, kl, _ = compute_on_policy_distillation_loss(
    old_log_prob=None,  # Not used
    log_prob=None,      # Not used
    advantages=None,    # Not used
    response_mask=mask,
    loss_agg_mode="token-mean",
    config={
        "distillation_type": "reverse_kl",
        "distillation_temperature": 1.0,
    },
    teacher_logits=teacher_logits,
    student_logits=student_logits,
)

# Backward and update
loss.backward()
optimizer.step()
```

## When to Use What

### Use Pure On-Policy Distillation When:
- You have a strong teacher model
- You want fast convergence
- You don't have a good reward model
- Task is primarily about matching teacher quality

### Use Hybrid RL + Distillation When:
- You have both teacher and reward signals
- You want to go beyond teacher performance
- Task requires specific optimization (e.g., safety, instruction following)
- You want stability from distillation + flexibility from RL

### Use Pure RL When:
- No teacher model available
- Task is very different from teacher's training
- You need exploration beyond teacher's distribution

## Integration with VERL Trainers

The on-policy distillation losses are registered in VERL's policy loss registry and can be used with any VERL trainer:

```python
# In your trainer config
from verl.trainer.ppo.core_algos import get_policy_loss_fn

policy_loss_fn = get_policy_loss_fn("on_policy_distillation")

# Or for hybrid
policy_loss_fn = get_policy_loss_fn("hybrid_rl_distillation")
```

## Mathematical Formulation

### Reverse KL (Mode-Seeking)
```
L = Σ_x π_student(x) * log(π_student(x) / π_teacher(x))
  = E_x~student [log π_student(x) - log π_teacher(x)]
```

### Forward KL (Mean-Seeking)
```
L = Σ_x π_teacher(x) * log(π_teacher(x) / π_student(x))
  = E_x~teacher [log π_teacher(x) - log π_student(x)]
```

### Generalized JSD
```
M = beta * π_teacher + (1-beta) * π_student
L = Σ_x π_student(x) * log(π_student(x) / M(x))
```

## Performance Tips

1. **Start with reverse KL** (`distillation_type="reverse_kl"`)
   - Most stable and effective for distillation

2. **Use temperature = 1.0** for most cases
   - Lower values make distribution sharper
   - Higher values make it smoother

3. **For hybrid mode**, balance the weights:
   ```python
   # Start with equal weights
   hybrid_lambda_rl = 0.5
   hybrid_lambda_distill = 0.5

   # Adjust based on task:
   # More RL weight → more task-specific optimization
   # More distill weight → faster convergence, stay closer to teacher
   ```

4. **Monitor KL divergence**
   - Should decrease over time
   - If it increases, lower learning rate or increase temperature

## References

1. **Original Paper**: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
   - https://arxiv.org/abs/2306.13649

2. **Thinking Machines Blog**: "On-Policy Distillation"
   - https://thinkingmachines.ai/blog/on-policy-distillation/

3. **HuggingFace TRL**: GKD Trainer
   - https://huggingface.co/docs/trl/main/en/gkd_trainer

## Citation

If you use this implementation, please cite:

```bibtex
@article{agarwal2023on,
  title={On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  author={Agarwal, Rishabh and Vieillard, Nino and Stanczyk, Piotr and Ramos, Sabela and Geist, Matthieu and Bachem, Olivier},
  journal={arXiv preprint arXiv:2306.13649},
  year={2023}
}
```
