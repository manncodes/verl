# GDPO: Group reward-Decoupled normalization Policy Optimization

GDPO is a multi-reward RL optimization method that addresses a fundamental limitation in GRPO when dealing with multiple reward signals.

## Paper

[GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](https://arxiv.org/abs/2601.05242)

## Problem

When directly applying GRPO to normalize distinct rollout reward combinations, they collapse into identical advantage values. This reduces the resolution of the training signal and results in suboptimal convergence.

## Solution

GDPO resolves this by decoupling the normalization of individual rewards, more faithfully preserving their relative differences.

### Key Difference from GRPO

**GRPO (standard approach):**
```
combined_reward = w1*r1 + w2*r2 + ...
normalized_adv = (combined_reward - group_mean) / group_std
```

**GDPO (decoupled approach):**
```
normalized_r1 = (r1 - group_mean(r1)) / group_std(r1)
normalized_r2 = (r2 - group_mean(r2)) / group_std(r2)
combined_adv = w1*normalized_r1 + w2*normalized_r2 + ...
```

## Usage

### Single Reward (Equivalent to GRPO)

When using a single reward signal, GDPO behaves identically to GRPO:

```yaml
algorithm:
  adv_estimator: gdpo
  norm_adv_by_std_in_grpo: True
```

### Multi-Reward Setup

To use GDPO with multiple rewards:

1. **Store rewards in batch**: Store each reward tensor in `data.batch` with a `reward_` prefix:
   - `data.batch["reward_correctness"]` - correctness reward tensor
   - `data.batch["reward_format"]` - format reward tensor
   - `data.batch["reward_length"]` - length reward tensor

2. **Configure weights**: Set weights in your config:
   ```yaml
   algorithm:
     adv_estimator: gdpo
     gdpo_reward_weights:
       correctness: 1.0
       format: 0.5
       length: 0.3
   ```

### Example: Tool Calling Task

For a tool calling task with correctness and format rewards:

```yaml
algorithm:
  adv_estimator: gdpo
  norm_adv_by_std_in_grpo: True
  gdpo_reward_weights:
    correctness: 1.0  # Main objective
    format: 0.5       # Secondary constraint
```

### Example: Math Reasoning Task

For math reasoning with accuracy and length constraints:

```yaml
algorithm:
  adv_estimator: gdpo
  norm_adv_by_std_in_grpo: True
  gdpo_reward_weights:
    accuracy: 1.0     # Primary reward
    length: 0.3       # Penalty for excessive length
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `adv_estimator` | str | - | Set to `"gdpo"` to enable GDPO |
| `norm_adv_by_std_in_grpo` | bool | `True` | Whether to normalize by std (True=GRPO style, False=Dr.GRPO style) |
| `gdpo_reward_weights` | dict | `None` | Mapping of reward names to weights |

## Implementing Custom Multi-Reward

To implement your own multi-reward setup:

1. In your reward function, return multiple reward tensors:
   ```python
   def compute_rewards(data):
       # Compute individual rewards
       correctness_reward = compute_correctness(data)
       format_reward = compute_format(data)

       # Store in batch with 'reward_' prefix
       data.batch["reward_correctness"] = correctness_reward
       data.batch["reward_format"] = format_reward

       # Also set combined reward for token_level_rewards
       data.batch["token_level_rewards"] = correctness_reward + format_reward
   ```

2. Configure GDPO in your training config:
   ```yaml
   algorithm:
     adv_estimator: gdpo
     gdpo_reward_weights:
       correctness: 1.0
       format: 0.5
   ```

## Comparison with GRPO

| Aspect | GRPO | GDPO |
|--------|------|------|
| Normalization | Combined reward normalized | Each reward normalized independently |
| Multi-reward | Rewards collapse into similar advantages | Preserves relative differences |
| Single reward | Standard behavior | Equivalent to GRPO |
| Training stability | May have issues with multi-reward | Improved stability |

## References

- [GDPO Paper](https://arxiv.org/abs/2601.05242)
- [GDPO GitHub](https://github.com/NVlabs/GDPO)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
