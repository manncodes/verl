# RayPPOTrainerWithSampling - Usage Guide

## Overview

`RayPPOTrainerWithSampling` is an extension of `RayPPOTrainer` that implements dynamic sampling (also known as "group filtering" or "prompt filtering"). This mechanism, originally developed for DAPO, filters out prompts that are too easy or too hard for the model, focusing training on prompts that provide meaningful learning signals.

## Key Concept

When training with multiple responses per prompt (e.g., `n=5` for GRPO), we can analyze the variance in rewards:

- **Too Easy**: All responses get high rewards (e.g., all correct) → std ≈ 0 → **FILTERED OUT**
- **Too Hard**: All responses get low rewards (e.g., all wrong) → std ≈ 0 → **FILTERED OUT**
- **Just Right**: Mixed results (some correct, some wrong) → std > 0 → **KEPT**

This focuses training on the "zone of proximal development" where the model has partial knowledge and can learn most effectively.

## Configuration

### Required Config Parameters

```yaml
algorithm:
  filter_groups:
    enable: true                    # Enable dynamic sampling
    metric: "acc"                   # Metric to filter on: "acc", "score", "seq_reward", "seq_final_reward"
    max_num_gen_batches: 10         # Max generation attempts (0 = unlimited)

data:
  gen_batch_size: 1536             # Prompts per generation batch
  train_batch_size: 512            # Target prompts per training batch

actor_rollout_ref:
  rollout:
    n: 5                            # Multiple responses per prompt (required for filtering)
```

### Supported Metrics

1. **`acc`**: Binary accuracy (0 or 1) from reward function
   - Requires your reward function to return `acc` in `reward_extra_info`
   - Best for mathematical reasoning, code generation, etc.

2. **`score`**: Continuous score value
   - Requires your reward function to return `score` in `reward_extra_info`
   - Best for nuanced tasks with graded rewards

3. **`seq_reward`**: Sum of token-level scores (before KL penalty)
   - Computed from `token_level_scores`
   - Always available

4. **`seq_final_reward`**: Sum of token-level rewards (after KL penalty)
   - Computed from `token_level_rewards`
   - Always available

## Usage Examples

### Example 1: GRPO with Dynamic Sampling

```python
from verl.trainer.ppo import RayPPOTrainerWithSampling
from verl.trainer.config import TrainerConfig

# Configure with dynamic sampling
config = TrainerConfig(
    algorithm=AlgoConfig(
        adv_estimator="grpo",
        use_kl_in_reward=False,
        norm_adv_by_std_in_grpo=True,
        filter_groups=FilterGroupsConfig(
            enable=True,
            metric="acc",  # Filter based on accuracy variance
            max_num_gen_batches=10,
        ),
    ),
    data=DataConfig(
        gen_batch_size=1536,
        train_batch_size=512,
    ),
    actor_rollout_ref=ActorRolloutRefConfig(
        rollout=RolloutConfig(
            n=5,  # Sample 5 responses per prompt
        ),
        actor=ActorConfig(
            use_kl_loss=True,
            kl_loss_coef=0.001,
        ),
    ),
    trainer=PPOTrainerConfig(
        critic_warmup=0,  # GRPO doesn't use critic
    ),
)

# Create trainer
trainer = RayPPOTrainerWithSampling(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    reward_fn=reward_fn,
    train_dataset=train_dataset,
)

# Train
trainer.fit()
```

### Example 2: PPO with Dynamic Sampling

```python
from verl.trainer.ppo import RayPPOTrainerWithSampling

# Configure PPO with dynamic sampling
config = TrainerConfig(
    algorithm=AlgoConfig(
        adv_estimator="gae",  # Standard PPO advantage estimation
        use_kl_in_reward=True,
        filter_groups=FilterGroupsConfig(
            enable=True,
            metric="seq_reward",  # Filter based on reward variance
            max_num_gen_batches=5,
        ),
    ),
    data=DataConfig(
        gen_batch_size=1024,
        train_batch_size=256,
    ),
    actor_rollout_ref=ActorRolloutRefConfig(
        rollout=RolloutConfig(
            n=4,  # Sample 4 responses per prompt
        ),
    ),
    trainer=PPOTrainerConfig(
        critic_warmup=100,  # Warmup critic for PPO
    ),
)

trainer = RayPPOTrainerWithSampling(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)

trainer.fit()
```

### Example 3: Command-line Usage

You can use the trainer with any existing VERL recipe by modifying the trainer class:

```python
# In your main training script (e.g., main_grpo.py)

# Before:
# from verl.trainer.ppo import RayPPOTrainer

# After:
from verl.trainer.ppo import RayPPOTrainerWithSampling as RayPPOTrainer

# Rest of the code remains the same...
trainer = RayPPOTrainer(...)
trainer.fit()
```

Then configure via command line:

```bash
python main_grpo.py \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    data.gen_batch_size=1536 \
    data.train_batch_size=512 \
    actor_rollout_ref.rollout.n=5
```

## How It Works

### Training Loop Flow

1. **Generate Batch**: Sample `gen_batch_size` prompts, generate `n` responses each
   - Total trajectories: `gen_batch_size * n`

2. **Compute Rewards**: Calculate metric for each response

3. **Filter Prompts**:
   ```python
   # Group responses by prompt UID
   for each prompt:
       metric_std = std(responses' metrics)
       if metric_std > 0:
           keep_prompt()
       else:
           filter_out_prompt()
   ```

4. **Check Batch Size**:
   - If `num_kept_prompts < train_batch_size`:
     - Generate another batch (up to `max_num_gen_batches`)
     - Accumulate filtered prompts
     - Repeat until enough prompts
   - Else:
     - Trim to exact `train_batch_size * n` trajectories
     - Proceed to training

5. **Train**: Standard PPO/GRPO update on filtered batch

### Monitoring

The trainer logs `train/num_gen_batches` metric showing how many generation batches were needed per training step:

- `num_gen_batches = 1`: All prompts had variance (data is well-suited)
- `num_gen_batches > 1`: Some filtering occurred (expected)
- `num_gen_batches = max_num_gen_batches`: Hit the limit (data may be too easy/hard)

## Performance Tuning

### Batch Size Selection

```python
# Rule of thumb:
gen_batch_size = 2 * train_batch_size  # to 3 * train_batch_size

# Example:
# - train_batch_size = 512
# - gen_batch_size = 1024 to 1536
# - Expected filtering: ~30-50% of prompts
```

### Max Generation Batches

```python
# Conservative (fail fast if data is problematic):
max_num_gen_batches = 5

# Moderate (recommended):
max_num_gen_batches = 10

# Aggressive (keep trying):
max_num_gen_batches = 20

# Unlimited (not recommended):
max_num_gen_batches = 0  # or negative
```

### Choosing the Right Metric

| Metric | Use Case | Requirements |
|--------|----------|--------------|
| `acc` | Binary tasks (correct/wrong) | Reward function must return `acc` |
| `score` | Graded tasks (continuous scores) | Reward function must return `score` |
| `seq_reward` | General (always available) | None |
| `seq_final_reward` | With KL penalty | `use_kl_in_reward=True` |

## Comparison: With vs Without Dynamic Sampling

### Without Dynamic Sampling (Standard Training)

```yaml
algorithm:
  filter_groups:
    enable: false

data:
  train_batch_size: 512
  # All 512 prompts are used, regardless of difficulty
```

**Characteristics:**
- Simpler, faster per step
- Uses all data
- May waste compute on uninformative prompts
- Consistent batch sizes

### With Dynamic Sampling

```yaml
algorithm:
  filter_groups:
    enable: true
    metric: acc
    max_num_gen_batches: 10

data:
  gen_batch_size: 1536
  train_batch_size: 512
  # Only ~512 prompts with variance are kept
```

**Characteristics:**
- More selective data usage
- Better sample efficiency
- Variable generation time per step
- Focuses on "learnable" prompts

## Integration with Existing Recipes

### DAPO Recipe

The original DAPO recipe (`recipe/dapo/`) uses a specialized trainer. You can replace it:

```python
# Before:
# from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

# After:
from verl.trainer.ppo import RayPPOTrainerWithSampling as RayDAPOTrainer
```

The core filtering logic is identical.

### Other Recipes

Any recipe using `RayPPOTrainer` can use dynamic sampling:

```python
# entropy recipe, sppo recipe, etc.
from verl.trainer.ppo import RayPPOTrainerWithSampling as RayPPOTrainer

# Just add filter_groups config
config.algorithm.filter_groups = FilterGroupsConfig(
    enable=True,
    metric="acc",
    max_num_gen_batches=10,
)
```

## Troubleshooting

### Error: "Generated too many batches"

**Problem**: Hit `max_num_gen_batches` limit
```
ValueError: num_gen_batches=10 >= max_num_gen_batches=10.
Generated too many batches. Please check if your data are too difficult.
```

**Solutions:**
1. Increase `max_num_gen_batches`
2. Increase `gen_batch_size` (generate more prompts per batch)
3. Check if your data is too homogeneous (all easy or all hard)
4. Consider using a different metric
5. Set `max_num_gen_batches=0` for unlimited attempts (not recommended)

### Slow Training

**Problem**: Each training step takes longer than expected

**Solutions:**
1. Reduce `gen_batch_size` (less generation overhead)
2. Increase `max_num_gen_batches` (fail faster)
3. Check `train/num_gen_batches` metric:
   - If consistently high (>5), your data may not be suitable for filtering
   - Consider disabling filtering: `filter_groups.enable=false`

### No Filtering Effect

**Problem**: `train/num_gen_batches` is always 1

**Possible causes:**
1. Your data has natural variance (good!)
2. Metric calculation is incorrect
3. `n` is too small (try `n ≥ 5`)

## References

- Original DAPO paper: [Link to paper if available]
- VERL DAPO recipe: `recipe/dapo/`
- Base trainer: `verl/trainer/ppo/ray_trainer.py`
- This implementation: `verl/trainer/ppo/ray_trainer_with_sampling.py`
