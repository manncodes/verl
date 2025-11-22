# Rollout Bucketing by Difficulty - User Guide

This guide explains how to use `bucket_rollouts_by_difficulty.py` to organize your JSONL rollouts into curriculum-based datasets.

## Overview

The script analyzes rollout data and buckets them by difficulty using various metrics. This is useful for:
- **Curriculum learning**: Training models progressively from easy to hard examples
- **Data analysis**: Understanding the difficulty distribution of your dataset
- **Stratified sampling**: Ensuring balanced difficulty in train/val/test splits

## Installation

The script requires Python 3.7+ with numpy:

```bash
pip install numpy
```

## Quick Start

### Basic Usage (3 buckets by mean reward)

```bash
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./bucketed \
    -m mean_reward \
    -n 3
```

This creates three files:
- `easy.jsonl` - Rollouts with highest mean reward
- `medium.jsonl` - Rollouts with medium mean reward
- `hard.jsonl` - Rollouts with lowest mean reward

## Difficulty Metrics

### 1. **mean_reward** (Recommended for most cases)
- Average reward score across all response choices
- **Interpretation**: Higher score = easier task
- **Use case**: General difficulty estimation

```bash
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward -n 3
```

### 2. **pass_at_k**
- Proportion of top-k responses passing a threshold (default: 0.5)
- **Interpretation**: Higher score = easier (more responses succeed)
- **Use case**: Focus on whether any response succeeds

```bash
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m pass_at_k -n 3
```

### 3. **max_reward**
- Best reward score achieved across all responses
- **Interpretation**: Higher score = easier (at least one good response)
- **Use case**: Optimistic difficulty (can the model solve it at all?)

```bash
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m max_reward -n 3
```

### 4. **variance**
- Variance in reward scores across responses
- **Interpretation**: Higher variance = more inconsistent/harder
- **Use case**: Identify tasks with inconsistent model performance

```bash
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m variance -n 3
```

### 5. **success_rate**
- Percentage of responses above threshold (default: 0.5)
- **Interpretation**: Higher rate = easier
- **Use case**: Binary success/failure analysis

```bash
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m success_rate -n 3
```

### 6. **inverse_mean**
- Inverse of mean reward (1 / mean_reward)
- **Interpretation**: Higher score = harder
- **Use case**: When you want harder tasks to have higher difficulty scores

```bash
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m inverse_mean -n 3
```

## Bucketing Methods

### Method 1: Percentile Bucketing (Default)

Divides rollouts into equal-sized buckets based on percentiles:

```bash
# 3 buckets (easy, medium, hard)
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward -n 3

# 5 buckets (very_easy, easy, medium, hard, very_hard)
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward -n 5

# Custom bucket names
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward -n 3 \
    --bucket-names beginner intermediate expert
```

### Method 2: Threshold Bucketing

Uses absolute threshold values:

```bash
# Create 3 buckets: score ≤ 0.3, 0.3 < score ≤ 0.7, score > 0.7
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward \
    --thresholds 0.3 0.7

# With custom names
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward \
    --thresholds 0.3 0.7 \
    --bucket-names low medium high
```

## Advanced Examples

### Example 1: Curriculum Learning Pipeline

```bash
# Step 1: Bucket by mean reward
python bucket_rollouts_by_difficulty.py \
    -i ./raw_rollouts \
    -o ./curriculum \
    -m mean_reward \
    -n 5 \
    --bucket-names stage1 stage2 stage3 stage4 stage5

# Now train progressively:
# - Phase 1: Train on stage1.jsonl
# - Phase 2: Train on stage1.jsonl + stage2.jsonl
# - Phase 3: Train on stage1-3.jsonl
# etc.
```

### Example 2: Analysis of Different Metrics

```bash
# Compare different metrics
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./analysis_mean -m mean_reward -n 3 --prefix mean_
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./analysis_variance -m variance -n 3 --prefix var_
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./analysis_passatk -m pass_at_k -n 3 --prefix pass_

# This creates:
# - analysis_mean/mean_easy.jsonl, mean_medium.jsonl, mean_hard.jsonl
# - analysis_variance/var_easy.jsonl, var_medium.jsonl, var_hard.jsonl
# - analysis_passatk/pass_easy.jsonl, pass_medium.jsonl, pass_hard.jsonl
```

### Example 3: Fine-Grained Difficulty Levels

```bash
# 10 difficulty levels for very fine-grained curriculum
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./fine_grained \
    -m mean_reward \
    -n 10 \
    --bucket-names level_0 level_1 level_2 level_3 level_4 level_5 level_6 level_7 level_8 level_9
```

## Understanding the Output

The script prints useful statistics:

```
============================================================
Bucketing by mean_reward using 3 percentile buckets
============================================================

Found 5 JSONL files in ./rollouts
Loaded 10000 rollouts

Difficulty percentiles (mean_reward):
  0%: 0.0000
  33%: 0.2500
  67%: 0.6250
  100%: 1.2500

Bucket statistics:
  easy: 3334 rollouts
  medium: 3333 rollouts
  hard: 3333 rollouts

Saved 3334 rollouts to ./bucketed/easy.jsonl
Saved 3333 rollouts to ./bucketed/medium.jsonl
Saved 3333 rollouts to ./bucketed/hard.jsonl

============================================================
Bucketing complete! Output saved to ./bucketed
============================================================
```

## Choosing the Right Metric

| Metric | Best For | Difficulty Direction |
|--------|----------|---------------------|
| `mean_reward` | General purpose difficulty | Higher = easier |
| `pass_at_k` | Code generation, any-correct scenarios | Higher = easier |
| `max_reward` | Optimistic difficulty (best case) | Higher = easier |
| `variance` | Finding inconsistent tasks | Higher = harder |
| `success_rate` | Binary success/failure tasks | Higher = easier |
| `inverse_mean` | When you want higher scores for harder tasks | Higher = harder |

## Recommendations

1. **For curriculum learning**: Start with `mean_reward` with 3-5 buckets
2. **For code generation**: Use `pass_at_k` to focus on solvability
3. **For debugging**: Use `variance` to find inconsistent examples
4. **For balanced datasets**: Use percentile bucketing with equal buckets

## Integration with Training

After bucketing, you can use the buckets in your training pipeline:

```python
# Example: Progressive curriculum training
buckets = ['easy.jsonl', 'medium.jsonl', 'hard.jsonl']

for i, bucket in enumerate(buckets):
    print(f"Training phase {i+1}: Adding {bucket}")
    # Load all buckets up to current level
    train_data = load_jsonl(buckets[:i+1])
    train_model(train_data)
```

## Troubleshooting

### Issue: All rollouts go to one bucket
- **Solution**: Try a different metric or check your reward values
- Some metrics like `variance` may have a skewed distribution

### Issue: Empty buckets with threshold bucketing
- **Solution**: Adjust thresholds based on your data distribution
- Run with percentile bucketing first to see the actual difficulty range

### Issue: Script runs slowly
- **Solution**: The script processes all JSONL files efficiently
- For very large datasets (millions of rollouts), consider processing in batches

## Command-Line Reference

```
usage: bucket_rollouts_by_difficulty.py [-h] -i INPUT_DIR -o OUTPUT_DIR
                                       [-m {mean_reward,max_reward,min_reward,variance,pass_at_k,success_rate,inverse_mean}]
                                       [-n NUM_BUCKETS]
                                       [--bucket-names BUCKET_NAMES [BUCKET_NAMES ...]]
                                       [--thresholds THRESHOLDS [THRESHOLDS ...]]
                                       [--prefix PREFIX]

Required arguments:
  -i, --input-dir       Input directory containing JSONL files
  -o, --output-dir      Output directory for bucketed JSONL files

Optional arguments:
  -m, --metric          Difficulty metric to use (default: mean_reward)
  -n, --num-buckets     Number of buckets for percentile bucketing (default: 3)
  --bucket-names        Custom names for buckets
  --thresholds          Absolute thresholds for bucketing
  --prefix              Prefix for output filenames
```

## Advanced: Custom Metrics

To add your own difficulty metric, edit the `DifficultyMetrics` class:

```python
class DifficultyMetrics:
    @staticmethod
    def my_custom_metric(rollout: Dict[str, Any]) -> float:
        """Your custom difficulty calculation."""
        # Example: Weighted combination of mean and variance
        mean = DifficultyMetrics.mean_reward(rollout)
        var = DifficultyMetrics.reward_variance(rollout)
        return 0.7 * mean + 0.3 * var

# Then add to DIFFICULTY_FUNCTIONS in RolloutBucketer
DIFFICULTY_FUNCTIONS = {
    # ... existing metrics ...
    'custom': DifficultyMetrics.my_custom_metric,
}
```
