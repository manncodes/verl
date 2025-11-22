# Quick Start: Bucket Rollouts by Difficulty

## TL;DR

```bash
# Install (optional - works without numpy too)
pip install numpy

# Basic usage - bucket into easy/medium/hard
python bucket_rollouts_by_difficulty.py \
    -i /path/to/rollouts \
    -o /path/to/output \
    -m mean_reward \
    -n 3
```

## What This Does

Your rollout JSONL files contain multiple response choices per prompt, each with a reward score. This script:

1. **Analyzes** all rollouts to calculate difficulty based on reward metrics
2. **Buckets** them into groups (e.g., easy/medium/hard)
3. **Saves** separate JSONL files for each difficulty level

## Common Use Cases

### 1. Curriculum Learning (Recommended)

Train your model progressively from easy to hard examples:

```bash
# Create 5-stage curriculum
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./curriculum \
    -m mean_reward \
    -n 5 \
    --bucket-names warmup basic intermediate advanced expert
```

### 2. Data Quality Analysis

Find which examples are too hard/easy:

```bash
# Bucket by variance to find inconsistent examples
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./analysis \
    -m variance \
    -n 3
```

### 3. Pass@K Analysis

Focus on code generation success rate:

```bash
# Bucket by pass@k
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./pass_at_k_buckets \
    -m pass_at_k \
    -n 3
```

## Understanding Your Data

Test with the example script to understand your rollout metrics:

```bash
python example_bucketing_usage.py 6
```

This shows all difficulty metrics for a sample rollout:
- `mean_reward`: Average success (0.31 in example)
- `max_reward`: Best response (1.25 in example)
- `pass_at_k`: Success rate (25% in example)
- `variance`: Consistency (0.33 in example)

## Files Created

After bucketing with 3 buckets, you'll have:

```
output/
├── easy.jsonl      # Highest mean reward / easiest tasks
├── medium.jsonl    # Middle difficulty
└── hard.jsonl      # Lowest mean reward / hardest tasks
```

Each file contains full rollout entries, just like your original JSONL files.

## Quick Reference

| Goal | Command |
|------|---------|
| 3 difficulty levels | `-n 3` |
| 5 difficulty levels | `-n 5` |
| Custom names | `--bucket-names easy medium hard` |
| Absolute thresholds | `--thresholds 0.3 0.7` |
| Different metric | `-m pass_at_k` |

## Available Metrics

- `mean_reward` - Average score (default, recommended)
- `pass_at_k` - Success rate for code/generation tasks
- `max_reward` - Best case performance
- `variance` - Consistency of responses
- `success_rate` - Binary pass/fail rate

## Full Documentation

See `BUCKETING_GUIDE.md` for detailed documentation and advanced usage.

## Troubleshooting

**Q: ImportError for numpy**
A: Script works without numpy! Just ignore the warning.

**Q: All rollouts in one bucket**
A: Your data may have limited reward variance. Try a different metric like `variance` or `pass_at_k`.

**Q: How do I know which metric to use?**
A: Start with `mean_reward`. If you're doing code generation, try `pass_at_k`.
