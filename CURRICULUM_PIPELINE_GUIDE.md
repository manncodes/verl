# Curriculum Learning Pipeline Guide

Complete workflow from rollouts to curriculum-based training.

## Overview

This pipeline enables curriculum learning by:
1. **Bucketing** rollouts by difficulty
2. **Preprocessing** bucketed data into training format
3. **Progressive training** from easy to hard examples

## Complete Workflow

### Step 1: Generate Rollouts

Generate rollouts using your existing pipeline:

```bash
# Your existing rollout generation
python generate_rollouts.py --config your_config.yaml
# Output: ./rollouts/*.jsonl
```

### Step 2: Bucket by Difficulty

Organize rollouts into difficulty levels:

```bash
# Basic 3-level curriculum (easy/medium/hard)
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./bucketed \
    -m mean_reward \
    -n 3

# Output:
# ./bucketed/easy.jsonl
# ./bucketed/medium.jsonl
# ./bucketed/hard.jsonl
```

### Step 3: Preprocess for Training

Convert bucketed rollouts to training format:

```bash
# Process all buckets separately
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./preprocessed

# Output:
# ./preprocessed/easy_train.parquet
# ./preprocessed/easy_val.parquet
# ./preprocessed/medium_train.parquet
# ./preprocessed/medium_val.parquet
# ./preprocessed/hard_train.parquet
# ./preprocessed/hard_val.parquet
```

### Step 4: Train with Curriculum

Progressive training strategy:

```bash
# Phase 1: Train on easy examples
python train.py --data ./preprocessed/easy_train.parquet

# Phase 2: Fine-tune on easy + medium
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./progressive \
    --buckets easy medium \
    --combine \
    --output-name phase2

python train.py --data ./progressive/phase2_train.parquet

# Phase 3: Fine-tune on all data
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./progressive \
    --buckets easy medium hard \
    --combine \
    --output-name phase3

python train.py --data ./progressive/phase3_train.parquet
```

## Common Workflows

### Workflow A: 5-Stage Curriculum

Fine-grained curriculum with 5 stages:

```bash
# Step 1: Bucket into 5 levels
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./curriculum_5stage \
    -m mean_reward \
    -n 5 \
    --bucket-names stage1 stage2 stage3 stage4 stage5

# Step 2: Preprocess each stage
python preprocess_bucketed_rollouts.py \
    -i ./curriculum_5stage \
    -o ./preprocessed_5stage

# Step 3: Progressive training
for stage in 1 2 3 4 5; do
    echo "Training stage $stage..."
    python train.py --data ./preprocessed_5stage/stage${stage}_train.parquet
done
```

### Workflow B: Code Generation with Pass@K

Optimize for code generation tasks:

```bash
# Step 1: Bucket by pass@k metric
python bucket_rollouts_by_difficulty.py \
    -i ./code_rollouts \
    -o ./code_bucketed \
    -m pass_at_k \
    -n 3

# Step 2: Only use examples with at least one passing response
python preprocess_bucketed_rollouts.py \
    -i ./code_bucketed \
    -o ./code_preprocessed \
    --filter passing_only

# Step 3: Train progressively
python train.py --data ./code_preprocessed/easy_train.parquet
python train.py --data ./code_preprocessed/medium_train.parquet
python train.py --data ./code_preprocessed/hard_train.parquet
```

### Workflow C: Data Quality Analysis

Identify and handle different difficulty levels:

```bash
# Step 1: Bucket by variance (find inconsistent examples)
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./analysis \
    -m variance \
    -n 3 \
    --bucket-names consistent moderate inconsistent

# Step 2: Analyze each bucket
python preprocess_bucketed_rollouts.py \
    -i ./analysis \
    -o ./analyzed

# Step 3: Train on consistent examples first
python train.py --data ./analyzed/consistent_train.parquet
```

### Workflow D: Incremental Curriculum

Gradually increase difficulty:

```bash
# Create cumulative datasets
# Stage 1: Just easy
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./cumulative \
    --buckets easy \
    --combine \
    --output-name cumulative_stage1

# Stage 2: Easy + Medium
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./cumulative \
    --buckets easy medium \
    --combine \
    --output-name cumulative_stage2

# Stage 3: All
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./cumulative \
    --buckets easy medium hard \
    --combine \
    --output-name cumulative_stage3

# Train incrementally
python train.py --data ./cumulative/cumulative_stage1_train.parquet --checkpoint ckpt1
python train.py --data ./cumulative/cumulative_stage2_train.parquet --load ckpt1 --checkpoint ckpt2
python train.py --data ./cumulative/cumulative_stage3_train.parquet --load ckpt2 --checkpoint ckpt3
```

## Advanced Options

### Custom Difficulty Thresholds

Use absolute thresholds instead of percentiles:

```bash
# Bucket by specific score thresholds
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./custom_buckets \
    -m mean_reward \
    --thresholds 0.3 0.7 \
    --bucket-names low medium high
```

### Filter Low-Quality Examples

Only include examples with good responses:

```bash
# Only include rollouts with at least one positive score
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./filtered \
    --filter best_only

# Only include rollouts with passing responses (score >= 0.5)
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./filtered \
    --filter passing_only
```

### Multiple Metrics Comparison

Compare different difficulty metrics:

```bash
# Try different metrics
for metric in mean_reward pass_at_k variance; do
    python bucket_rollouts_by_difficulty.py \
        -i ./rollouts \
        -o ./compare_${metric} \
        -m ${metric} \
        -n 3

    python preprocess_bucketed_rollouts.py \
        -i ./compare_${metric} \
        -o ./preprocessed_${metric}
done

# Train and compare results
```

## Understanding the Data Format

### Bucketed JSONL Format

Each rollout in the bucketed JSONL contains:

```json
{
    "num": 4127,
    "prompt": [...],
    "response": {...},
    "reward": [
        {"score": 0, "V_i": 0.0, ...},
        {"score": 1.25, "V_i": 0.25, ...}
    ],
    "parameters": {...},
    "key": "...",
    "messages": [...],
    "ground_truth": "...",
    "extra_info": {...}
}
```

### Preprocessed Training Format

After preprocessing, data is in verl format:

```json
{
    "data_source": "bucketed_curriculum",
    "prompt": [
        {"role": "user", "content": "..."}
    ],
    "ability": "instruction_following",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "instruction_id_list": [...],
            "kwargs": [...]
        }
    },
    "extra_info": {
        "curriculum_stage": "easy",
        "difficulty_metrics": {
            "mean_score": 0.31,
            "max_score": 1.25
        },
        "best_response": {...}
    }
}
```

## Monitoring Progress

### Track Curriculum Stages

Monitor training across stages:

```python
# Add to your training script
stage_metrics = {
    "stage1": {"train_loss": [], "val_loss": []},
    "stage2": {"train_loss": [], "val_loss": []},
    "stage3": {"train_loss": [], "val_loss": []}
}

# Log when transitioning stages
print(f"Transitioning from stage {i} to stage {i+1}")
```

### Evaluate on Each Difficulty Level

Test model performance on each difficulty:

```bash
# Evaluate on easy examples
python evaluate.py --data ./preprocessed/easy_val.parquet

# Evaluate on medium examples
python evaluate.py --data ./preprocessed/medium_val.parquet

# Evaluate on hard examples
python evaluate.py --data ./preprocessed/hard_val.parquet
```

## Pipeline Automation

### Complete Pipeline Script

Create `run_curriculum_pipeline.sh`:

```bash
#!/bin/bash

ROLLOUT_DIR="./rollouts"
BUCKET_DIR="./bucketed"
PREPROCESS_DIR="./preprocessed"
NUM_STAGES=3

# Step 1: Bucket rollouts
echo "Step 1: Bucketing rollouts..."
python bucket_rollouts_by_difficulty.py \
    -i $ROLLOUT_DIR \
    -o $BUCKET_DIR \
    -m mean_reward \
    -n $NUM_STAGES

# Step 2: Preprocess each bucket
echo "Step 2: Preprocessing buckets..."
python preprocess_bucketed_rollouts.py \
    -i $BUCKET_DIR \
    -o $PREPROCESS_DIR

# Step 3: Create cumulative datasets
echo "Step 3: Creating cumulative datasets..."
python preprocess_bucketed_rollouts.py \
    -i $BUCKET_DIR \
    -o ${PREPROCESS_DIR}/cumulative \
    --buckets easy medium \
    --combine \
    --output-name stage1_and_2

# Step 4: Train
echo "Step 4: Training..."
python train.py --data ${PREPROCESS_DIR}/easy_train.parquet --output stage1.ckpt
python train.py --data ${PREPROCESS_DIR}/cumulative/stage1_and_2_train.parquet --load stage1.ckpt --output stage2.ckpt
python train.py --data ${PREPROCESS_DIR}/hard_train.parquet --load stage2.ckpt --output final.ckpt

echo "Pipeline complete!"
```

## Troubleshooting

### Issue: Empty buckets after preprocessing

**Cause**: Filter mode too strict
**Solution**: Use `--filter all` or check your rollout scores

### Issue: All examples in one bucket

**Cause**: Limited score variance in rollouts
**Solution**: Try different difficulty metric or check rollout generation

### Issue: Training not improving

**Cause**: Curriculum stages too similar
**Solution**: Use more buckets (5-7) or different metric (try `variance` or `pass_at_k`)

## Best Practices

1. **Start with 3 stages**: Easy to manage and debug
2. **Validate each stage**: Check val performance before moving to next stage
3. **Use mean_reward**: Good default metric for most cases
4. **Monitor transitions**: Watch for performance drops when increasing difficulty
5. **Keep validation sets separate**: Don't mix train data from different stages
6. **Save checkpoints**: Keep model checkpoints between curriculum stages
7. **Analyze difficulty distribution**: Use example scripts to understand your data

## Next Steps

- See `BUCKETING_GUIDE.md` for detailed bucketing options
- See `QUICK_START_BUCKETING.md` for quick reference
- Check `example_bucketing_usage.py` for programmatic usage
