# Curriculum Learning for VERL - Complete Solution

This directory contains a complete curriculum learning pipeline for organizing and training on rollouts by difficulty.

## 🚀 Quick Start (One Command)

```bash
# Complete pipeline: bucket rollouts + preprocess for training
./run_curriculum_pipeline.sh -i ./rollouts -n 3 -c

# Output:
# - Bucketed JSONLs (easy/medium/hard)
# - Training parquet files with train/val splits
# - Progressive combined datasets for curriculum training
```

## 📁 Files Overview

### Core Scripts

1. **`bucket_rollouts_by_difficulty.py`** - Organize rollouts by difficulty
   - Multiple metrics: mean_reward, pass@k, variance, etc.
   - Flexible bucketing: percentiles or thresholds
   - Works without numpy (pure Python fallback)

2. **`preprocess_bucketed_rollouts.py`** - Convert to training format
   - Extracts prompts and ground truth from rollouts
   - Creates train/val splits
   - Supports quality filtering
   - Combines stages for progressive training

3. **`run_curriculum_pipeline.sh`** - End-to-end automation
   - Runs bucketing + preprocessing
   - Creates all necessary datasets
   - Configurable stages and metrics

### Documentation

- **`QUICK_START_BUCKETING.md`** - Quick reference for bucketing
- **`BUCKETING_GUIDE.md`** - Detailed bucketing documentation
- **`CURRICULUM_PIPELINE_GUIDE.md`** - Complete workflow guide
- **`example_bucketing_usage.py`** - Programmatic usage examples

## 🎯 Use Cases

### 1. Standard 3-Stage Curriculum

```bash
# Automatic pipeline
./run_curriculum_pipeline.sh -i ./rollouts

# Manual steps
python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward -n 3
python preprocess_bucketed_rollouts.py -i ./bucketed -o ./preprocessed

# Train progressively
python train.py --data ./preprocessed/easy_train.parquet
python train.py --data ./preprocessed/medium_train.parquet
python train.py --data ./preprocessed/hard_train.parquet
```

### 2. Fine-Grained 5-Stage Curriculum

```bash
./run_curriculum_pipeline.sh \
    -i ./rollouts \
    -n 5 \
    --stages "warmup basic intermediate advanced expert" \
    -c
```

### 3. Code Generation (Pass@K)

```bash
./run_curriculum_pipeline.sh \
    -i ./code_rollouts \
    -m pass_at_k \
    -f passing_only \
    -c
```

### 4. Progressive Training (Cumulative)

```bash
# Creates stage1 (easy), stage2 (easy+medium), stage3 (all)
./run_curriculum_pipeline.sh -i ./rollouts -n 3 -c

# Train incrementally
python train.py --data ./curriculum_output/progressive/stage1_train.parquet --checkpoint ckpt1
python train.py --data ./curriculum_output/progressive/stage2_train.parquet --load ckpt1 --checkpoint ckpt2
python train.py --data ./curriculum_output/progressive/stage3_train.parquet --load ckpt2 --checkpoint final
```

## 📊 Difficulty Metrics

Choose the right metric for your use case:

| Metric | Use For | Interpretation |
|--------|---------|----------------|
| `mean_reward` | General purpose (default) | Higher = easier |
| `pass_at_k` | Code generation, any-correct tasks | Higher = easier |
| `max_reward` | Optimistic difficulty | Higher = easier |
| `variance` | Finding inconsistent examples | Higher = harder |
| `success_rate` | Binary pass/fail tasks | Higher = easier |

**Example with different metrics:**

```bash
# General instruction following
./run_curriculum_pipeline.sh -i ./rollouts -m mean_reward

# Code generation
./run_curriculum_pipeline.sh -i ./rollouts -m pass_at_k

# Find inconsistent examples
./run_curriculum_pipeline.sh -i ./rollouts -m variance
```

## 🔍 Understanding Your Rollouts

Test difficulty metrics on sample data:

```bash
python example_bucketing_usage.py 6
```

Output shows all metrics for your rollout structure:
```
Rollout reward scores: [0, 1.25, 0, 0, 0, 1.25, 0, 0]

Difficulty metrics:
  mean_reward: 0.3125    # Average success
  max_reward: 1.2500     # Best response
  pass_at_k: 0.2500      # 25% success rate
  variance: 0.3348       # Consistency measure
```

## 📦 Output Structure

After running the pipeline:

```
curriculum_output/
├── bucketed/              # Bucketed rollout JSONLs
│   ├── easy.jsonl         # Easiest examples
│   ├── medium.jsonl
│   └── hard.jsonl         # Hardest examples
│
├── preprocessed/          # Individual stage datasets
│   ├── easy_train.parquet
│   ├── easy_val.parquet
│   ├── medium_train.parquet
│   ├── medium_val.parquet
│   ├── hard_train.parquet
│   └── hard_val.parquet
│
└── progressive/           # Cumulative datasets (if -c used)
    ├── stage1_train.parquet      # Just easy
    ├── stage1_val.parquet
    ├── stage2_train.parquet      # Easy + medium
    ├── stage2_val.parquet
    ├── stage3_train.parquet      # All stages
    └── stage3_val.parquet
```

## 🎓 Training Strategies

### Strategy 1: Separate Stage Training
Train on each difficulty level independently:

```bash
for stage in easy medium hard; do
    python train.py --data ./preprocessed/${stage}_train.parquet
done
```

### Strategy 2: Progressive Training
Start easy, gradually increase difficulty:

```bash
python train.py --data ./progressive/stage1_train.parquet --checkpoint ckpt1
python train.py --data ./progressive/stage2_train.parquet --load ckpt1 --checkpoint ckpt2
python train.py --data ./progressive/stage3_train.parquet --load ckpt2 --checkpoint final
```

### Strategy 3: Mixed Training
Combine specific stages:

```bash
python preprocess_bucketed_rollouts.py \
    -i ./bucketed \
    -o ./mixed \
    --buckets easy medium \
    --combine \
    --output-name easy_medium_mix

python train.py --data ./mixed/easy_medium_mix_train.parquet
```

## ⚙️ Advanced Options

### Custom Difficulty Thresholds

```bash
# Use absolute score thresholds instead of percentiles
python bucket_rollouts_by_difficulty.py \
    -i ./rollouts \
    -o ./bucketed \
    -m mean_reward \
    --thresholds 0.3 0.7 \
    --bucket-names low medium high
```

### Filter Low-Quality Examples

```bash
# Only include rollouts with at least one good response
./run_curriculum_pipeline.sh -i ./rollouts -f best_only

# Only include rollouts with passing responses (score >= 0.5)
./run_curriculum_pipeline.sh -i ./rollouts -f passing_only
```

### Custom Stage Names

```bash
./run_curriculum_pipeline.sh \
    -i ./rollouts \
    -n 4 \
    --stages "beginner intermediate advanced expert"
```

## 🔧 Integration with Existing Pipeline

This curriculum pipeline integrates seamlessly with your existing verl pipeline:

```bash
# 1. Your existing rollout generation
python generate_rollouts.py --config config.yaml
# → Outputs: ./rollouts/*.jsonl

# 2. Our curriculum bucketing
./run_curriculum_pipeline.sh -i ./rollouts -c
# → Outputs: ./curriculum_output/preprocessed/*.parquet

# 3. Your existing training (now with curriculum)
python train.py --data ./curriculum_output/progressive/stage1_train.parquet
```

The preprocessed parquet files have the same format as your existing `preprocess_ifeval_data.py` output:

```python
{
    "data_source": "bucketed_curriculum",
    "prompt": [...],  # Same format as ifeval
    "ability": "instruction_following",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "instruction_id_list": [...],
            "kwargs": [...]
        }
    },
    "extra_info": {
        "curriculum_stage": "easy",  # NEW: stage metadata
        "difficulty_metrics": {...},  # NEW: difficulty stats
        # ... rest same as ifeval
    }
}
```

## 📈 Monitoring Progress

### Evaluate on Each Difficulty Level

```bash
# Test how well your model handles each difficulty
for stage in easy medium hard; do
    echo "Evaluating on $stage..."
    python evaluate.py --data ./preprocessed/${stage}_val.parquet
done
```

### Track Curriculum Performance

Monitor improvement across stages:

```python
# Add to your training script
from pathlib import Path

stages = ['easy', 'medium', 'hard']
results = {}

for stage in stages:
    val_data = f'./preprocessed/{stage}_val.parquet'
    results[stage] = evaluate(model, val_data)
    print(f"{stage}: {results[stage]}")
```

## 🐛 Troubleshooting

### All rollouts in one bucket?
- **Cause**: Limited reward variance
- **Fix**: Try different metric (`variance`, `pass_at_k`) or check rollout generation

### Empty buckets after preprocessing?
- **Cause**: Filter too strict
- **Fix**: Use `--filter all` or lower quality threshold

### Pipeline fails with numpy error?
- **Cause**: numpy not installed
- **Fix**: Either install numpy or ignore warning (script works without it)

## 📚 Documentation Index

- **Quick Start**: This file + `QUICK_START_BUCKETING.md`
- **Detailed Bucketing**: `BUCKETING_GUIDE.md`
- **Complete Workflow**: `CURRICULUM_PIPELINE_GUIDE.md`
- **Examples**: `example_bucketing_usage.py`

## 🤝 Example Workflow

Complete example from rollouts to trained model:

```bash
# 1. Generate rollouts (your existing pipeline)
python generate_rollouts.py --config my_config.yaml
# → ./my_rollouts/*.jsonl

# 2. Run curriculum pipeline
./run_curriculum_pipeline.sh \
    -i ./my_rollouts \
    -o ./my_curriculum \
    -n 3 \
    -c

# 3. Train progressively
python train.py \
    --data ./my_curriculum/progressive/stage1_train.parquet \
    --output stage1.ckpt

python train.py \
    --data ./my_curriculum/progressive/stage2_train.parquet \
    --load stage1.ckpt \
    --output stage2.ckpt

python train.py \
    --data ./my_curriculum/progressive/stage3_train.parquet \
    --load stage2.ckpt \
    --output final.ckpt

# 4. Evaluate
python evaluate.py \
    --data ./my_curriculum/preprocessed/hard_val.parquet \
    --checkpoint final.ckpt
```

## ✅ Next Steps

1. **Run the pipeline** on your rollouts:
   ```bash
   ./run_curriculum_pipeline.sh -i /path/to/your/rollouts
   ```

2. **Inspect the output** to understand difficulty distribution

3. **Choose a training strategy** (separate, progressive, or mixed)

4. **Start training** with your existing training scripts

5. **Monitor performance** across difficulty levels

6. **Iterate** on metrics and stage numbers as needed

## 📞 Questions?

- See detailed docs in `CURRICULUM_PIPELINE_GUIDE.md`
- Check examples in `example_bucketing_usage.py`
- Review bucketing options in `BUCKETING_GUIDE.md`
