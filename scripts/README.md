# VERL Scripts

This directory contains standalone scripts for various VERL tasks.

## Difficulty Filtering & Analysis

Three scripts for filtering and bucketing training problems by difficulty:

### 1. `generate_rollouts.py` - Generate Rollouts Only
Generate multiple responses per problem without reward calculation.

**Use case**: Separate generation (slow) from analysis (fast). Generate once, analyze many times.

```bash
python scripts/generate_rollouts.py \
    --use_vllm \
    --vllm_url http://localhost:8000/v1 \
    --data_path data.parquet \
    --output_path rollouts.parquet \
    --num_samples 10
```

📖 [Full Documentation](README_generate_rollouts.md)

### 2. `analyze_rollouts.py` - Reward Calculation & Difficulty Filtering
Calculate rewards and perform difficulty bucketing on existing rollouts.

**Use case**: Analyze pre-generated rollouts. Try different bucketing strategies without regenerating.

```bash
python scripts/analyze_rollouts.py \
    --rollouts rollouts.parquet \
    --model_path /path/to/model \
    --output_dir ./results \
    --bucketing_strategy percentile
```

📖 [Full Documentation](README_analyze_rollouts.md)

### 3. `filter_difficulty.py` - Complete Pipeline
All-in-one script that generates rollouts, calculates rewards, and performs difficulty bucketing.

**Use case**: Quick one-shot analysis. Good for small datasets or testing.

```bash
python scripts/filter_difficulty.py \
    --use_vllm \
    --vllm_url http://localhost:8000/v1 \
    --model_path /path/to/model \
    --data_path data.parquet \
    --output_dir ./results \
    --num_samples 5
```

📖 [Full Documentation](README_difficulty_filter.md)

## Which Script Should I Use?

### Recommended Workflow (Separation of Concerns)

**For production and large datasets**, use the two-step approach:

```bash
# Step 1: Generate rollouts once (slow, GPU-intensive)
python scripts/generate_rollouts.py \
    --use_vllm \
    --data_path data.parquet \
    --output_path rollouts.parquet

# Step 2: Analyze many times (fast, CPU-only)
python scripts/analyze_rollouts.py \
    --rollouts rollouts.parquet \
    --output_dir ./results_percentile \
    --bucketing_strategy percentile

python scripts/analyze_rollouts.py \
    --rollouts rollouts.parquet \
    --output_dir ./results_pass_rate \
    --bucketing_strategy pass_rate
```

**Benefits**:
- Generate once, analyze many times
- Save compute and time
- Experiment with different bucketing strategies
- Keep rollouts for future analysis

### Quick Testing (All-in-One)

**For small datasets or quick tests**, use the complete pipeline:

```bash
python scripts/filter_difficulty.py \
    --use_vllm \
    --model_path /path/to/model \
    --data_path small_dataset.parquet \
    --output_dir ./quick_test
```

**Benefits**:
- Single command
- No intermediate files
- Good for testing and small experiments

## Complete Workflow Example

See [example_complete_workflow.sh](example_complete_workflow.sh) for a complete end-to-end example.

```bash
bash scripts/example_complete_workflow.sh
```

This script demonstrates:
1. Generating rollouts with VLLM
2. Analyzing with all 4 bucketing strategies
3. Comparing results across strategies

## Bucketing Strategies

All scripts support multiple difficulty bucketing strategies:

- **percentile**: Bucket by percentile of mean@5 (default)
- **pass_rate**: Bucket by pass rate (% of samples passing)
- **mean_reward**: Bucket by mean reward across samples
- **adaptive**: K-means clustering on multiple features

## Output Files

All analysis scripts produce:

- `rollouts_with_rewards.parquet`: All rollouts with reward scores
- `problem_metrics.json`: Per-problem metrics (mean@k, pass rate, etc.)
- `problem_buckets.json`: Difficulty bucket assignments
- `rollouts_{bucket}.parquet`: Separate datasets per difficulty bucket
- `summary.json`: Overall statistics

## Requirements

```bash
# Core dependencies (usually already installed with verl)
pip install pandas pyarrow torch transformers

# For VLLM generation
pip install openai  # For VLLM client

# For adaptive bucketing
pip install scikit-learn
```

## Performance Tips

### Use VLLM for Generation
VLLM is 10-50x faster than HuggingFace for generation:

```bash
# Start VLLM server first
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Then use --use_vllm flag
python scripts/generate_rollouts.py --use_vllm --vllm_base_url http://localhost:8000/v1 ...
```

### Multi-VLLM Load Balancing (Even Faster!)
Run multiple VLLM instances and load balance across them for 1.5-3x better throughput:

```bash
# Launch 4 instances with TP=2 each (8 GPUs total)
bash scripts/launch_multi_vllm.sh Qwen/Qwen2.5-7B-Instruct 4 2

# Generate with load balancing (comma-separated URLs)
python scripts/generate_rollouts.py \
    --use_vllm \
    --vllm_base_url "http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1" \
    --vllm_max_concurrent 300 \
    ...
```

📖 See [Multi-VLLM Guide](README_multi_vllm.md) for details.

### Async VLLM for Massive Speedup
Use `--max_concurrent` to control concurrent requests:

```bash
python scripts/generate_rollouts.py \
    --use_vllm \
    --max_concurrent 100  # Process 100 prompts concurrently
```

### Generate Once, Analyze Many Times
The two-step workflow saves significant compute:

```bash
# Generate once (1 hour)
python scripts/generate_rollouts.py ... --output_path rollouts.parquet

# Analyze many times (1-2 minutes each)
python scripts/analyze_rollouts.py --rollouts rollouts.parquet --bucketing_strategy percentile ...
python scripts/analyze_rollouts.py --rollouts rollouts.parquet --bucketing_strategy pass_rate ...
python scripts/analyze_rollouts.py --rollouts rollouts.parquet --bucketing_strategy adaptive ...
```

## Troubleshooting

### VLLM Connection Errors

Make sure VLLM server is running:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Test connection:
```bash
curl http://localhost:8000/v1/models
```

### Out of Memory

Reduce batch size or number of concurrent requests:
```bash
python scripts/generate_rollouts.py --max_concurrent 20  # Default is 100
```

### Slow Generation

Make sure you're using VLLM (`--use_vllm` flag) and have sufficient `--max_concurrent`:
```bash
python scripts/generate_rollouts.py --use_vllm --max_concurrent 100
```

## Contributing

When adding new scripts:
1. Make them standalone (no config files required)
2. Add comprehensive docstrings
3. Create a README_{script_name}.md
4. Add entry to this main README
5. Include example usage in comments

## See Also

- [Difficulty Filter Documentation](README_difficulty_filter.md)
- [Generate Rollouts Documentation](README_generate_rollouts.md)
- [Analyze Rollouts Documentation](README_analyze_rollouts.md)
- [Complete Workflow Example](example_complete_workflow.sh)
