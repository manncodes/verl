# Difficulty Filtering and Bucketing Script

This script helps filter and bucket training problems based on their difficulty by:
1. Loading a model and dataset
2. Generating multiple responses per problem
3. Calculating IFeval-style rewards
4. Computing mean@k metrics
5. Bucketing problems by difficulty using various strategies

## Features

### Difficulty Metrics Calculated

For each problem, the script calculates:
- **Mean Reward**: Average reward across all generated samples
- **Standard Deviation**: Variance in rewards (indicates consistency)
- **Max/Min Reward**: Best and worst rewards achieved
- **Pass Rate**: Percentage of samples that passed (reward > 0)
- **Mean@k**: Average of top-k rewards (k=1,3,5,10)
- **Difficulty Score**: Combined metric (lower mean reward + variance penalty)

### Bucketing Strategies

The script supports multiple strategies for bucketing problems:

#### 1. Percentile-based (default)
Buckets problems into quartiles based on difficulty score:
- **Easy**: Bottom 25% (easiest)
- **Medium**: 25-50%
- **Hard**: 50-75%
- **Very Hard**: Top 25% (hardest)

#### 2. Pass Rate-based
Buckets based on how often the model solves the problem:
- **Easy**: Pass rate ≥ 80%
- **Medium**: Pass rate 50-80%
- **Hard**: Pass rate 20-50%
- **Very Hard**: Pass rate < 20%

#### 3. Mean Reward-based
Buckets based on absolute reward thresholds:
- **Easy**: Mean reward ≥ 0.75
- **Medium**: Mean reward 0.50-0.75
- **Hard**: Mean reward 0.25-0.50
- **Very Hard**: Mean reward < 0.25

#### 4. Adaptive Clustering
Uses K-means clustering on multiple features (mean reward, std, pass rate, difficulty score) to automatically discover difficulty clusters.

## Usage

### Basic Usage

```bash
python scripts/filter_difficulty.py \
    --model_path /path/to/model \
    --data_path /path/to/dataset.parquet \
    --output_dir ./difficulty_results \
    --num_samples 5 \
    --bucketing_strategy percentile
```

### Advanced Options

```bash
python scripts/filter_difficulty.py \
    --model_path /path/to/model \
    --data_path /path/to/dataset.parquet \
    --output_dir ./difficulty_results \
    --num_samples 10 \
    --bucketing_strategy adaptive \
    --max_new_tokens 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --device cuda
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | Required | Path to the model for evaluation |
| `--data_path` | str | Required | Path to dataset (parquet format) |
| `--output_dir` | str | `./difficulty_results` | Directory to save results |
| `--num_samples` | int | 5 | Number of samples per problem for mean@k |
| `--bucketing_strategy` | str | `percentile` | Strategy: `percentile`, `pass_rate`, `mean_reward`, `adaptive` |
| `--max_new_tokens` | int | 512 | Maximum tokens to generate |
| `--temperature` | float | 0.7 | Sampling temperature |
| `--top_p` | float | 0.9 | Nucleus sampling parameter |
| `--device` | str | `cuda`/`cpu` | Device to run on |

## Input Data Format

The script expects a parquet file with the following structure:

```python
{
    "data_source": "openai/gsm8k",  # Dataset identifier for reward function
    "prompt": [                      # Chat format or string
        {"role": "user", "content": "Question text..."}
    ],
    "reward_model": {
        "style": "rule",
        "ground_truth": "42"         # Expected answer
    },
    "extra_info": {                  # Optional metadata
        "split": "train",
        "index": 0
    }
}
```

## Output Files

The script generates several output files in the specified output directory:

### 1. Detailed Metrics JSON
`difficulty_metrics_{strategy}.json`

Contains full metrics for each problem:
```json
[
  {
    "problem_id": "0",
    "data_source": "openai/gsm8k",
    "mean_reward": 0.85,
    "std_reward": 0.12,
    "max_reward": 1.0,
    "min_reward": 0.6,
    "pass_rate": 80.0,
    "mean_at_k": {
      "1": 1.0,
      "3": 0.93,
      "5": 0.85
    },
    "difficulty_score": 0.186,
    "difficulty_bucket": "easy",
    "raw_rewards": [1.0, 0.8, 0.9, 0.6, 1.0]
  }
]
```

### 2. Metrics Parquet
`difficulty_metrics_{strategy}.parquet`

Same data in parquet format for easy filtering and analysis.

### 3. Summary Statistics
`summary_{strategy}.json`

Overall statistics and bucket distribution:
```json
{
  "total_problems": 100,
  "bucketing_strategy": "percentile",
  "bucket_distribution": {
    "easy": 25,
    "medium": 25,
    "hard": 25,
    "very_hard": 25
  },
  "bucket_statistics": {
    "easy": {
      "count": 25,
      "mean_reward": 0.92,
      "mean_pass_rate": 95.0,
      "mean_difficulty_score": 0.15
    }
  }
}
```

### 4. Filtered Datasets
`data_{strategy}_{bucket}.parquet`

Separate parquet files for each difficulty bucket containing the original data filtered to that bucket. These can be used directly for training:
- `data_percentile_easy.parquet`
- `data_percentile_medium.parquet`
- `data_percentile_hard.parquet`
- `data_percentile_very_hard.parquet`

## Example Workflow

### 1. Filter GSM8K problems by difficulty
```bash
python scripts/filter_difficulty.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --data_path data/gsm8k_train.parquet \
    --output_dir ./gsm8k_difficulty \
    --num_samples 5 \
    --bucketing_strategy percentile
```

### 2. Use adaptive clustering for code problems
```bash
python scripts/filter_difficulty.py \
    --model_path deepseek-ai/deepseek-coder-7b-instruct \
    --data_path data/code_contests.parquet \
    --output_dir ./code_difficulty \
    --num_samples 10 \
    --bucketing_strategy adaptive \
    --max_new_tokens 2048
```

### 3. Train on specific difficulty buckets
After filtering, you can use the bucketed datasets for curriculum learning:

```bash
# Start with easy problems
python train.py --data_path ./gsm8k_difficulty/data_percentile_easy.parquet

# Progress to medium
python train.py --data_path ./gsm8k_difficulty/data_percentile_medium.parquet

# Then hard
python train.py --data_path ./gsm8k_difficulty/data_percentile_hard.parquet
```

## Supported Datasets

The script uses verl's `default_compute_score` function, which supports:
- **Math**: `openai/gsm8k`, `lighteval/MATH`, `hiyouga/geometry3k`
- **Code**: `codecontests`, `apps`, `codeforces`, `taco`
- **Search/QA**: `searchR1_nq`, `searchR1_hotpotqa`, `searchR1_popqa`, `searchR1_triviaqa`
- **Custom**: Define your own reward function in the data

## Advanced: Custom Reward Functions

If your dataset uses a custom reward function, you can:

1. Register it in `verl/utils/reward_score/__init__.py`
2. Or provide a `compute_score` function in your data's extra_info

## Tips for Best Results

1. **Number of Samples**: Use more samples (10-20) for more stable metrics, but this increases computation time.
2. **Temperature**: Higher temperature (0.8-1.0) gives more diverse responses and better estimates of pass rate.
3. **Bucketing Strategy**:
   - Use `percentile` for balanced buckets across datasets
   - Use `pass_rate` when you care about absolute performance
   - Use `adaptive` for automatic discovery of natural difficulty clusters
4. **Curriculum Learning**: Train on easy→medium→hard for better convergence

## Requirements

- torch
- transformers
- pandas
- numpy
- tqdm
- scikit-learn (optional, for adaptive clustering)

## Troubleshooting

**Issue**: Out of memory errors
- Reduce `num_samples`
- Reduce `max_new_tokens`
- Use a smaller model
- Process data in batches (modify script)

**Issue**: Reward computation errors
- Check that `data_source` matches supported datasets
- Verify ground_truth format matches expected format for your dataset

**Issue**: All problems in same bucket
- Try a different bucketing strategy
- Check if rewards are being computed correctly
- Increase `num_samples` for more accurate estimates
