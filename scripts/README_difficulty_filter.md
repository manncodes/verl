# Difficulty Filtering and Bucketing Script

This script helps filter and bucket training problems based on their difficulty by:
1. Loading a model and dataset using verl's infrastructure
2. Generating multiple responses per problem
3. Calculating rewards using verl's reward managers (same as in PPO training)
4. Computing mean@k metrics
5. Bucketing problems by difficulty using various strategies

## Key Features

### Built on verl's Infrastructure

This script **reuses the same code** from verl's PPO trainer:
- **`load_reward_manager()`**: Same reward calculation as training
- **`RLHFDataset`**: Same data loading as training
- **`DataProto`**: Same data structures as training
- **`dump_generations()`**: Same generation dumping pattern as validation

This ensures **consistency** between difficulty filtering and actual training.

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

### Configuration-based (Recommended)

Create a configuration file (YAML) with your settings:

```yaml
# config/my_filter.yaml
model:
  path: /path/to/model

data:
  path: /path/to/data.parquet
  batch_size: 8
  max_prompt_length: 1024

generation:
  max_new_tokens: 1024
  temperature: 0.8
  do_sample: true

reward_model:
  enable: false  # true if using model-based RM
  reward_manager: naive

output_dir: ./difficulty_results
num_samples: 10
bucketing_strategy: percentile
```

Then run:

```bash
python scripts/filter_difficulty.py \
    --config-path=/path/to/config \
    --config-name=my_filter
```

### Command-line Override

You can override any config value from the command line:

```bash
python scripts/filter_difficulty.py \
    model.path=/path/to/model \
    data.path=/path/to/data.parquet \
    output_dir=./results \
    num_samples=5 \
    bucketing_strategy=adaptive
```

### Using Existing verl Configs

You can reuse configurations from verl's training:

```bash
# Copy and modify an existing PPO config
cp verl/trainer/config/ppo_trainer.yaml scripts/config/my_difficulty_filter.yaml

# Edit to set paths and parameters
vim scripts/config/my_difficulty_filter.yaml

# Run
python scripts/filter_difficulty.py --config-name=my_difficulty_filter
```

## Configuration Reference

### Model Configuration

```yaml
model:
  path: /path/to/model  # Required
  use_shm: false        # Use shared memory for faster loading
```

### Data Configuration

```yaml
data:
  path: /path/to/data.parquet  # Required
  batch_size: 4                # Batch size for processing
  prompt_key: prompt           # Key for prompt in data
  max_prompt_length: 1024      # Max tokens in prompt
  truncation: right            # Truncation direction
  trust_remote_code: false     # Trust remote code in model
  reward_fn_key: reward_model  # Key for reward model info
```

### Generation Configuration

```yaml
generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true
```

### Reward Model Configuration

```yaml
reward_model:
  enable: false           # Set true for model-based RM
  reward_manager: naive   # naive, batch, prime, dapo
  reward_kwargs: {}       # Additional kwargs for reward manager
```

### Output Configuration

```yaml
output_dir: ./difficulty_results
num_samples: 5              # Samples per problem for mean@k
bucketing_strategy: percentile  # percentile, pass_rate, mean_reward, adaptive
```

## Input Data Format

The script expects a parquet file with the same format as verl's training data:

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

**The data format is identical to verl's PPO training data format!**

## Output Files

The script generates several output files in the specified output directory:

### 1. Detailed Metrics JSON
`difficulty_metrics_{strategy}.json`

Contains full metrics for each problem:
```json
[
  {
    "problem_id": "uuid-here",
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

### 4. Generation Dumps (Optional)
`generations_0.jsonl`

If you want to save all generated samples, the script includes a `dump_generations()` method that follows verl's validation dump pattern.

## Example Workflows

### 1. Filter GSM8K problems by difficulty

```yaml
# config/gsm8k_filter.yaml
model:
  path: meta-llama/Llama-3.1-8B-Instruct

data:
  path: data/gsm8k_train.parquet
  batch_size: 8

generation:
  max_new_tokens: 512

output_dir: ./gsm8k_difficulty
num_samples: 5
bucketing_strategy: percentile
```

```bash
python scripts/filter_difficulty.py --config-name=gsm8k_filter
```

### 2. Use adaptive clustering for code problems

```bash
python scripts/filter_difficulty.py \
    model.path=deepseek-ai/deepseek-coder-7b-instruct \
    data.path=data/code_contests.parquet \
    output_dir=./code_difficulty \
    num_samples=10 \
    bucketing_strategy=adaptive \
    generation.max_new_tokens=2048
```

### 3. Curriculum learning workflow

After filtering, you can create a custom curriculum:

```python
import pandas as pd

# Load difficulty metrics
metrics = pd.read_parquet('./difficulty_results/difficulty_metrics_percentile.parquet')

# Filter to specific difficulty
easy_ids = metrics[metrics['difficulty_bucket'] == 'easy']['problem_id'].tolist()
hard_ids = metrics[metrics['difficulty_bucket'] == 'hard']['problem_id'].tolist()

# Load original data and filter
data = pd.read_parquet('data/gsm8k_train.parquet')
# ... filter by IDs and create curriculum
```

## Supported Datasets

The script uses verl's `default_compute_score` function, which supports:
- **Math**: `openai/gsm8k`, `lighteval/MATH`, `hiyouga/geometry3k`
- **Code**: `codecontests`, `apps`, `codeforces`, `taco`
- **Search/QA**: `searchR1_nq`, `searchR1_hotpotqa`, `searchR1_popqa`, `searchR1_triviaqa`
- **Custom**: Define your own reward function via config

## Advanced: Custom Reward Functions

If your dataset uses a custom reward function, you can specify it in the config:

```yaml
custom_reward_function:
  path: /path/to/my_reward_fn.py
  name: compute_score
  reward_kwargs:
    my_param: value
```

This is the **same mechanism** used in verl's PPO training!

## Tips for Best Results

1. **Number of Samples**: Use more samples (10-20) for more stable metrics, but this increases computation time.
2. **Temperature**: Higher temperature (0.8-1.0) gives more diverse responses and better estimates of pass rate.
3. **Bucketing Strategy**:
   - Use `percentile` for balanced buckets across datasets
   - Use `pass_rate` when you care about absolute performance
   - Use `adaptive` for automatic discovery of natural difficulty clusters
4. **Curriculum Learning**: Train on easy→medium→hard for better convergence
5. **Batch Size**: Larger batches process faster but use more memory

## Troubleshooting

**Issue**: Out of memory errors
- Reduce `data.batch_size`
- Reduce `generation.max_new_tokens`
- Reduce `num_samples`

**Issue**: Reward computation errors
- Check that `data_source` matches supported datasets
- Verify ground_truth format matches expected format for your dataset
- Check reward_model configuration

**Issue**: Script doesn't find config file
- Use `--config-path` to specify directory
- Use `--config-name` for the file name (without .yaml)
- Or use command-line overrides

## Comparison with Training

This script uses the **exact same** reward calculation as verl's PPO training:
- Same `load_reward_manager()` function
- Same `RLHFDataset` for data loading
- Same `DataProto` data structures
- Same reward functions (rule-based or model-based)

This ensures that difficulty estimates are **consistent** with actual training rewards!

## Requirements

All dependencies are the same as verl:
- torch
- transformers
- ray
- hydra-core
- omegaconf
- pandas
- numpy
- tqdm
- scikit-learn (optional, for adaptive clustering)
