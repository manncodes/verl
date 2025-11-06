# Generate Rollouts Script

**Simple script to generate rollouts (prompts + responses) without reward calculation.**

Just generate and save - figure out rewards later!

## 🚀 Quick Start

```bash
# With VLLM (fast!)
python scripts/generate_rollouts.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path data.parquet \
    --output_path rollouts.parquet \
    --num_samples 8 \
    --use_vllm \
    --vllm_base_url http://localhost:8000/v1

# With HuggingFace model (no server needed)
python scripts/generate_rollouts.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path data.parquet \
    --output_path rollouts.parquet \
    --num_samples 8
```

## Why This Script?

**Separation of concerns:**
1. **Generation is slow** → Save rollouts once
2. **Reward calculation is fast** → Experiment with different reward functions later
3. **Simpler workflow** → No complex config files needed

## What It Does

1. ✅ Loads your data
2. ✅ Generates N responses per prompt
3. ✅ Saves everything to parquet/jsonl
4. ❌ NO reward calculation

**That's it!** Calculate rewards separately when you're ready.

## Features

### Async VLLM Support
- **Concurrent requests** - all samples generated in parallel
- **Semaphore control** - prevent overwhelming server
- **Massive speedup** - 20-30x faster than sequential

### Simple Output Format

**Parquet/JSONL with columns:**
```
prompt          - The input prompt (text)
response        - Generated response (text)
sample_idx      - Which sample (0 to num_samples-1)
data_idx        - Index in original data
temperature     - Generation temperature used
top_p           - Top-p value used
max_tokens      - Max tokens generated
original_*      - All original data columns (prefixed)
```

### Flexible Input

Handles different prompt formats:
- Plain text strings
- Chat format (list of messages)
- Any column name (specify with `--prompt_key`)

## Usage

### Basic Example

```bash
python scripts/generate_rollouts.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path data.parquet \
    --output_path rollouts.parquet \
    --num_samples 8
```

### With VLLM Server

```bash
# Terminal 1: Start VLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --port 8000

# Terminal 2: Generate rollouts
python scripts/generate_rollouts.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path data.parquet \
    --output_path rollouts.parquet \
    --num_samples 8 \
    --use_vllm \
    --vllm_base_url http://localhost:8000/v1 \
    --vllm_max_concurrent 100
```

### Advanced Options

```bash
python scripts/generate_rollouts.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path data.parquet \
    --output_path rollouts.jsonl \
    --output_format jsonl \
    --num_samples 16 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_tokens 2048 \
    --batch_size 16 \
    --use_vllm \
    --vllm_max_concurrent 200
```

## Command-Line Arguments

### Required
- `--model_path` - Model path or name
- `--data_path` - Input data (parquet)

### VLLM Options
- `--use_vllm` - Use VLLM server (much faster!)
- `--vllm_base_url` - VLLM server URL (default: http://localhost:8000/v1)
- `--vllm_model_name` - Model name on server (default: same as model_path)
- `--vllm_max_concurrent` - Max concurrent requests (default: 100)

### Generation Options
- `--num_samples` - Responses per prompt (default: 8)
- `--temperature` - Sampling temperature (default: 0.7)
- `--top_p` - Nucleus sampling (default: 0.9)
- `--max_tokens` - Max tokens to generate (default: 512)
- `--batch_size` - Batch size (default: 8)

### Data Options
- `--prompt_key` - Column name for prompts (default: "prompt")
- `--trust_remote_code` - Trust remote code in model

### Output Options
- `--output_path` - Output file path (default: ./rollouts.parquet)
- `--output_format` - Format: parquet or jsonl (default: parquet)

## Output Format

### Parquet Example

```python
import pandas as pd

# Load rollouts
df = pd.read_parquet("rollouts.parquet")

# View structure
print(df.head())
#    prompt                 response        sample_idx  data_idx  temperature  ...
# 0  "What is 2+2?"        "2+2 equals 4"           0         0          0.7  ...
# 1  "What is 2+2?"        "The answer is 4"        1         0          0.7  ...
# 2  "What is 2+2?"        "4"                      2         0          0.7  ...
# 3  "What is 3+3?"        "3+3 equals 6"           0         1          0.7  ...

# Group by prompt to see all responses
for prompt, group in df.groupby("data_idx"):
    print(f"\nPrompt {prompt}:")
    print(group[["sample_idx", "response"]])
```

### JSONL Example

```python
import json

# Load rollouts
with open("rollouts.jsonl") as f:
    rollouts = [json.loads(line) for line in f]

# Each line is one rollout
print(rollouts[0])
# {
#   "prompt": "What is 2+2?",
#   "response": "2+2 equals 4",
#   "sample_idx": 0,
#   "data_idx": 0,
#   "temperature": 0.7,
#   "top_p": 0.9,
#   "max_tokens": 512,
#   "original_data_source": "math",
#   ...
# }
```

## Calculate Rewards Later

Once you have rollouts, calculate rewards separately:

```python
import pandas as pd
from verl.utils.reward_score import default_compute_score

# Load rollouts
df = pd.read_parquet("rollouts.parquet")

# Calculate rewards
rewards = []
for _, row in df.iterrows():
    reward = default_compute_score(
        data_source=row["original_data_source"],
        solution=row["response"],
        ground_truth=row["original_ground_truth"],
    )
    rewards.append(reward["score"])

df["reward"] = rewards

# Now you have rollouts + rewards!
df.to_parquet("rollouts_with_rewards.parquet")
```

## Performance

### With VLLM (Recommended)

For 1000 prompts × 8 samples = 8000 rollouts:

- **Sequential**: ~7 hours (8000 × 3s per request)
- **Async (100 concurrent)**: ~4-5 minutes (80-100x speedup!)

### Batch Processing

The script processes data in batches:
- Loads `batch_size` prompts at a time
- Generates `num_samples` responses per prompt
- Total concurrent: `batch_size × num_samples` requests

Example with `batch_size=8` and `num_samples=8`:
- 64 concurrent requests per batch
- With `max_concurrent=100`, limited to 100 at once

## Tips

1. **Use VLLM for speed** - 20-30x faster than HuggingFace
2. **Tune max_concurrent** - Higher = faster, but don't overwhelm server
3. **Save as parquet** - More efficient than jsonl for large datasets
4. **Generate once, analyze many times** - Reuse rollouts for different reward functions

## Workflow Example

```bash
# Step 1: Generate rollouts (slow, do once)
python scripts/generate_rollouts.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path data.parquet \
    --output_path rollouts.parquet \
    --num_samples 16 \
    --use_vllm

# Step 2: Calculate rewards (fast, do many times)
python analyze_rewards.py --rollouts rollouts.parquet --reward_fn ifeval
python analyze_rewards.py --rollouts rollouts.parquet --reward_fn math
python analyze_rewards.py --rollouts rollouts.parquet --reward_fn custom

# Step 3: Filter by difficulty (using the difficulty filter script)
python scripts/filter_difficulty.py \
    --data_path rollouts_with_rewards.parquet \
    --bucketing_strategy percentile
```

## No Config Files Needed!

Everything is controlled via command-line arguments. Simple and straightforward.

```bash
python scripts/generate_rollouts.py --help
```

That's it! 🚀
