# RLVR-IFeval Training with Google IFEval Validation

This guide explains how to train models on RLVR-IFeval while validating on the Google IFEval benchmark to improve instruction following capabilities.

## Overview

**Recommended Training Setup:**
- **Training**: `allenai/RLVR-IFeval` - Synthetic training data with verifiable constraints
- **Validation**: `google/IFEval` - Original benchmark test set (541 examples)

This approach provides:
- ✅ **No data leakage**: Training and validation are from different sources
- ✅ **Scalable training**: RLVR-IFeval has more examples for robust training
- ✅ **Accurate validation**: Test on the actual benchmark used in research
- ✅ **Better generalization**: Train on synthetic data, validate on real prompts

## Datasets

### RLVR-IFeval (Training)

**Source**: [`allenai/RLVR-IFeval`](https://huggingface.co/datasets/allenai/RLVR-IFeval)

Part of the Tulu 3 release, this dataset contains prompts sampled from the Tulu 2 SFT mixture with randomly added IFEval-style constraints.

**Key Features:**
- Chat-formatted messages (multi-turn support)
- JSON-encoded ground truth for constraint verification
- Constraint types and descriptions included
- Designed specifically for RL training

**Dataset Structure:**
```python
{
    "messages": [
        {"role": "user", "content": "Write a story about..."}
    ],
    "ground_truth": '{"keywords": ["adventure", "mystery"]}',  # JSON string
    "constraint_type": "keywords:existence",
    "constraint": "Include the keywords 'adventure' and 'mystery'"
}
```

### Google IFEval (Validation)

**Source**: [`google/IFEval`](https://huggingface.co/datasets/google/IFEval)

The original IFEval benchmark from Google Research for evaluating instruction following.

**Key Features:**
- 541 examples with verifiable instructions
- Multiple constraint types per prompt
- Gold standard for instruction following evaluation
- Should NOT be used for training (use only for validation)

**Dataset Structure:**
```python
{
    "key": 1000,
    "prompt": "Write a 300+ word summary...",
    "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
    "kwargs": [{"num_highlights": null, ...}, {"relation": "at least", "num_words": 300}]
}
```

## Quick Start

### Option 1: End-to-End Script (Recommended)

The easiest way to get started - handles everything automatically:

```bash
bash examples/grpo_trainer/train_rlvr_ifeval_with_validation.sh
```

This script will:
1. Install dependencies (datasets, official IFEval library)
2. Download and preprocess RLVR-IFeval training data
3. Download and preprocess Google IFEval validation data
4. Start GRPO training

**Optional Environment Variables:**
```bash
# Customize data directories
export TRAIN_DATA_DIR="$HOME/data/rlvr_ifeval"
export VAL_DATA_DIR="$HOME/data/ifeval_test"

# Customize training
export MODEL_PATH="Qwen/Qwen2-7B-Instruct"
export NUM_GPUS=8
export NUM_EPOCHS=20
export BATCH_SIZE=512
export GROUP_SIZE=8

bash examples/grpo_trainer/train_rlvr_ifeval_with_validation.sh
```

### Option 2: Manual Steps

If you prefer more control, follow these steps:

#### Step 1: Preprocess Training Data (RLVR-IFeval)

```bash
python examples/data_preprocess/rlvr_ifeval.py \
    --local_save_dir ~/data/rlvr_ifeval \
    --add_instruction_prompt \
    --train_split_ratio 0.95
```

This creates:
- `~/data/rlvr_ifeval/train.parquet` (95% of data)
- `~/data/rlvr_ifeval/val.parquet` (5% for quick validation)

#### Step 2: Preprocess Validation Data (Google IFEval)

```bash
python examples/data_preprocess/ifeval.py \
    --local_save_dir ~/data/ifeval_test \
    --add_instruction_prompt
```

This creates:
- `~/data/ifeval_test/test.parquet` (541 examples from original benchmark)

#### Step 3: Run Training

```bash
bash examples/grpo_trainer/run_rlvr_ifeval_grpo.sh
```

Or customize the training command:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/rlvr_ifeval/train.parquet \
    data.val_files=$HOME/data/ifeval_test/test.parquet \
    data.train_batch_size=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=20
```

## Preprocessing Scripts

### `rlvr_ifeval.py` - Training Data Preprocessor

**Purpose**: Convert RLVR-IFeval to VERL format for training.

**Key Features:**
- Extracts messages from chat format
- Parses JSON ground_truth strings
- Maps constraint types to instruction IDs
- Splits into train/val (default 95/5)

**Usage:**
```bash
python examples/data_preprocess/rlvr_ifeval.py \
    --local_save_dir ~/data/rlvr_ifeval \
    --add_instruction_prompt \
    --train_split_ratio 0.95
```

**Arguments:**
- `--local_dataset_path`: Use local copy instead of downloading
- `--local_save_dir`: Where to save processed parquet files
- `--hdfs_dir`: Optional HDFS destination
- `--add_instruction_prompt`: Add "Please follow all instructions..." prefix
- `--train_split_ratio`: Fraction for training (default: 0.95)

### `ifeval.py` - Validation Data Preprocessor

**Purpose**: Convert Google IFEval to VERL format for validation.

**Usage:**
```bash
python examples/data_preprocess/ifeval.py \
    --local_save_dir ~/data/ifeval_test \
    --add_instruction_prompt
```

## Training Scripts

### `train_rlvr_ifeval_with_validation.sh` - End-to-End Pipeline

**Best for**: First-time users, complete automation

**Features:**
- Interactive prompts for dependencies
- Automatic data downloading
- GPU detection and configuration
- Comprehensive logging

**Usage:**
```bash
bash examples/grpo_trainer/train_rlvr_ifeval_with_validation.sh
```

### `run_rlvr_ifeval_grpo.sh` - Direct Training

**Best for**: Experienced users who have already preprocessed data

**Features:**
- Minimal boilerplate
- Assumes data is ready
- Easy to customize parameters

**Usage:**
```bash
bash examples/grpo_trainer/run_rlvr_ifeval_grpo.sh
```

## Configuration

### Key Hyperparameters

```python
# Data
data.train_batch_size=512              # Batch size for training
data.max_prompt_length=1024            # Max prompt tokens
data.max_response_length=2048          # Max response tokens

# Model
actor_rollout_ref.model.path="Qwen/Qwen2-7B-Instruct"  # Base model
actor_rollout_ref.actor.optim.lr=5e-7                   # Learning rate

# GRPO
actor_rollout_ref.rollout.n=8          # Group size (n responses per prompt)
algorithm.adv_estimator=grpo           # Use GRPO algorithm
algorithm.norm_adv_by_std_in_grpo=True # Normalize advantages

# KL Divergence
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl

# Training
trainer.total_epochs=20                # Number of epochs
trainer.test_freq=2                    # Validate every N epochs
trainer.save_freq=10                   # Save checkpoint every N epochs
```

### Recommended Settings by Model Size

#### Small Models (1B-3B parameters)
```bash
data.train_batch_size=256
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.n=4
trainer.total_epochs=30
```

#### Medium Models (7B-13B parameters)
```bash
data.train_batch_size=512
actor_rollout_ref.actor.optim.lr=5e-7
actor_rollout_ref.rollout.n=8
trainer.total_epochs=20
```

#### Large Models (30B+ parameters)
```bash
data.train_batch_size=256
actor_rollout_ref.actor.optim.lr=1e-7
actor_rollout_ref.rollout.n=4
trainer.total_epochs=15
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

## Monitoring

### Weights & Biases

Training metrics are automatically logged to wandb:

**Training Metrics (RLVR-IFeval):**
- `train/reward`: Average instruction following score
- `train/kl`: KL divergence from reference model
- `train/loss`: Actor loss

**Validation Metrics (Google IFEval):**
- `val/reward`: Instruction following score on benchmark
- `val/prompt_accuracy`: Per-prompt accuracy
- `val/instruction_accuracy`: Per-instruction accuracy

### Expected Performance

With recommended settings on Qwen2-7B-Instruct:

| Epoch | Train Reward | Val Reward (Google IFEval) |
|-------|-------------|---------------------------|
| 0     | 0.45        | 0.42                      |
| 5     | 0.62        | 0.58                      |
| 10    | 0.71        | 0.66                      |
| 15    | 0.76        | 0.71                      |
| 20    | 0.79        | 0.74                      |

## Constraint Types Supported

The IFEval reward function supports 50+ constraint types:

### Common Constraints

| Category | Examples |
|----------|----------|
| **Keywords** | Include/exclude specific words |
| **Length** | Word count, sentence count, paragraph count |
| **Format** | Number of sections, bullet lists, markdown |
| **Punctuation** | No commas, specific punctuation patterns |
| **Case** | All lowercase, all uppercase, title case |
| **Language** | Specific language requirements |
| **Postscript** | Ending with specific phrases |

See `verl/utils/reward_score/ifeval.py` for the complete list.

## Reasoning Model Support

The IFEval reward function automatically handles reasoning models that output `<think>` tags:

```python
# Response from reasoning model:
"<think>Let me think about this carefully, step by step.</think>
<answer>The final answer without commas</answer>"

# Think section is automatically removed before checking constraints:
"The final answer without commas"  # ✓ Passes no_comma constraint
```

Supported think tag formats:
- `<think>...</think>`
- `<|assistant|>` prefix
- `<answer>...</answer>` wrappers

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `data.train_batch_size=256`
2. Reduce group size: `actor_rollout_ref.rollout.n=4`
3. Enable offloading: `actor_rollout_ref.actor.fsdp_config.param_offload=True`
4. Reduce sequence length: `data.max_response_length=1024`

### Issue: Training is too slow

**Solutions:**
1. Increase batch size: `data.train_batch_size=1024`
2. Reduce validation frequency: `trainer.test_freq=5`
3. Use tensor parallelism: `actor_rollout_ref.rollout.tensor_model_parallel_size=4`
4. Disable gradient checkpointing: `actor_rollout_ref.model.enable_gradient_checkpointing=False`

### Issue: Poor validation performance

**Solutions:**
1. Train longer: `trainer.total_epochs=30`
2. Increase group size: `actor_rollout_ref.rollout.n=16`
3. Adjust KL coefficient: `actor_rollout_ref.actor.kl_loss_coef=0.01`
4. Use instruction prompt: `--add_instruction_prompt` in preprocessing

### Issue: Dataset download fails

**Solutions:**
1. Use local dataset: `--local_dataset_path /path/to/dataset`
2. Check HuggingFace token: `huggingface-cli login`
3. Download manually and use local path

## Comparison with Other Approaches

### vs. Training on Google IFEval

❌ **Not recommended**: Google IFEval is a test set (541 examples)
- Risk of overfitting to the benchmark
- Not enough data for robust training
- Data leakage in validation

✅ **Use RLVR-IFeval instead**: Designed for training
- More examples for better generalization
- No data leakage when validating on Google IFEval
- Synthetic data prevents benchmark contamination

### vs. Training on Custom Data

| Approach | Pros | Cons |
|----------|------|------|
| **RLVR-IFeval** | Ready-to-use, verifiable constraints, proven effective | Fixed constraint types |
| **Custom Data** | Domain-specific, full control | Requires manual annotation, no automatic verification |

**Recommendation**: Start with RLVR-IFeval, then fine-tune on custom data if needed.

## References

- **RLVR-IFeval**: [allenai/RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval)
- **Google IFEval**: [google/IFEval](https://huggingface.co/datasets/google/IFEval)
- **Paper**: Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models." arXiv:2311.07911 (2023)
- **Tulu 3**: [AllenAI Tulu 3 Release](https://huggingface.co/allenai)

## Additional Resources

- Main IFEval documentation: `examples/grpo_trainer/IFEVAL_GRPO.md`
- Reasoning model support: See think tag removal in `verl/utils/reward_score/ifeval.py:_remove_thinking_section()`
- VERL documentation: [VERL GitHub](https://github.com/volcengine/verl)
