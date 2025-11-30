# AIME 2025 + IFEval Test Dataset Preprocessing

This guide shows how to create a combined test dataset with AIME 2025 and Google's IFEval for evaluation.

## Quick Start

### 1. Preprocess the Dataset

```bash
# Create combined test.parquet with both AIME and IFEval
python examples/data_preprocess/aime2025_ifeval_test.py \
    --local_save_dir ~/data/aime_ifeval_test

# Or with custom sampling
python examples/data_preprocess/aime2025_ifeval_test.py \
    --local_save_dir ~/data/aime_ifeval_test \
    --max_aime_samples 50 \
    --max_ifeval_samples 100
```

### 2. Use with GRPO Training

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=/path/to/your/train.parquet \
    data.val_files=~/data/aime_ifeval_test/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    reward_model.reward_manager=batch \
    custom_reward_function.path=verl/utils/reward_score/ifeval_batch.py \
    custom_reward_function.name=compute_score_batched \
    custom_reward_function.reward_kwargs.strict=True \
    custom_reward_function.reward_kwargs.alpha_threshold=0.5 \
    trainer.project_name='verl_ifeval_grpo' \
    trainer.experiment_name='qwen2.5_3b_ifeval_eval' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

## Dataset Options

### IFEval Only
```bash
python examples/data_preprocess/aime2025_ifeval_test.py \
    --local_save_dir ~/data/ifeval_test \
    --ifeval_only
```

### AIME Only
```bash
python examples/data_preprocess/aime2025_ifeval_test.py \
    --local_save_dir ~/data/aime_test \
    --aime_only
```

### Custom Dataset Paths
```bash
python examples/data_preprocess/aime2025_ifeval_test.py \
    --local_save_dir ~/data/aime_ifeval_test \
    --aime_local_path /path/to/aime/dataset \
    --ifeval_local_path /path/to/ifeval/dataset
```

## Output Format

The generated `test.parquet` contains examples with:

```python
{
    "data_source": "ifeval" | "aime_2025",
    "prompt": [{"role": "user", "content": "..."}],
    "ability": "instruction_following" | "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "instruction_id_list": [...],  # IFEval only
            "kwargs": [...]                 # IFEval only
        } | "answer"  # AIME
    },
    "extra_info": {
        "ability": "...",
        "split": "test",
        "index": 0,
        "instruction_id_list": [...],  # IFEval only
        "kwargs": [...],               # IFEval only
        "prompt": "...",               # Required for judge
        ...
    }
}
```

## Dataset Sources

- **AIME 2025**: Advanced math reasoning (tries `Maxwell-Jia/AIME_2024` or `hendrycks/math`)
- **IFEval**: Google's official instruction following eval (`google/IFEval`)

## Reward Function Compatibility

The output is designed to work with:
- `verl/utils/reward_score/ifeval_batch.py::compute_score_batched`
- Batched reward computation with concurrent judge evaluation
- Proper handling of both math (AIME) and instruction following (IFEval) examples

## Judge Configuration

The batched IFEval reward function uses a StructuredJudge with:
- `max_workers=128` for concurrent evaluations
- `timeout=2.0s` per evaluation (prevents hanging)
- `max_retries=0` (returns None on timeout)

To adjust these settings, modify `verl/utils/reward_score/ifeval_batch.py:54-60`.
