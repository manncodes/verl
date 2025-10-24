# Training with IFEval and GRPO

This guide explains how to use GRPO (Group Relative Policy Optimization) to train language models on instruction following using the IFEval benchmark.

## What is IFEval?

IFEval (Instruction-Following Evaluation) is a benchmark developed by Google Research for evaluating how well language models follow specific, verifiable instructions. Unlike subjective evaluations, IFEval focuses on instructions that can be automatically verified, such as:

- **Length constraints**: "Write in more than 400 words"
- **Keyword requirements**: "Mention the keyword 'AI' at least 3 times"
- **Format requirements**: "Use JSON format", "Include a title"
- **Structural constraints**: "Use exactly 5 paragraphs", "End with postscript P.S."
- **Content constraints**: "Don't use the word 'the'", "Start with 'Dear'"

The benchmark contains around 500 prompts with one or more verifiable instructions per prompt.

**Reference**: Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models." arXiv preprint arXiv:2311.07911 (2023).

**Dataset**: https://huggingface.co/datasets/google/IFEval

## Why Use GRPO for Instruction Following?

GRPO is particularly well-suited for instruction following tasks because:

1. **Group-based learning**: Generates multiple responses per prompt and learns from relative quality
2. **Sparse rewards**: Works well with binary/sparse rewards (instruction followed or not)
3. **Efficient**: No need for a separate critic model
4. **Stable**: Relative rewards reduce variance in training

## Quick Start

### Step 1: Prepare the IFEval Dataset

First, download and preprocess the IFEval dataset:

```bash
python examples/data_preprocess/ifeval.py \
    --local_save_dir ~/data/ifeval \
    --add_instruction_prompt
```

This will:
- Download the `google/IFEval` dataset from Hugging Face
- Convert it to the verl parquet format
- Split into train (80%) and test (20%) sets
- Save to `~/data/ifeval/`

### Step 2: Run GRPO Training

Run the example training script:

```bash
bash examples/grpo_trainer/run_ifeval_grpo.sh
```

Or customize your own training:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/ifeval/train.parquet \
    data.val_files=$HOME/data/ifeval/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.rollout.n=8 \
    trainer.total_epochs=10
```

### Step 3: Monitor Training

The training will log metrics including:
- `inst_strict_acc`: Instruction-level accuracy (average across all instructions)
- `prompt_strict_acc`: Prompt-level accuracy (all instructions in prompt followed)
- `num_instructions`: Total number of instructions evaluated
- `num_followed`: Number of instructions successfully followed

## How IFEval Rewards Work

### Reward Function

The IFEval reward function (`verl/utils/reward_score/ifeval.py`) evaluates each response against the instructions in the prompt:

```python
def compute_score(solution_str, ground_truth, extra_info):
    # Verifies each instruction in the prompt
    # Returns dict with:
    # - score: instruction-level accuracy (0.0 to 1.0)
    # - prompt_strict_acc: 1.0 if all instructions followed, else 0.0
    # - inst_strict_acc: fraction of instructions followed
    # - num_instructions: count of instructions
    # - num_followed: count of successfully followed instructions
```

### Verification Methods

The reward function supports two modes:

1. **Official Library** (recommended): Uses Google's official `instruction_following_eval` library
   ```bash
   pip install git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval
   ```

2. **Built-in Verification**: Fallback implementation covering common instruction types

### Strict vs. Loose Accuracy

- **Strict**: Evaluates the response exactly as generated
- **Loose**: Removes first/last lines and markdown formatting before evaluation (more forgiving)

## Configuration Details

### Key GRPO Parameters for IFEval

```yaml
# Generate multiple responses per prompt (group size)
actor_rollout_ref.rollout.n: 8  # Try 5-10 for instruction following

# Batch sizes (adjust based on GPU memory)
data.train_batch_size: 512
actor_rollout_ref.actor.ppo_mini_batch_size: 128

# Sequence lengths (IFEval responses can be longer)
data.max_prompt_length: 1024
data.max_response_length: 2048  # Longer than math tasks

# Learning rate (lower for instruction following fine-tuning)
actor_rollout_ref.actor.optim.lr: 5e-7

# GRPO-specific settings
algorithm.adv_estimator: grpo
algorithm.norm_adv_by_std_in_grpo: True
actor_rollout_ref.actor.use_kl_loss: True
actor_rollout_ref.actor.kl_loss_coef: 0.001
```

### Adjusting for Your Setup

**Single GPU Training:**
```bash
trainer.n_gpus_per_node=1 \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
```

**Multi-node Training:**
```bash
trainer.nnodes=4 \
trainer.n_gpus_per_node=8
```

**LoRA Fine-tuning:**
```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
actor_rollout_ref.actor.lora.enable=True \
actor_rollout_ref.actor.lora.r=64 \
actor_rollout_ref.actor.lora.alpha=128
```

## Dataset Format

The preprocessed IFEval dataset follows the verl format:

```json
{
    "data_source": "google/IFEval",
    "prompt": [
        {
            "role": "user",
            "content": "Write a limerick about a cat. Use exactly 5 lines and mention the word 'whiskers' at least twice."
        }
    ],
    "ability": "instruction_following",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "instruction_id_list": [
                "length_constraints:number_sentences",
                "keywords:existence"
            ],
            "kwargs": [
                {"num_sentences": 5, "relation": "at least"},
                {"keywords": ["whiskers"], "frequency": 2}
            ]
        }
    },
    "extra_info": {
        "key": "ifeval_123",
        "instruction_id_list": [...],
        "kwargs": [...],
        "prompt": "..."
    }
}
```

## Common Instruction Types

The IFEval benchmark includes ~25 types of verifiable instructions:

| Category | Example Instructions |
|----------|---------------------|
| **Keywords** | Mention "AI" at least 3 times, Include the word "robot" |
| **Length** | Write more than 400 words, Use at least 5 sentences |
| **Format** | Use JSON format, Include a title, Use bullet points |
| **Structure** | Use exactly 5 paragraphs, End with "P.S." |
| **Content** | Don't use the word "the", Start with "Dear" |
| **Case** | Respond in all lowercase, Use all caps for the title |
| **Repetition** | Repeat the prompt, Echo the first word twice |

## Tips for Best Results

1. **Group Size**: Use 8-10 responses per prompt for better exploration
2. **Learning Rate**: Start with 5e-7, lower if training is unstable
3. **Sequence Length**: IFEval needs longer responses (2048+) than math tasks
4. **Evaluation Frequency**: Test every 2-5 epochs to monitor instruction following
5. **Mixed Training**: Consider mixing IFEval with other datasets for general capability

## Troubleshooting

### Low Instruction Following Rate

- Increase `actor_rollout_ref.rollout.n` (group size)
- Decrease learning rate
- Increase training epochs
- Try different base models (some follow instructions better)

### GPU Out of Memory

- Decrease `ppo_micro_batch_size_per_gpu`
- Decrease `max_response_length`
- Enable gradient checkpointing
- Use LoRA instead of full fine-tuning

### Verification Errors

- Install official library: `pip install git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval`
- Check dataset format matches expected structure
- Enable loose accuracy mode for more forgiving evaluation

## Evaluating Your Model

After training, evaluate on the test set:

```bash
python -m verl.trainer.main_eval \
    data.val_files=$HOME/data/ifeval/test.parquet \
    actor_rollout_ref.model.path=/path/to/trained/model
```

Compare metrics:
- **Instruction-level accuracy**: Should be > 0.7 for good instruction following
- **Prompt-level accuracy**: Should be > 0.5 (harder metric)

## References

- **IFEval Paper**: https://arxiv.org/abs/2311.07911
- **Dataset**: https://huggingface.co/datasets/google/IFEval
- **GRPO Documentation**: `examples/grpo_trainer/README.md`
- **Official Library**: https://github.com/google-research/google-research/tree/master/instruction_following_eval

## Citation

If you use IFEval in your research, please cite:

```bibtex
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```
