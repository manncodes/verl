# Prompt Distillation

Prompt distillation trains a student model to internalize the knowledge encoded in a detailed system prompt, so it can perform the task **without the system prompt** at inference time. This reduces inference cost and latency by eliminating long prompt overhead.

## How It Works

1. **Teacher**: A model receives a rich system prompt (e.g., detailed classification rules, chain-of-thought instructions) along with user input and produces high-quality responses.
2. **Student**: Trained to replicate the teacher's behavior using ONLY the user input (no system prompt).
3. **Result**: The student has "baked" the system prompt knowledge into its weights.

## Two Approaches

### Off-Policy (SFT-based)
Generate teacher-labeled data once, then train the student via supervised fine-tuning.

- **Simpler**, requires only data generation + existing SFT trainer
- Good for classification, formatting, and rule-following tasks
- Best when the task is well-defined and teacher outputs are consistent

### On-Policy (KL-based)
Student generates responses, teacher provides logprobs (with system prompt), train via KL divergence.

- **Iterative** improvement: student keeps getting better each round
- Better for open-ended generation and reasoning tasks
- Requires a running teacher server during training

## Quick Start: Off-Policy (SFT)

### Step 1: Prepare Teacher Data

```bash
python -m recipe.prompt_distillation.prepare_data \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --system_prompt_file recipe/prompt_distillation/examples/language_classify/system_prompt.txt \
    --input_file ~/data/inputs.parquet \
    --output_file ~/data/distill_train.parquet \
    --input_key question \
    --temperature 0.7 \
    --max_new_tokens 512 \
    --tp_size 4
```

This generates teacher responses WITH the system prompt and saves them as training pairs WITHOUT the system prompt.

### Step 2: Train Student via SFT

```bash
bash recipe/prompt_distillation/run_sft.sh 4 \
    ~/data/distill_train.parquet \
    Qwen/Qwen2.5-7B-Instruct \
    checkpoints/prompt_distill \
    model.lora_rank=32 \
    optim.lr=1e-4 \
    trainer.total_epochs=3
```

## Quick Start: On-Policy (KL Distillation)

### Step 1: Start Teacher Server

Use the GKD teacher server (the teacher model runs separately and serves logprobs via ZMQ):

```bash
cd recipe/gkd/teacher
bash start_server.sh \
    --ckpt-path /path/to/teacher_model \
    --port 15555 \
    --n-logprobs 256 \
    --tp-size 4
```

### Step 2: Run On-Policy Training

```bash
python -m recipe.prompt_distillation.main_prompt_distill \
    data.train_files=/path/to/train.parquet \
    data.prompt_key=question \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.train_batch_size=256 \
    actor_rollout_ref.model.path=/path/to/student_model \
    actor_rollout_ref.teacher.server_ip=localhost \
    actor_rollout_ref.teacher.server_port=15555 \
    prompt_distillation.system_prompt_path=/path/to/system_prompt.txt \
    trainer.total_epochs=10 \
    trainer.n_gpus_per_node=4
```

## Configuration Reference

### prompt_distillation section

| Parameter | Type | Description |
|-----------|------|-------------|
| `system_prompt_path` | str | Path to a text file containing the system prompt |
| `system_prompt` | str | Inline system prompt string (alternative to file) |

### Data preparation (`prepare_data.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | required | Teacher model path |
| `--system_prompt_file` | null | Path to system prompt text file |
| `--system_prompt` | null | Inline system prompt string |
| `--input_file` | required | Input data (Parquet/JSON/JSONL) |
| `--output_file` | required | Output Parquet path |
| `--input_key` | "question" | Column name for user inputs |
| `--batch_size` | 64 | Generation batch size |
| `--temperature` | 0.7 | Sampling temperature |
| `--max_new_tokens` | 512 | Max tokens to generate |
| `--n_samples` | 1 | Responses per input |
| `--tp_size` | 1 | Tensor parallel size |

### On-policy training

All GKD configuration options are supported. See `config/prompt_distill_trainer.yaml` for the full config and `recipe/gkd/README.md` for GKD-specific documentation.

## Example: Language Classification

See `examples/language_classify/` for a complete example that distills a 70-line language classification system prompt into model weights. The example:

1. Uses a detailed system prompt with script detection rules and edge-case handling
2. Generates teacher labels for multilingual text samples
3. Trains a student to classify 13 languages without the system prompt

```bash
cd recipe/prompt_distillation/examples/language_classify
bash run.sh
```

## Architecture

```
Off-policy (SFT):
  Teacher (with system prompt) ──► Generate labeled data ──► SFT Trainer ──► Student

On-policy (KL):
  ┌─────────────────────────────────────────────────────────┐
  │  Training Loop (PromptDistillTrainer)                   │
  │                                                         │
  │  Student rollout ──► Generate responses (no sys prompt) │
  │       │                                                 │
  │       ▼                                                 │
  │  Teacher query ──► Get logprobs (WITH sys prompt)       │
  │       │                                                 │
  │       ▼                                                 │
  │  KL loss ──► Update student weights                     │
  │       │                                                 │
  │       ▼                                                 │
  │  Sync weights to rollout ──► Next iteration             │
  └─────────────────────────────────────────────────────────┘
```

## References

- [Knowledge Injection via Prompt Distillation](https://arxiv.org/abs/2412.14964) - Foundational paper on prompt distillation
- [GKD Recipe](../gkd/) - On-policy knowledge distillation recipe that this extends
- [verl SFT Trainer](../../verl/trainer/fsdp_sft_trainer.py) - FSDP SFT trainer used for off-policy distillation
