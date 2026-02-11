# How2Everything x VeRL Integration Plan

## Overview

This plan describes how to integrate the [how2everything](https://github.com/lilakk/how2everything) dataset and reward system into the VeRL reinforcement learning framework. The goal is to train language models to generate better procedural instructions using GRPO with How2Score (LLM-as-judge) rewards powered by the How2Judge 8B model.

**Key artifacts from how2everything:**
- **how2everything/how2train_rl_100k** -- 100K procedural RL training examples (goal, resources, steps)
- **how2everything/how2bench** -- 7K evaluation benchmark
- **how2everything/how2judge** -- 8B judge model (distilled from GPT-5) for scoring procedural quality
- **How2Score** -- LLM-as-judge protocol detecting "critical failures" in generated procedures

---

## Architecture Decision

The How2Score reward is **not a rule-based string-match reward** (like GSM8K's `#### NUMBER`). It requires a **generative reward model (GenRM)** -- the How2Judge 8B model -- to evaluate whether a generated procedure contains critical failures.

This maps directly to VeRL's **FAPO-style GenRM integration pattern**, where:
1. The policy model generates a procedure given a goal + resources
2. The How2Judge model evaluates the generated procedure against the reference
3. The judge's verdict is parsed into a scalar reward for GRPO training

---

## File Structure

```
recipe/how2everything/
├── INTEGRATION_PLAN.md          # This document
├── config/
│   └── genrm_config.yaml        # Hydra config extending ppo_trainer (enables GenRM)
├── data_preprocess.py           # Convert how2train_rl_100k + how2bench to verl parquet
├── reward_fn.py                 # Custom reward function (sync + async variants)
├── judge_prompt.py              # How2Score judge prompt template
├── run_grpo_7b.sh               # Training script: 7B model with GRPO + How2Judge
├── run_grpo_7b_rule.sh          # Training script: 7B model with rule-only reward (ablation)
└── README.md                    # Recipe documentation (deferred)
```

---

## Component 1: Data Preprocessing (`data_preprocess.py`)

### Input
- HuggingFace dataset: `how2everything/how2train_rl_100k` (train split)
- HuggingFace dataset: `how2everything/how2bench` (test/eval split)

### Expected source schema (how2train_rl_100k)
Each example contains:
- `goal` (str) -- what the procedure aims to accomplish
- `resources` (list[str]) -- tools/materials needed
- `steps` (list[str]) -- reference step-by-step procedure
- `topic` (str) -- one of 14 topic categories (optional metadata)

### Target verl parquet schema
```python
{
    "data_source": "how2everything/how2train",
    "prompt": [
        {
            "role": "user",
            "content": "You will be given a goal and a list of resources. "
                       "Your task is to output a list of steps that complete "
                       "the goal using the given resources.\n\n"
                       "Goal:\n{goal}\n\n"
                       "Resources:\n{resources}\n\n"
                       "Output exactly {n} steps to achieve the goal "
                       "using the given resources."
        }
    ],
    "ability": "procedural",
    "reward_model": {
        "style": "genrm",           # Signals that GenRM is needed (not rule-based)
        "ground_truth": json.dumps({ # Reference procedure for judge comparison
            "goal": goal,
            "resources": resources,
            "steps": steps,
            "n_steps": len(steps)
        })
    },
    "extra_info": {
        "split": "train",
        "index": idx,
        "topic": topic,
        "goal": goal,
        "resources": resources,
        "reference_steps": steps,
    }
}
```

### Implementation notes
- Resources list is formatted as a bracketed comma-separated string: `[item1, item2, ...]`
- The `n` in the prompt matches the number of reference steps, giving the model a target length
- Deduplication against how2bench should be run beforehand (cosine similarity threshold 0.65, following the upstream repo's `dedup_against_test.py`)
- Both train and test splits are saved as separate parquet files

### Pseudocode
```python
import json, datasets

def make_map_fn(split):
    def process_fn(example, idx):
        goal = example["goal"]
        resources = example["resources"]  # list[str]
        steps = example["steps"]          # list[str]
        n_steps = len(steps)

        resources_str = "[" + ", ".join(resources) + "]"

        prompt_text = (
            "You will be given a goal and a list of resources. "
            "Your task is to output a list of steps that complete "
            "the goal using the given resources.\n\n"
            f"Goal:\n{goal}\n\n"
            f"Resources:\n{resources_str}\n\n"
            f"Output exactly {n_steps} steps to achieve the goal "
            f"using the given resources."
        )

        ground_truth = json.dumps({
            "goal": goal,
            "resources": resources,
            "steps": steps,
            "n_steps": n_steps,
        })

        return {
            "data_source": "how2everything/how2train",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "procedural",
            "reward_model": {"style": "genrm", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "topic": example.get("topic", "unknown"),
                "goal": goal,
                "resources": resources,
                "reference_steps": steps,
            },
        }
    return process_fn
```

---

## Component 2: How2Score Judge Prompt (`judge_prompt.py`)

This implements the How2Score rubric as a prompt template for the How2Judge GenRM.

### Rubric (from the paper)
The judge evaluates whether the candidate procedure (L2, generated by the policy model) contains **critical failures** compared to the reference procedure (L1, ground truth):

- **Contradictions**: Steps that contradict the goal or diverge significantly from L1
- **Logical issues**: Internal inconsistencies, incoherence, or severe vagueness
- **Missing/extraneous actions**: Omitted essential steps or unnecessary additions that would prevent achieving the goal

The judge outputs structured reasoning + a list of critical failures (or declares none found).

### Prompt template
```python
HOW2SCORE_JUDGE_TEMPLATE = """\
You are evaluating whether a candidate procedure (L2) correctly achieves a stated goal, \
using a reference procedure (L1) as a reliable guide.

[Goal]
{goal}

[Resources]
{resources}

[Reference Procedure (L1)]
{reference_steps}

[Candidate Procedure (L2)]
{candidate_steps}

A "critical failure" is any issue that would prevent achieving the goal. This includes:
- Steps that contradict the goal or diverge significantly from the reference
- Internal inconsistencies, incoherence, or severe vagueness
- Missing essential steps or unnecessary additions that would prevent success

L1 reliably achieves the goal as written, but it may not be the only valid way. \
Use it as a reliable reference, not the exclusive solution. \
Minor phrasing differences and additional practical steps that don't interfere \
with the outcome are acceptable.

First, provide detailed reasoning explaining your evaluation step by step. \
Then, list any critical failures found. If there are no critical failures, \
state "No critical failures found."

Finally, on the last line, output your verdict as exactly one of:
VERDICT: PASS
VERDICT: FAIL
"""
```

### Parsing logic
```python
def parse_judge_verdict(judge_output: str) -> dict:
    """Parse the How2Judge output into a reward signal."""
    text = judge_output.strip()
    # Look for VERDICT line
    if "VERDICT: PASS" in text.upper():
        return {"score": 1.0, "verdict": "pass", "has_critical_failure": False}
    elif "VERDICT: FAIL" in text.upper():
        return {"score": -1.0, "verdict": "fail", "has_critical_failure": True}
    else:
        # If judge didn't produce a clear verdict, treat as ambiguous → 0
        return {"score": 0.0, "verdict": "ambiguous", "has_critical_failure": None}
```

---

## Component 3: Custom Reward Function (`reward_fn.py`)

Two variants, following verl patterns:

### Variant A: Async GenRM reward (primary -- uses How2Judge during training)

This follows the FAPO pattern: the How2Judge 8B model runs on a separate GPU pool as a vLLM/SGLang server, and the reward function calls it asynchronously.

```python
async def compute_score_how2(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,              # Address of How2Judge vLLM server
    reward_model_tokenizer: PreTrainedTokenizer,
):
    """
    Compute How2Score reward using the How2Judge generative reward model.

    Flow:
    1. Parse ground_truth JSON to get reference procedure
    2. Parse solution_str to extract candidate steps
    3. Construct judge prompt from template
    4. Send to How2Judge GenRM via reward_router_address
    5. Parse verdict into scalar reward
    """
    import json
    ref = json.loads(ground_truth)

    judge_prompt = HOW2SCORE_JUDGE_TEMPLATE.format(
        goal=ref["goal"],
        resources="[" + ", ".join(ref["resources"]) + "]",
        reference_steps=format_steps(ref["steps"]),
        candidate_steps=solution_str,          # Raw model output
    )

    # Tokenize and send to GenRM
    prompt_ids = reward_model_tokenizer.apply_chat_template(
        [{"role": "user", "content": judge_prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )

    grm_outputs = await generate_aiohttp(
        router_address=reward_router_address,
        prompt_ids=prompt_ids,
        sampling_params={"max_new_tokens": 2048},
    )

    grm_response_ids = grm_outputs.get("output_ids", None)
    if grm_response_ids is not None:
        grm_response = reward_model_tokenizer.decode(
            grm_response_ids, skip_special_tokens=True
        )
        result = parse_judge_verdict(grm_response)
    else:
        result = {"score": 0.0, "verdict": "error", "has_critical_failure": None}

    return {
        "score": result["score"],
        "verdict": result["verdict"],
        "has_critical_failure": result["has_critical_failure"],
    }
```

### Variant B: Rule-based heuristic reward (ablation / no-GPU fallback)

A lightweight reward that doesn't require the judge model. Useful for ablation studies and debugging.

```python
def compute_score_how2_rule(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
):
    """
    Heuristic reward for procedural generation (no judge model needed).

    Scoring criteria:
    - +0.3: Response contains numbered steps
    - +0.3: Number of steps matches expected count (within +/- 1)
    - +0.2: Response mentions key resources from the prompt
    - +0.2: Response doesn't contain obvious refusal patterns
    - Negative: -1.0 for empty or refusal responses
    """
    import json, re
    ref = json.loads(ground_truth)

    if not solution_str.strip() or any(
        p in solution_str.lower() for p in ["i cannot", "i can't", "i'm sorry"]
    ):
        return {"score": -1.0, "acc": False}

    score = 0.0
    # Check for numbered steps
    numbered = re.findall(r"^\d+[\.\)]\s", solution_str, re.MULTILINE)
    if numbered:
        score += 0.3

    # Check step count
    expected_n = ref.get("n_steps", 0)
    if expected_n > 0 and abs(len(numbered) - expected_n) <= 1:
        score += 0.3

    # Check resource coverage
    resources = ref.get("resources", [])
    if resources:
        mentioned = sum(1 for r in resources if r.lower() in solution_str.lower())
        score += 0.2 * (mentioned / len(resources))

    # Base credit for non-trivial response
    if len(solution_str.strip()) > 50:
        score += 0.2

    return {"score": score, "acc": score >= 0.8}
```

---

## Component 4: Hydra Config (`config/genrm_config.yaml`)

Extends the base `ppo_trainer` config to enable the GenRM reward model pool for How2Judge.

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

reward_model:
  _target_: verl.workers.config.RewardModelConfig

  reward_manager: naive
  enable: False  # Overridden to True when using GenRM in run script

  enable_resource_pool: False
  n_gpus_per_node: 0
  nnodes: 0

  model:
    type: discriminative
    path: how2everything/how2judge  # HuggingFace model ID
    external_lib: ${actor_rollout_ref.model.external_lib}
    trust_remote_code: False

  rollout:
    _target_: verl.workers.config.RolloutConfig
    name: vllm
    dtype: bfloat16
    gpu_memory_utilization: 0.90
    enforce_eager: true
    cudagraph_capture_sizes: null
    free_cache_engine: true
    data_parallel_size: 1
    expert_parallel_size: 1
    tensor_model_parallel_size: 1  # 8B fits on 1 GPU in bf16
    max_num_batched_tokens: 8192
    max_model_len: null
    max_num_seqs: 256
    load_format: auto
    engine_kwargs: {}
    limit_images: null
    enable_chunked_prefill: true
    enable_prefix_caching: true
    disable_log_stats: true
    skip_tokenizer_init: true
    prompt_length: 2048
    response_length: 2048
```

---

## Component 5: Run Scripts

### `run_grpo_7b.sh` -- GRPO with How2Judge GenRM

```bash
#!/usr/bin/env bash
set -xeuo pipefail

project_name='How2Everything'
exp_name='GRPO-7B-How2Judge'

adv_estimator=grpo

# No KL -- pure GRPO
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# Sequence lengths
max_prompt_length=1024
max_response_length=2048

# GRPO group sampling
train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=32

# Paths (user must set these env vars)
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
JUDGE_PATH=${JUDGE_PATH:-"how2everything/how2judge"}
TRAIN_FILE=${TRAIN_FILE:-"~/data/how2everything/train.parquet"}
TEST_FILE=${TEST_FILE:-"~/data/how2everything/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"~/verl/ckpts/how2everything/${exp_name}"}

# Ray cluster
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
RM_NODES=${RM_NODES:-1}

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/how2everything/config"

python3 -m verl.trainer.main_ppo \
    --config-path $CONFIG_PATH \
    --config-name genrm_config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.enable=True \
    reward_model.enable_resource_pool=True \
    reward_model.n_gpus_per_node=8 \
    reward_model.nnodes="${RM_NODES}" \
    reward_model.model.path="${JUDGE_PATH}" \
    reward_model.rollout.name=vllm \
    reward_model.rollout.gpu_memory_utilization=0.90 \
    reward_model.rollout.tensor_model_parallel_size=1 \
    reward_model.rollout.free_cache_engine=False \
    custom_reward_function.path=recipe/how2everything/reward_fn.py \
    custom_reward_function.name=compute_score_how2 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=150 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
```

### `run_grpo_7b_rule.sh` -- Rule-only ablation (no judge model)

Same structure but with:
```bash
reward_model.enable=False \
custom_reward_function.path=recipe/how2everything/reward_fn.py \
custom_reward_function.name=compute_score_how2_rule \
```

---

## Component 6: Evaluation

### During-training validation
The `val_reward_fn` in the trainer automatically evaluates on `test.parquet` (how2bench) at `test_freq` intervals using the same reward function.

### Post-training evaluation with How2Bench
After training, run the official how2bench evaluation:
```bash
# Generate procedures from the trained model
uv run h2e bench run --config eval_config.yaml
```

Where `eval_config.yaml` points to the VeRL checkpoint converted back to a HuggingFace model.

---

## Implementation Order

### Phase 1: Data Pipeline
1. **`data_preprocess.py`** -- Download how2train_rl_100k and how2bench from HF, convert to verl parquet format. Verify schema with a quick `datasets.load_dataset(...).to_parquet(...)` test.

### Phase 2: Reward Functions
2. **`judge_prompt.py`** -- Implement the How2Score judge prompt template and verdict parser.
3. **`reward_fn.py`** -- Implement both the async GenRM reward (`compute_score_how2`) and the rule-based heuristic reward (`compute_score_how2_rule`).

### Phase 3: Configuration
4. **`config/genrm_config.yaml`** -- Hydra config extending ppo_trainer with How2Judge GenRM settings.

### Phase 4: Run Scripts & Testing
5. **`run_grpo_7b_rule.sh`** -- Start with the rule-based ablation to validate the data pipeline end-to-end without needing GPU resources for the judge.
6. **`run_grpo_7b.sh`** -- Full setup with How2Judge GenRM once the pipeline is validated.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RL algorithm | GRPO (no critic) | Matches how2everything paper (used with open-instruct). GRPO avoids the overhead of a critic model and works well for generative tasks. |
| Reward model | GenRM (How2Judge 8B, async) | How2Score requires LLM-based judgment -- not extractable via regex. The 8B model fits on 1 GPU in bf16. |
| Reward signal | Binary {-1, +1} + ambiguous 0 | PASS/FAIL verdicts from the judge. Ambiguous outputs (parse failure) scored as 0 rather than penalized. |
| Group sampling | n=8 per prompt | Standard GRPO setting. 8 candidate procedures per goal, then group-normalized advantage. |
| Prompt format | Matches how2everything inference_inst.txt | Uses the upstream prompt template so the model learns the expected output format. |
| Config pattern | Follows recipe/fapo/ | FAPO is the closest existing recipe (async GenRM + GRPO). |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| How2Judge verdict parsing fails frequently | Reward signal degrades to mostly 0 | Add robust regex fallback; log parse failure rate; tune prompt for structured output |
| How2Judge latency bottleneck | Training throughput drops | Use async reward computation (FAPO pattern); batch judge calls; increase RM GPU pool |
| Dataset schema mismatch (HF columns differ from expected) | Data pipeline breaks | Inspect actual HF dataset schema first in data_preprocess.py; add column validation |
| Judge reward is too sparse (mostly PASS or mostly FAIL) | GRPO advantage estimation degenerates | Monitor pass/fail ratio during training; consider reward shaping (partial credit via rule-based component) |
| 8B judge may not fit alongside training on limited GPUs | OOM | Use separate resource pool (enable_resource_pool=True); offload judge to dedicated nodes |

---

## Optional Extensions

1. **Hybrid reward**: Combine the rule-based heuristic (Component 3B) with the GenRM verdict as `score = 0.5 * rule_score + 0.5 * judge_score` for smoother reward signal.
2. **Multi-topic curriculum**: Start training on easier topics, gradually add harder ones using verl's dynamic dataset support (`verl/experimental/`).
3. **How2Mine integration**: Use the mining pipeline to continuously generate new training data, feeding it into verl's dynamic dataset reload.
4. **Overlong buffer penalty**: Apply DAPO-style overlong penalty if the model generates excessively long procedures (use `DAPORewardManager` instead of `NaiveRewardManager`).
