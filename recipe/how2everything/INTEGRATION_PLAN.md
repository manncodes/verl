# How2Everything x VeRL Integration Plan

## Overview

This plan describes how to integrate the [how2everything](https://github.com/lilakk/how2everything) dataset and reward system into the VeRL reinforcement learning framework. The goal is to train language models to generate better procedural instructions using GRPO with How2Score (LLM-as-judge) rewards powered by the How2Judge 8B model.

**Key artifacts from how2everything:**
- **how2everything/how2train_rl_100k** -- 100K procedural RL training examples (goal, resources, steps)
- **how2everything/how2bench** -- 7K evaluation benchmark
- **how2everything/how2judge** -- 8B judge model (distilled from GPT-5) for scoring procedural quality
- **How2Score** -- LLM-as-judge protocol detecting "critical failures" in generated procedures

---

## Architecture

The How2Score reward is **not a rule-based string-match reward** (like GSM8K's `#### NUMBER`). It requires a **generative reward model (GenRM)** -- the How2Judge 8B model -- to evaluate whether a generated procedure contains critical failures.

### Async Reward Pipeline (FAPO pattern)

VeRL provides two paths for reward computation. This recipe uses the **async rollout mode** path, which is the same infrastructure used by the FAPO recipe:

```
actor_rollout_ref.rollout.mode=async
        ↓
AgentLoopManager (verl/experimental/agent_loop/)
        ↓
RewardLoopManager (verl/experimental/reward/reward_manager.py)
  - Deploys How2Judge as a sglang server via RewardModelManager
  - Creates RewardLoopWorker instances (Ray actors)
        ↓
RewardLoopWorker._init_reward_fn()
  - Loads the custom reward function via get_custom_reward_fn(config)
  - Creates NaiveRewardLoopManager with reward_router_address + reward_model_tokenizer
        ↓
NaiveRewardLoopManager.run_single(data)
  - Detects that compute_score_how2 is async (inspect.iscoroutinefunction)
  - Calls: await compute_score_how2(
      data_source, solution_str, ground_truth, extra_info,
      reward_router_address=...,      # ← injected by the infrastructure
      reward_model_tokenizer=...,     # ← injected by the infrastructure
    )
        ↓
compute_score_how2() in reward_fn.py
  - Builds judge prompt from ground_truth JSON
  - Sends to How2Judge via aiohttp POST to reward_router_address
  - Parses JSON response: {"reasoning": ..., "critical_failures": [...]}
  - Returns {"score": 1.0/-1.0, "has_failure": bool, ...}
```

**Key insight**: `reward_router_address` and `reward_model_tokenizer` are NOT passed via `custom_reward_function.reward_kwargs`. They are injected automatically by `NaiveRewardLoopManager.run_single()` using values stored in `RewardLoopWorker._init_reward_fn()`.

### Sync Reward Pipeline (rule-based ablation)

For the rule-based ablation (`compute_score_how2_rule`), the standard sync path is used:

```
reward_model.enable=False  (no async mode needed)
        ↓
NaiveRewardManager.__call__(data)
  - Calls compute_score_how2_rule(data_source, solution_str, ground_truth, extra_info)
  - Returns reward_tensor
```

---

## File Structure

```
recipe/how2everything/
├── INTEGRATION_PLAN.md          # This document
├── __init__.py
├── config/
│   └── genrm_config.yaml        # Hydra config extending ppo_trainer
├── data_preprocess.py           # Convert how2train_rl_100k + how2bench to verl parquet
├── reward_fn.py                 # Self-contained: judge prompt, parser, async + sync reward fns
├── run_grpo_7b.sh               # Full GRPO training with How2Judge GenRM (async mode)
└── run_grpo_7b_rule.sh          # Rule-only ablation (sync mode, no judge)
```

---

## How2Judge Response Format

The How2Judge model outputs **structured JSON** (not free-text verdicts):

```json
{
  "reasoning": "Step-by-step evaluation of the candidate procedure...",
  "critical_failures": [
    {
      "failure": "Description of the critical failure",
      "L1_steps": [3, 5],
      "L2_steps": [2]
    }
  ]
}
```

- **PASS**: `critical_failures` is an empty list → score = +1.0
- **FAIL**: `critical_failures` has one or more entries → score = -1.0
- **Parse failure**: JSON could not be extracted → score = 0.0

The judge prompt ends with `"Return only valid json."` to encourage structured output.

---

## Judge Prompt Template

The upstream judge prompt uses **3 template variables** (no resources):

```
{goal}              - The procedural goal
{reference_steps}   - Ground truth steps (L1), numbered
{steps}             - Candidate steps (L2), raw model output
```

---

## Data Schema

### Upstream formats (auto-detected by data_preprocess.py)

**Flat format** (how2bench, likely how2train):
```python
{"goal": str, "steps": list[str], "resources": list[str], "topic": str, ...}
```

**Nested format** (how2mine export):
```python
{
  "source_example": {"id": str, "topic": str, "url": str, "text": str},
  "final_procedure": {"goal": str, "steps": list[str], "resources": list[str]}
}
```

### VeRL parquet schema

```python
{
    "data_source": "how2everything/how2train",
    "prompt": [{"role": "user", "content": "<instruction template>"}],
    "ability": "procedural",
    "reward_model": {
        "style": "rule",                    # "rule" so validation runs on test data
        "ground_truth": json.dumps({
            "goal": str, "resources": list[str],
            "steps": list[str], "n_steps": int
        })
    },
    "extra_info": {"split": str, "index": int, "topic": str, ...}
}
```

Note: `reward_model.style` is set to `"rule"` (not `"model"`) because the trainer skips validation for `style == "model"` examples. Our GenRM reward is triggered by the custom reward function, not by the style field.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RL algorithm | GRPO (no critic) | Matches how2everything paper. GRPO avoids the overhead of a critic model. |
| Rollout mode | `async` | Required by the experimental reward loop infrastructure that handles GenRM. Same as FAPO. |
| Reward model engine | `sglang` | Supports the `/generate` endpoint needed for GenRM text generation. FAPO uses sglang for the same reason. |
| Judge output format | JSON | How2Judge was trained to output `{"reasoning": ..., "critical_failures": [...]}`. |
| Reward signal | Binary {-1, +1} + parse-failure 0 | Empty critical_failures → PASS (+1.0). Non-empty → FAIL (-1.0). Unparseable → 0. |
| reward_model.style | `"rule"` | Ensures validation runs on test data. The GenRM is invoked via custom_reward_function, not the style field. |
| Self-contained reward_fn.py | No cross-module imports | Custom modules are loaded via `importlib.spec_from_file_location` -- sibling imports from `recipe.*` are unreliable. |

---

## How to Run

### Step 1: Prepare data
```bash
python recipe/how2everything/data_preprocess.py --local_save_dir ~/data/how2everything
```

### Step 2: Validate with rule-based reward (no judge needed)
```bash
bash recipe/how2everything/run_grpo_7b_rule.sh
```

### Step 3: Train with How2Judge GenRM
```bash
JUDGE_PATH=how2everything/how2judge bash recipe/how2everything/run_grpo_7b.sh
```

### Step 4: Evaluate with upstream how2bench
```bash
uv run h2e bench run --config <eval_config pointing to trained checkpoint>
```
