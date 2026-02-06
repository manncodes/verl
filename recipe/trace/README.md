# TRACE: Truncated Reasoning AUC Evaluation

Implementation of the TRACE method for detecting implicit reward hacking in
reasoning models during RL training.

## Paper

> Wang, X., Joshi, N., Plank, B., Angell, R., & He, H. (2025).
> "Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring
> Reasoning Effort." [arXiv:2510.01367](https://arxiv.org/abs/2510.01367)

## Overview

Reward hacking occurs when a reasoning model exploits loopholes in a reward
function to achieve high rewards without actually solving the intended task.
This behavior may be **implicit** -- the chain-of-thought (CoT) appears
benign and bypasses CoT monitors.

TRACE detects implicit reward hacking by measuring **reasoning effort**:

1. **Truncate** the model's CoT at various fractions (10%, 20%, ..., 100%)
2. **Force the model to answer** from the truncated prefix
3. **Estimate expected reward** at each truncation level (sample N completions)
4. **Compute AUC** of the reward-vs-truncation curve (the **TRACE score**)

A hacking model achieves high reward with little reasoning (high AUC), while a
legitimately reasoning model needs most of its CoT (low AUC).

## Module Structure

```
verl/trainer/trace/
├── __init__.py              # Public API
├── config.py                # TRACEConfig dataclass
├── core.py                  # Core algorithms (truncation, AUC computation)
├── detector.py              # TRACEDetector (baseline calibration + classification)
├── callback.py              # TRACECallback (training loop integration)
├── reward_manager.py        # TRACE-aware reward manager wrapper
└── loophole_discovery.py    # Unsupervised loophole discovery via clustering
```

## Usage

### 1. As a Training Callback (Recommended)

Integrate TRACE into your existing PPO training loop:

```python
from verl.trainer.trace import TRACEConfig, TRACEDetector
from verl.trainer.trace.callback import TRACECallback

# Configure TRACE
trace_config = TRACEConfig(
    enable=True,
    truncation_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    num_completions=4,
    detection_frequency=10,      # Run TRACE every 10 steps
    baseline_steps=5,            # Collect baseline from first 5 steps
    use_for_reward_penalty=True, # Penalize hacking samples
    reward_penalty_coef=1.0,
)

# Create callback
trace_callback = TRACECallback(trace_config, tokenizer, reward_fn)

# In your training loop:
for step, batch in enumerate(dataloader):
    # After rollout
    batch = trace_callback.on_after_rollout(batch, step, generate_fn)

    # After reward computation
    batch = trace_callback.on_after_reward(batch, step)

    # Log TRACE metrics
    metrics.update(trace_callback.get_metrics())
```

### 2. As a Reward Manager Wrapper

Wrap your existing reward manager with TRACE detection:

```python
from verl.trainer.trace.reward_manager import TRACERewardManager

# Use "trace" as the reward manager name in config
# It wraps your existing reward manager
```

### 3. Standalone TRACE Score Computation

```python
from verl.trainer.trace.core import compute_trace_scores_batch

results = compute_trace_scores_batch(
    prompt_ids=prompt_ids,
    response_ids=response_ids,
    response_mask=response_mask,
    attention_mask=attention_mask,
    truncation_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    generate_fn=model.generate,
    reward_fn=compute_score,
    tokenizer=tokenizer,
    num_completions=4,
)

trace_scores = results["trace_scores"]  # (batch_size,)
```

### 4. Loophole Discovery

```python
from verl.trainer.trace import TRACELoopholeDiscovery

discovery = TRACELoopholeDiscovery(trace_config)
clusters = discovery.cluster_samples(trace_scores)
analysis = discovery.analyze_clusters(clusters, data_sources=data_sources)

# Check for loopholes
if analysis["loophole_candidates"]:
    print("Potential loopholes found in:", analysis["loophole_candidates"])
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable TRACE detection |
| `truncation_fractions` | `[0.1, ..., 1.0]` | CoT truncation points |
| `num_completions` | `4` | Completions per truncation point |
| `temperature` | `0.7` | Sampling temperature |
| `detection_frequency` | `10` | Steps between TRACE runs |
| `baseline_steps` | `5` | Initial steps for baseline |
| `threshold_method` | `baseline_mean_std` | How to set threshold |
| `threshold_n_sigma` | `2.0` | Std devs above baseline mean |
| `use_for_filtering` | `False` | Filter hacking samples |
| `use_for_reward_penalty` | `False` | Penalize hacking rewards |
| `reward_penalty_coef` | `1.0` | Penalty strength |
| `enable_loophole_discovery` | `False` | Enable clustering-based discovery |

## Logged Metrics

| Metric | Description |
|--------|-------------|
| `trace/hacking_fraction` | Fraction of samples detected as hacking |
| `trace/mean_score` | Mean TRACE score in batch |
| `trace/threshold` | Current detection threshold |
| `trace/baseline_mean` | Baseline distribution mean |
| `trace/baseline_std` | Baseline distribution std |
| `trace_loophole/separation` | Cluster separation (higher = clearer signal) |
