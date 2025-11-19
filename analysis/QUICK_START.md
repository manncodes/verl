# Quick Start Guide

## Installation

```bash
pip install plotly pandas numpy
```

## Basic Usage (3 Steps)

### 1. Generate Dashboard

```bash
python -m analysis.run_analysis /path/to/ROLLOUT_DIR --combined-dashboard
```

This creates `analysis_output/training_dashboard.html` with all charts in one file.

### 2. Open Dashboard

Open the HTML file in your browser to explore:
- Prompt learning dynamics
- Instruction type difficulty
- Sample efficiency quadrants
- Upsampling candidates
- And 10+ other charts

### 3. Make Decisions

Based on the charts:

#### **For Upsampling**
Check **"Upsampling Candidates Table"** (Curriculum tab)
- Red rows = high priority to upsample
- Look for prompts with low final scores and slow learning rates

#### **For Curriculum Learning**
Check **"Sample Efficiency"** chart (Curriculum tab)
- **Bottom-Right quadrant**: Hard prompts - introduce late in curriculum
- **Top-Left quadrant**: Easy prompts - introduce early
- Order prompts by iterations-to-learn

#### **For Data Cleaning**
Check **"Plateau Detection"** chart (Curriculum tab)
- Red bars = plateaued prompts
- If plateaued at low score → consider removing
- If plateaued at high score → keep (already learned)

#### **For Understanding Training**
- **Reward Case Evolution**: Shows if training is improving (more bonuses over time)
- **Forgetting Analysis**: Red cells = prompts that regressed
- **Instruction Interference**: Red cells = instruction pairs that conflict

## Common Commands

```bash
# Summary statistics only (no charts)
python -m analysis.run_analysis ROLLOUT_DIR --summary-only

# Analyze specific iterations
python -m analysis.run_analysis ROLLOUT_DIR --iterations 0 1 2 3 --combined-dashboard

# Generate individual chart files
python -m analysis.run_analysis ROLLOUT_DIR --output-dir ./my_analysis

# Generate PNG instead of HTML
python -m analysis.run_analysis ROLLOUT_DIR --format png
```

## Programmatic Usage

```python
from analysis import TrainingDashboard

# Initialize
dashboard = TrainingDashboard('/path/to/ROLLOUT_DIR')

# Get upsampling candidates
top_candidates = dashboard.upsampling_data.head(10)
print(top_candidates[['prompt_hash', 'upsampling_score', 'final_score']])

# Get hardest instruction types
hard_instructions = dashboard.inst_type_metrics.groupby('instruction_type')['success_rate'].mean().sort_values().head(5)
print(hard_instructions)

# Generate dashboard
dashboard.save_combined_dashboard('dashboard.html')
```

## Key Metrics Explained

### Upsampling Score
- **Higher = more need for upsampling**
- Combines: low final score + slow learning + forgetting events
- Top 10-20 candidates should be upsampled 2-5x in next run

### Iterations to Learn
- How many iterations until first success
- Use for curriculum ordering (easy → hard)

### Forgetting Events
- Times a prompt went from success → failure
- High count = unstable learning

### Plateau Status
- `is_plateau = True` when variance < 0.01 in last 3 iterations
- If plateaued at low score → upsample or remove
- If plateaued at high score → keep

## Decision Tree

```
Is training not converging well?
├─ Yes → Check "Reward Case Evolution"
│         ├─ Still mostly penalties/failures → Need more iterations or better reward
│         └─ Good bonus rate but low scores → Check upsampling candidates
│
└─ No → Want to improve data efficiency?
          ├─ Check "Sample Efficiency" chart
          │   └─ Focus training on bottom-right quadrant (hard prompts)
          │
          ├─ Check "Upsampling Candidates" table
          │   └─ Upsample top 10-20 prompts 2-5x
          │
          └─ Check "Plateau Detection"
              └─ Remove prompts plateaued at low scores
```

## Next Steps

1. Run your training and save rollouts to `ROLLOUT_DIR/{iteration}.jsonl`
2. Generate dashboard: `python -m analysis.run_analysis ROLLOUT_DIR --combined-dashboard`
3. Open `analysis_output/training_dashboard.html` in browser
4. Review upsampling candidates and make data adjustments
5. Re-run training with updated dataset
6. Compare new dashboard with previous run

See `README.md` for full documentation.
