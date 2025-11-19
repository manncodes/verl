# IFEval+GRPO Training Analysis Dashboard

Comprehensive observability and analysis system for IFEval+GRPO reinforcement learning training.

## Overview

This analysis system provides multi-granularity insights into your RL training dynamics, helping you make informed decisions about:

- **Data upsampling/downsampling**: Identify hard prompts that need more samples
- **Curriculum learning**: Order prompts by difficulty and learning dynamics
- **Forgetting detection**: Track catastrophic forgetting events
- **Instruction analysis**: Understand which instruction types are hard/easy
- **Reward composition**: Monitor bonus/penalty distribution over time
- **Exploration-exploitation**: Track variance and consistency across rollouts

## Installation

```bash
# Install required dependencies
pip install plotly pandas numpy
```

## Quick Start

### 1. Command Line Interface

```bash
# Analyze all iterations and generate combined dashboard
python -m analysis.run_analysis /path/to/rollout_dir --combined-dashboard

# Analyze specific iterations
python -m analysis.run_analysis /path/to/rollout_dir --iterations 0 1 2 3 --combined-dashboard

# Generate individual chart files (HTML)
python -m analysis.run_analysis /path/to/rollout_dir --output-dir ./outputs

# Generate PNG charts
python -m analysis.run_analysis /path/to/rollout_dir --format png

# Print summary statistics only
python -m analysis.run_analysis /path/to/rollout_dir --summary-only
```

### 2. Programmatic Usage

```python
from analysis import TrainingDashboard

# Initialize dashboard
dashboard = TrainingDashboard('/path/to/rollout_dir')

# Print summary statistics
dashboard.print_summary_statistics()

# Generate all charts
charts = dashboard.generate_all_charts()

# Save combined dashboard
dashboard.save_combined_dashboard('training_dashboard.html')

# Access individual metrics
prompt_metrics = dashboard.prompt_metrics
learning_dynamics = dashboard.learning_dynamics
upsampling_candidates = dashboard.upsampling_data
```

## Charts & Insights

### 📊 Core Training Dynamics

#### 1. **Prompt Dynamics Heatmaps** (4 variants)
- **F_i (Final Reward)**: Overall training progress per prompt
- **V_i (Instruction Accuracy)**: Pure constraint following
- **S_i (Preference Score)**: Quality without constraints
- **Prompt Strict Accuracy**: Binary success/failure
- **Sorted by**: First success iteration (earliest learners first)
- **Use for**: Identify learning progression, spot plateaus, detect forgetting

#### 2. **Instruction Type Heatmap**
- Shows success rate per instruction type over iterations
- Sorted by average success rate (hardest first)
- Difficulty tiers: Hard, Medium, Easy
- **Use for**: Understand which instruction types need more training data

#### 3. **Reward Case Evolution** (Stacked Area)
- Distribution of bonus/penalty/failure cases over time
- Shows shift from penalties → bonuses during training
- **Use for**: Monitor training stability and reward composition

### 🎯 Curriculum Learning & Data Selection

#### 4. **Sample Efficiency Scatter**
- X-axis: Iterations to first success
- Y-axis: Final score
- Color: Number of instructions (complexity)
- **Quadrants**:
  - **Top-Left (Fast Learners)**: Keep in dataset
  - **Top-Right (Slow Learners)**: Upsample or curriculum
  - **Bottom-Right (Hard/Failed)**: Upsample or remove
  - **Bottom-Left (Quick Failures)**: Check data quality
- **Use for**: Decide which prompts to upsample/downsample/remove

#### 5. **Plateau Detection**
- Bar chart of recent variance per prompt
- Red bars = plateaued (variance < threshold)
- **Use for**: Identify prompts that stopped improving (candidates for removal/upsampling)

#### 6. **Upsampling Candidates Table**
- Top 20 prompts ranked by upsampling priority
- Factors: Low final score, slow learning, forgetting events
- Color-coded: Red (high priority), Orange (medium), Yellow (low)
- **Use for**: Direct guidance on which prompts to oversample in next run

### 🔍 Exploration & Stability

#### 7. **Exploration Heatmap**
- Variance across N rollouts per prompt
- High variance = exploring, Low variance = exploiting
- **Use for**: Ensure adequate exploration across prompts

#### 8. **Forgetting Analysis**
- Heatmap of accuracy changes iteration-to-iteration
- Red cells = forgetting events (accuracy drops)
- Sorted by forgetting frequency
- **Use for**: Track catastrophic forgetting, identify unstable prompts

#### 9. **Score Distribution Evolution** (Violin Plots)
- Shows score distribution per iteration
- Tighter distributions = more consistent training
- **Use for**: Monitor convergence and stability

### 🔬 Reward Analysis

#### 10. **V_i vs S_i Correlation Scatter**
- Shows relationship between instruction following (V_i) and quality (S_i)
- Color: Iteration (temporal evolution)
- Threshold line: α = 0.5 (bonus/penalty boundary)
- **Quadrants**:
  - High V_i + High S_i → Bonus (green)
  - High V_i + Low S_i → Penalty (orange)
- **Use for**: Understand if constraints and quality are aligned or conflicting

### 🧩 Instruction Analysis

#### 11. **Instruction Interference Matrix**
- Heatmap showing joint success rate for instruction pairs
- Low values (red) = instruction pairs that conflict
- **Use for**: Identify instruction combinations that are hard to satisfy together

## Data Format

Your rollout directory should contain JSONL files named `{iteration}.jsonl`:

```
ROLLOUT_DIR/
├── 0.jsonl
├── 1.jsonl
├── 2.jsonl
└── ...
```

Each line in the JSONL file should have this structure:

```json
{
  "input": "system\\n\\n...",
  "output": "...",
  "score": 1.4,
  "V_i": 0.4,
  "S_i": 0.9999818801879883,
  "alpha_threshold": 0.5,
  "reward_case": "V_i > 0 and S_i > α (bonus)",
  "prompt_strict_acc": 0.0,
  "inst_strict_acc": 0.4,
  "num_instructions": 5,
  "num_followed": 2,
  "format_score": 0.0,
  "gts": {
    "instruction_id_list": ["keywords:letter_frequency", "detectable_format:title", ...],
    "kwargs": [...]
  }
}
```

## Metrics Glossary

### Prompt-Level Metrics
- **mean_score**: Average F_i (final reward) across N rollouts
- **std_score**: Standard deviation of scores (exploration indicator)
- **mean_V_i**: Average instruction accuracy
- **mean_S_i**: Average preference score
- **prompt_strict_acc**: Whether ANY rollout achieved 100% instruction accuracy
- **best_mean_gap**: Max score - mean score (exploration quality)
- **consistency_score**: 1 / (1 + std_score)

### Learning Dynamics
- **first_success_iter**: First iteration where prompt succeeded
- **iterations_to_learn**: How long it took to first success
- **num_forgetting_events**: Times accuracy dropped after success
- **learning_rate**: Slope of score improvement over time
- **recent_variance**: Variance in last 3 iterations (plateau indicator)
- **is_plateau**: Boolean flag for plateaued prompts

### Upsampling Score
Composite metric combining:
- `(1 - final_score) * 2` - Low final score
- `(-learning_rate) * 1` - Slow/negative learning
- `num_forgetting_events * 0.5` - Instability

Higher score = higher upsampling priority

## Curriculum Learning Workflow

1. **Run training** with current dataset
2. **Generate dashboard**: `python -m analysis.run_analysis ROLLOUT_DIR --combined-dashboard`
3. **Analyze insights**:
   - Check **Sample Efficiency** chart for quadrant placement
   - Review **Upsampling Candidates** table for top priorities
   - Check **Instruction Type Heatmap** for underrepresented types
   - Review **Plateau Detection** for stalled prompts
4. **Make decisions**:
   - **Upsample**: Hard prompts in bottom-right quadrant
   - **Downsample**: Easy prompts that learned quickly
   - **Remove**: Plateaued prompts with low final scores
   - **Curriculum order**: Sort by `iterations_to_learn` or `first_success_iter`
5. **Repeat** training with adjusted dataset

## Advanced Usage

### Custom Metrics

```python
from analysis import RolloutDataLoader, MetricsComputer

loader = RolloutDataLoader('/path/to/rollout_dir')
loader.load_iterations()

metrics = MetricsComputer(loader)

# Access raw data
df = loader.to_dataframe()

# Custom analysis
custom_metric = df.groupby('iteration')['score'].quantile(0.9)
```

### Filtering Specific Prompts

```python
# Analyze only prompts with specific instruction types
dashboard = TrainingDashboard('/path/to/rollout_dir')
df = dashboard.metrics.df

# Filter by instruction type
ifeval_prompts = df[df['instruction_types'].str.contains('keywords:frequency')]

# Get learning dynamics for filtered prompts
filtered_hashes = ifeval_prompts['prompt_hash'].unique()
filtered_dynamics = dashboard.learning_dynamics[
    dashboard.learning_dynamics['prompt_hash'].isin(filtered_hashes)
]
```

## Output Structure

```
analysis_output/
├── prompt_dynamics_fi.html       # F_i heatmap
├── prompt_dynamics_vi.html       # V_i heatmap
├── prompt_dynamics_si.html       # S_i heatmap
├── prompt_accuracy.html          # Strict accuracy heatmap
├── instruction_types.html        # Instruction type success
├── reward_case_evolution.html    # Reward composition
├── sample_efficiency.html        # Curriculum decisions
├── exploration.html              # Exploration heatmap
├── vi_si_correlation.html        # V_i vs S_i scatter
├── plateau_detection.html        # Plateau detection
├── forgetting_analysis.html      # Forgetting tracker
├── instruction_interference.html # Instruction conflicts
├── upsampling_candidates.html    # Top upsampling priorities
└── score_distribution.html       # Score distribution evolution
```

Or single combined dashboard:
```
training_dashboard.html           # All charts in one file with tabs
```

## Architecture

```
analysis/
├── __init__.py                   # Module exports
├── data_loader.py                # JSONL parsing and grouping
├── metrics_computer.py           # All metric computations
├── dashboard.py                  # Main dashboard class
├── run_analysis.py               # CLI entry point
└── charts/                       # Individual chart modules
    ├── prompt_dynamics_heatmap.py
    ├── instruction_type_heatmap.py
    ├── reward_case_evolution.py
    ├── sample_efficiency.py
    ├── exploration_heatmap.py
    ├── vi_si_correlation.py
    ├── plateau_detection.py
    ├── forgetting_analysis.py
    ├── instruction_interference.py
    ├── upsampling_candidates.py
    └── score_distribution.py
```

## Tips & Best Practices

1. **Start with summary statistics**: Use `--summary-only` for quick insights
2. **Focus on actionable charts**: Sample Efficiency, Upsampling Candidates, Plateau Detection
3. **Compare across runs**: Generate dashboards for multiple experiments
4. **Iterate quickly**: Use `--iterations 0 1 2` to analyze early training
5. **Combine with training logs**: Cross-reference with loss curves and other metrics

## Troubleshooting

**Issue**: Chart shows "No data"
- **Solution**: Ensure JSONL files have the correct format with all required fields

**Issue**: Memory error with large datasets
- **Solution**: Analyze specific iterations using `--iterations` flag

**Issue**: Heatmap too large
- **Solution**: Charts auto-scale height based on number of prompts

## Contributing

To add a new chart:

1. Create module in `analysis/charts/your_chart.py`
2. Implement `create_your_chart(data: pd.DataFrame) -> go.Figure`
3. Add to `charts/__init__.py`
4. Add to `dashboard.py::generate_all_charts()`
5. Add to combined dashboard sections

## License

Same as parent project.
