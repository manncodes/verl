# Analysis Dashboard - Complete Feature List

## 📊 14 Interactive Charts

### 1-4. Prompt Dynamics Heatmaps (4 variants)
- **F_i (Final Reward)** - Overall training progress
- **V_i (Instruction Accuracy)** - Constraint following
- **S_i (Preference Score)** - Quality metrics
- **Prompt Strict Accuracy** - Binary success/failure
- Sorted by first success iteration
- Hover shows detailed metrics per prompt/iteration

### 5. Instruction Type Heatmap
- Success rate per instruction type over time
- Sorted by difficulty (hardest first)
- Difficulty tiers: Hard, Medium, Easy
- Hover shows V_i stats and sample counts

### 6. Reward Case Evolution (Stacked Area)
- Distribution of bonus/penalty/failure over iterations
- Shows training stability
- Color-coded by reward case type

### 7. Sample Efficiency Scatter
- X: Iterations to learn, Y: Final score
- Color: Number of instructions
- 4 quadrants for decision-making
- Direct guidance for upsampling/curriculum

### 8. Exploration Heatmap
- Variance across N rollouts per prompt
- High variance = exploring, Low = exploiting
- Same sorting as prompt dynamics

### 9. V_i vs S_i Correlation Scatter
- Shows alignment between constraints and quality
- Color by iteration (temporal evolution)
- Threshold line for bonus/penalty boundary
- Correlation coefficient displayed

### 10. Plateau Detection Bar Chart
- Recent variance per prompt
- Color-coded: Red = plateaued, Green = active
- Threshold line and count statistics

### 11. Forgetting Analysis Heatmap
- Iteration-to-iteration accuracy changes
- Red cells = forgetting events
- Sorted by forgetting frequency
- Tracks catastrophic forgetting

### 12. Instruction Interference Matrix
- Joint success rate for instruction pairs
- Symmetric heatmap
- Red cells = conflicting instructions
- Hover shows co-occurrence counts

### 13. Upsampling Candidates Table
- Top 20 prompts ranked by priority
- Color-coded: Red (high), Orange (medium), Yellow (low)
- Shows upsampling score, learning rate, forgetting events
- Direct actionable guidance

### 14. Score Distribution Evolution (Violin Plots)
- Distribution per iteration
- Shows convergence and consistency
- Mean line and box plot included

## 📈 Computed Metrics

### Prompt-Level (per prompt, per iteration)
- `mean_score`, `std_score`, `max_score`, `min_score`
- `mean_V_i`, `std_V_i`
- `mean_S_i`, `std_S_i`
- `prompt_strict_acc`, `mean_prompt_acc`
- `num_rollouts`
- `best_mean_gap` (exploration quality)
- `consistency_score`

### Learning Dynamics (per prompt, across iterations)
- `first_success_iter`
- `iterations_to_learn`
- `num_forgetting_events`
- `learning_rate` (slope of score improvement)
- `final_score`, `initial_score`, `score_improvement`
- `recent_variance` (last 3 iterations)
- `is_plateau` (boolean flag)

### Instruction-Type Level (per type, per iteration)
- `success_rate`
- `mean_V_i`, `std_V_i`
- `num_samples`

### Reward Distribution (per iteration)
- Count and percentage per reward case
- Temporal evolution tracking

### Instruction Interference
- `co_occurrence_count`
- `joint_success_rate`
- Pairwise instruction relationships

### Upsampling Priority
- `upsampling_score` (composite metric)
- Ranking and recommendations

## 🎯 Use Cases

### Curriculum Learning
1. **Sample Efficiency Chart** - quadrant-based decisions
2. **Learning Dynamics** - sort by iterations_to_learn
3. **Plateau Detection** - identify stalled prompts

### Data Upsampling/Downsampling
1. **Upsampling Candidates Table** - top priorities
2. **Sample Efficiency Chart** - hard prompts (bottom-right)
3. **Instruction Type Heatmap** - underrepresented types

### Forgetting Detection
1. **Forgetting Analysis Heatmap** - visual tracking
2. **Learning Dynamics** - forgetting event counts
3. **Prompt Dynamics** - accuracy drops over time

### Instruction Analysis
1. **Instruction Type Heatmap** - difficulty ranking
2. **Instruction Interference Matrix** - conflicting pairs
3. **Instruction-Type Metrics** - detailed statistics

### Reward Composition
1. **Reward Case Evolution** - bonus/penalty distribution
2. **V_i vs S_i Correlation** - alignment analysis
3. **Score Distribution** - convergence tracking

### Exploration-Exploitation
1. **Exploration Heatmap** - variance tracking
2. **Prompt Metrics** - best-mean gap
3. **Score Distribution** - consistency over time

## 🛠️ Interfaces

### Command Line
```bash
python -m analysis.run_analysis ROLLOUT_DIR [OPTIONS]
```

Options:
- `--output-dir DIR` - Output directory
- `--iterations [N...]` - Specific iterations
- `--format {html,png}` - Output format
- `--combined-dashboard` - Single HTML file
- `--summary-only` - Statistics only

### Programmatic API
```python
from analysis import TrainingDashboard

dashboard = TrainingDashboard(rollout_dir)
dashboard.print_summary_statistics()
charts = dashboard.generate_all_charts()
dashboard.save_combined_dashboard('output.html')
dashboard.save_all_charts('output_dir/')

# Access metrics
dashboard.prompt_metrics
dashboard.learning_dynamics
dashboard.upsampling_data
dashboard.inst_type_metrics
# ... and more
```

### Direct Chart Generation
```python
from analysis.charts import create_sample_efficiency_chart

fig = create_sample_efficiency_chart(learning_dynamics_df)
fig.write_html('chart.html')
```

## 📁 Data Format

Input: `ROLLOUT_DIR/{iteration}.jsonl`

Required fields per rollout:
- `input`, `output`
- `score`, `V_i`, `S_i`
- `alpha_threshold`, `reward_case`
- `prompt_strict_acc`, `inst_strict_acc`
- `num_instructions`, `num_followed`
- `format_score`
- `gts` (with `instruction_id_list` and `kwargs`)

## 🎨 Output Formats

### Combined Dashboard
- Single HTML file with tabs
- All 14 charts organized in 6 sections
- Interactive navigation
- Includes Plotly.js

### Individual Charts
- HTML files (interactive)
- PNG files (static)
- One file per chart

### Summary Statistics
- Text output to console
- Training overview
- Top hardest instructions
- Upsampling recommendations

## 🏗️ Architecture

```
analysis/
├── data_loader.py          # JSONL parsing, grouping
├── metrics_computer.py     # All metric calculations
├── dashboard.py            # Main dashboard class
├── run_analysis.py         # CLI entry point
├── example_usage.py        # Example scripts
└── charts/                 # 11 chart modules
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

## 🚀 Performance

- Handles 1000s of prompts
- Efficient pandas operations
- Auto-scaling chart heights
- Memory-efficient data structures
- Fast metric computation

## 🔮 Future Extensions

Potential additions:
- Real-time monitoring mode
- Comparison across multiple runs
- Statistical significance tests
- Automatic anomaly detection
- Export to TensorBoard
- Integration with wandb/mlflow
- Custom metric definitions
- Advanced filtering/querying
