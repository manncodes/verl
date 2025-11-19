"""Main dashboard combining all training analysis charts."""

from pathlib import Path
from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import RolloutDataLoader
from .metrics_computer import MetricsComputer
from .charts import (
    create_prompt_dynamics_heatmap,
    create_instruction_type_heatmap,
    create_reward_case_evolution,
    create_sample_efficiency_chart,
    create_exploration_heatmap,
    create_vi_si_correlation,
    create_plateau_detection_chart,
    create_forgetting_analysis,
    create_instruction_interference_matrix,
    create_upsampling_candidates_table,
    create_score_distribution_evolution,
)


class TrainingDashboard:
    """Interactive dashboard for RL training analysis."""

    def __init__(self, rollout_dir: str | Path, iterations: Optional[List[int]] = None):
        """Initialize dashboard.

        Args:
            rollout_dir: Directory containing {iteration}.jsonl files
            iterations: List of iteration numbers to load. If None, loads all.
        """
        self.loader = RolloutDataLoader(rollout_dir)
        self.loader.load_iterations(iterations)
        self.metrics = MetricsComputer(self.loader)

        # Pre-compute all metrics
        self.prompt_metrics = self.metrics.compute_prompt_level_metrics()
        self.learning_dynamics = self.metrics.compute_learning_dynamics()
        self.inst_type_metrics = self.metrics.compute_instruction_type_metrics()
        self.reward_dist = self.metrics.compute_reward_case_distribution()
        self.exploration_metrics = self.metrics.compute_exploration_metrics()
        self.vi_si_data = self.metrics.compute_vi_si_correlation()
        self.interference_data = self.metrics.compute_instruction_interference()
        self.upsampling_data = self.metrics.compute_upsampling_candidates()
        self.score_distributions = self.metrics.compute_score_distributions()

    def generate_all_charts(self) -> dict:
        """Generate all charts.

        Returns:
            Dictionary mapping chart names to Plotly figures
        """
        charts = {}

        print("Generating prompt dynamics heatmap (F_i)...")
        charts['prompt_dynamics_fi'] = create_prompt_dynamics_heatmap(
            self.prompt_metrics,
            self.learning_dynamics,
            metric='mean_score',
            sort_by='first_success'
        )

        print("Generating prompt dynamics heatmap (V_i)...")
        charts['prompt_dynamics_vi'] = create_prompt_dynamics_heatmap(
            self.prompt_metrics,
            self.learning_dynamics,
            metric='mean_V_i',
            sort_by='first_success'
        )

        print("Generating prompt dynamics heatmap (S_i)...")
        charts['prompt_dynamics_si'] = create_prompt_dynamics_heatmap(
            self.prompt_metrics,
            self.learning_dynamics,
            metric='mean_S_i',
            sort_by='first_success'
        )

        print("Generating prompt strict accuracy heatmap...")
        charts['prompt_accuracy'] = create_prompt_dynamics_heatmap(
            self.prompt_metrics,
            self.learning_dynamics,
            metric='prompt_strict_acc',
            sort_by='first_success'
        )

        print("Generating instruction type heatmap...")
        charts['instruction_types'] = create_instruction_type_heatmap(self.inst_type_metrics)

        print("Generating reward case evolution...")
        charts['reward_case_evolution'] = create_reward_case_evolution(self.reward_dist)

        print("Generating sample efficiency chart...")
        charts['sample_efficiency'] = create_sample_efficiency_chart(self.learning_dynamics)

        print("Generating exploration heatmap...")
        charts['exploration'] = create_exploration_heatmap(
            self.exploration_metrics,
            self.learning_dynamics
        )

        print("Generating V_i vs S_i correlation...")
        charts['vi_si_correlation'] = create_vi_si_correlation(self.vi_si_data)

        print("Generating plateau detection chart...")
        charts['plateau_detection'] = create_plateau_detection_chart(self.learning_dynamics)

        print("Generating forgetting analysis...")
        charts['forgetting_analysis'] = create_forgetting_analysis(
            self.prompt_metrics,
            self.learning_dynamics
        )

        print("Generating instruction interference matrix...")
        charts['instruction_interference'] = create_instruction_interference_matrix(
            self.interference_data
        )

        print("Generating upsampling candidates table...")
        charts['upsampling_candidates'] = create_upsampling_candidates_table(
            self.upsampling_data,
            top_n=20
        )

        print("Generating score distribution evolution...")
        charts['score_distribution'] = create_score_distribution_evolution(
            self.score_distributions
        )

        return charts

    def save_all_charts(self, output_dir: str | Path, format: str = 'html') -> None:
        """Save all charts to files.

        Args:
            output_dir: Directory to save charts
            format: Output format ('html' or 'png')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        charts = self.generate_all_charts()

        for name, fig in charts.items():
            output_path = output_dir / f"{name}.{format}"
            print(f"Saving {output_path}...")

            if format == 'html':
                fig.write_html(str(output_path))
            elif format == 'png':
                fig.write_image(str(output_path))
            else:
                raise ValueError(f"Unsupported format: {format}")

        print(f"\nAll charts saved to {output_dir}")

    def create_combined_dashboard(self) -> go.Figure:
        """Create a single HTML page with all charts.

        Returns:
            Plotly Figure with all charts in tabs/sections
        """
        from plotly.subplots import make_subplots

        charts = self.generate_all_charts()

        # Create a master HTML with all charts
        # Note: Plotly doesn't support native tabs, so we'll return all charts as separate sections
        # The HTML export will need to handle tabs via custom HTML/CSS/JS

        return charts

    def save_combined_dashboard(self, output_path: str | Path) -> None:
        """Save a combined dashboard with all charts.

        Args:
            output_path: Path to save the dashboard HTML
        """
        charts = self.generate_all_charts()

        # Create custom HTML with tabs
        html_parts = [
            """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>IFEval+GRPO Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .tabs {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
        }
        .tabs button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 14px;
        }
        .tabs button:hover {
            background-color: #ddd;
        }
        .tabs button.active {
            background-color: #ccc;
        }
        .tabcontent {
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
        }
        .tabcontent.active {
            display: block;
        }
        .chart-container {
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>IFEval+GRPO Training Dashboard</h1>
        <p>Comprehensive analysis of RL training dynamics</p>
    </div>
    <div class="tabs">
        <button class="tablinks active" onclick="openTab(event, 'overview')">Overview</button>
        <button class="tablinks" onclick="openTab(event, 'prompts')">Prompt Dynamics</button>
        <button class="tablinks" onclick="openTab(event, 'instructions')">Instructions</button>
        <button class="tablinks" onclick="openTab(event, 'rewards')">Rewards</button>
        <button class="tablinks" onclick="openTab(event, 'exploration')">Exploration</button>
        <button class="tablinks" onclick="openTab(event, 'curriculum')">Curriculum</button>
    </div>
"""
        ]

        # Overview Tab
        html_parts.append('<div id="overview" class="tabcontent active">')
        html_parts.append('<h2>Training Overview</h2>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['reward_case_evolution'].to_html(include_plotlyjs=False, div_id='reward_case'))
        html_parts.append('</div>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['score_distribution'].to_html(include_plotlyjs=False, div_id='score_dist'))
        html_parts.append('</div>')
        html_parts.append('</div>')

        # Prompt Dynamics Tab
        html_parts.append('<div id="prompts" class="tabcontent">')
        html_parts.append('<h2>Prompt-Level Training Dynamics</h2>')
        for i, (name, title) in enumerate([
            ('prompt_dynamics_fi', 'Final Reward (F_i)'),
            ('prompt_dynamics_vi', 'Instruction Accuracy (V_i)'),
            ('prompt_dynamics_si', 'Preference Score (S_i)'),
            ('prompt_accuracy', 'Prompt Strict Accuracy'),
        ]):
            html_parts.append(f'<div class="chart-container"><h3>{title}</h3>')
            html_parts.append(charts[name].to_html(include_plotlyjs=False, div_id=f'prompt_{i}'))
            html_parts.append('</div>')
        html_parts.append('</div>')

        # Instructions Tab
        html_parts.append('<div id="instructions" class="tabcontent">')
        html_parts.append('<h2>Instruction Analysis</h2>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['instruction_types'].to_html(include_plotlyjs=False, div_id='inst_types'))
        html_parts.append('</div>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['instruction_interference'].to_html(include_plotlyjs=False, div_id='inst_interference'))
        html_parts.append('</div>')
        html_parts.append('</div>')

        # Rewards Tab
        html_parts.append('<div id="rewards" class="tabcontent">')
        html_parts.append('<h2>Reward Analysis</h2>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['vi_si_correlation'].to_html(include_plotlyjs=False, div_id='vi_si'))
        html_parts.append('</div>')
        html_parts.append('</div>')

        # Exploration Tab
        html_parts.append('<div id="exploration" class="tabcontent">')
        html_parts.append('<h2>Exploration & Stability</h2>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['exploration'].to_html(include_plotlyjs=False, div_id='exploration'))
        html_parts.append('</div>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['forgetting_analysis'].to_html(include_plotlyjs=False, div_id='forgetting'))
        html_parts.append('</div>')
        html_parts.append('</div>')

        # Curriculum Tab
        html_parts.append('<div id="curriculum" class="tabcontent">')
        html_parts.append('<h2>Curriculum Learning & Data Selection</h2>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['sample_efficiency'].to_html(include_plotlyjs=False, div_id='efficiency'))
        html_parts.append('</div>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['plateau_detection'].to_html(include_plotlyjs=False, div_id='plateau'))
        html_parts.append('</div>')
        html_parts.append('<div class="chart-container">')
        html_parts.append(charts['upsampling_candidates'].to_html(include_plotlyjs=False, div_id='upsampling'))
        html_parts.append('</div>')
        html_parts.append('</div>')

        # JavaScript for tabs
        html_parts.append("""
    <script>
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].className = tabcontent[i].className.replace(" active", "");
        }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(tabName).className += " active";
        evt.currentTarget.className += " active";
    }
    </script>
</body>
</html>
""")

        # Write to file
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_parts))

        print(f"Combined dashboard saved to {output_path}")

    def print_summary_statistics(self) -> None:
        """Print summary statistics of the training run."""
        print("\n" + "="*80)
        print("TRAINING SUMMARY STATISTICS")
        print("="*80)

        # Overall stats
        total_prompts = len(self.learning_dynamics)
        total_iterations = len(self.prompt_metrics['iteration'].unique())
        print(f"\nTotal Prompts: {total_prompts}")
        print(f"Total Iterations: {total_iterations}")

        # Learning success
        learned_prompts = self.learning_dynamics[self.learning_dynamics['first_success_iter'].notna()]
        print(f"\nPrompts that learned (achieved success): {len(learned_prompts)} ({len(learned_prompts)/total_prompts*100:.1f}%)")

        if len(learned_prompts) > 0:
            avg_iters = learned_prompts['iterations_to_learn'].mean()
            print(f"Average iterations to learn: {avg_iters:.1f}")

        # Forgetting
        total_forgetting = self.learning_dynamics['num_forgetting_events'].sum()
        print(f"\nTotal forgetting events: {total_forgetting}")

        # Plateau
        plateaued = self.learning_dynamics['is_plateau'].sum()
        print(f"Prompts plateaued: {plateaued} ({plateaued/total_prompts*100:.1f}%)")

        # Instruction types
        print(f"\nUnique instruction types: {len(self.inst_type_metrics['instruction_type'].unique())}")

        # Hardest instructions
        print("\nTop 5 hardest instruction types:")
        avg_success = self.inst_type_metrics.groupby('instruction_type')['success_rate'].mean().sort_values()
        for inst, rate in avg_success.head(5).items():
            print(f"  - {inst}: {rate:.1%}")

        # Upsampling candidates
        print(f"\nTop 5 upsampling candidates:")
        for i, row in self.upsampling_data.head(5).iterrows():
            print(f"  - {row['prompt_hash'][:12]}: score={row['upsampling_score']:.2f}, final={row['final_score']:.2f}")

        print("\n" + "="*80 + "\n")
