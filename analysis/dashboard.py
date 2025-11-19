"""Main dashboard combining all training analysis charts."""

from pathlib import Path
from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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

    def export_all_metrics(self, output_dir: str | Path) -> None:
        """Export all computed metrics to CSV files.

        Args:
            output_dir: Directory to save CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("EXPORTING METRICS TO CSV")
        print("="*80 + "\n")

        # Export prompt-level metrics
        print(f"Exporting prompt_metrics.csv ({len(self.prompt_metrics)} rows)...")
        self.prompt_metrics.to_csv(output_dir / 'prompt_metrics.csv', index=False)

        # Export learning dynamics
        print(f"Exporting learning_dynamics.csv ({len(self.learning_dynamics)} rows)...")
        self.learning_dynamics.to_csv(output_dir / 'learning_dynamics.csv', index=False)

        # Export instruction type metrics
        print(f"Exporting instruction_type_metrics.csv ({len(self.inst_type_metrics)} rows)...")
        self.inst_type_metrics.to_csv(output_dir / 'instruction_type_metrics.csv', index=False)

        # Export reward distribution
        print(f"Exporting reward_distribution.csv ({len(self.reward_dist)} rows)...")
        self.reward_dist.to_csv(output_dir / 'reward_distribution.csv', index=False)

        # Export exploration metrics
        print(f"Exporting exploration_metrics.csv ({len(self.exploration_metrics)} rows)...")
        self.exploration_metrics.to_csv(output_dir / 'exploration_metrics.csv', index=False)

        # Export V_i vs S_i data
        print(f"Exporting vi_si_correlation.csv ({len(self.vi_si_data)} rows)...")
        self.vi_si_data.to_csv(output_dir / 'vi_si_correlation.csv', index=False)

        # Export instruction interference
        print(f"Exporting instruction_interference.csv ({len(self.interference_data)} rows)...")
        self.interference_data.to_csv(output_dir / 'instruction_interference.csv', index=False)

        # Export upsampling candidates
        print(f"Exporting upsampling_candidates.csv ({len(self.upsampling_data)} rows)...")
        self.upsampling_data.to_csv(output_dir / 'upsampling_candidates.csv', index=False)

        # Export instruction type summary (aggregated)
        print("Exporting instruction_type_summary.csv...")
        inst_summary = self.inst_type_metrics.groupby('instruction_type').agg({
            'success_rate': ['mean', 'std', 'min', 'max'],
            'num_samples': 'sum'
        }).round(4)
        inst_summary.columns = ['_'.join(col).strip() for col in inst_summary.columns.values]
        inst_summary = inst_summary.sort_values('success_rate_mean')
        inst_summary.to_csv(output_dir / 'instruction_type_summary.csv')

        # Export prompt hardness ranking (all prompts sorted by difficulty)
        print(f"Exporting prompt_hardness_ranking.csv ({len(self.learning_dynamics)} prompts)...")
        hardness_ranking = self.learning_dynamics[[
            'prompt_hash', 'iterations_to_learn', 'final_score', 'learning_rate',
            'num_forgetting_events', 'instruction_types', 'num_instructions'
        ]].sort_values('iterations_to_learn', ascending=False)
        hardness_ranking.to_csv(output_dir / 'prompt_hardness_ranking.csv', index=False)

        print(f"\n✓ All metrics exported to {output_dir}\n")

    def _save_chart(self, fig: go.Figure, name: str, output_dir: Path, format: str = 'html') -> None:
        """Save a single chart immediately.

        Args:
            fig: Plotly figure to save
            name: Chart name (without extension)
            output_dir: Directory to save to
            format: Output format ('html' or 'png')
        """
        output_path = output_dir / f"{name}.{format}"
        print(f"  💾 Saving {output_path.name}...")

        if format == 'html':
            fig.write_html(str(output_path))
        elif format == 'png':
            fig.write_image(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")

    def generate_all_charts(self, max_prompts: Optional[int] = None, skip_heatmaps: bool = False,
                            output_dir: Optional[Path] = None, format: str = 'html') -> dict:
        """Generate all charts, optionally saving immediately to avoid memory issues.

        Args:
            max_prompts: Maximum number of prompts to include in heatmaps (None = all)
            skip_heatmaps: If True, skip computationally expensive heatmaps
            output_dir: If provided, save charts immediately as they're generated
            format: Output format if saving ('html' or 'png')

        Returns:
            Dictionary mapping chart names to Plotly figures
        """
        charts = {}

        # Subsample data if needed
        if max_prompts is not None and len(self.learning_dynamics) > max_prompts:
            print(f"\n⚠️  Large dataset detected ({len(self.learning_dynamics)} prompts)")
            print(f"   Limiting heatmaps to {max_prompts} prompts for performance")
            print(f"   (Use --max-prompts to adjust or export full data to CSV)\n")

            # Sample diverse prompts: hardest, easiest, and random
            sorted_dynamics = self.learning_dynamics.sort_values('iterations_to_learn', ascending=False)
            n_hard = max_prompts // 3
            n_easy = max_prompts // 3
            n_random = max_prompts - n_hard - n_easy

            sampled_hashes = set()
            sampled_hashes.update(sorted_dynamics.head(n_hard)['prompt_hash'])
            sampled_hashes.update(sorted_dynamics.tail(n_easy)['prompt_hash'])
            sampled_hashes.update(sorted_dynamics.sample(n=n_random, random_state=42)['prompt_hash'])

            # Filter metrics
            filtered_prompt_metrics = self.prompt_metrics[
                self.prompt_metrics['prompt_hash'].isin(sampled_hashes)
            ]
            filtered_learning_dynamics = self.learning_dynamics[
                self.learning_dynamics['prompt_hash'].isin(sampled_hashes)
            ]
            filtered_exploration = self.exploration_metrics[
                self.exploration_metrics['prompt_hash'].isin(sampled_hashes)
            ]
        else:
            filtered_prompt_metrics = self.prompt_metrics
            filtered_learning_dynamics = self.learning_dynamics
            filtered_exploration = self.exploration_metrics

        if not skip_heatmaps:
            print("Generating prompt dynamics heatmap (F_i)...")
            fig = create_prompt_dynamics_heatmap(
                filtered_prompt_metrics,
                filtered_learning_dynamics,
                metric='mean_score',
                sort_by='first_success'
            )
            charts['prompt_dynamics_fi'] = fig
            if output_dir:
                self._save_chart(fig, 'prompt_dynamics_fi', output_dir, format)

            print("Generating prompt dynamics heatmap (V_i)...")
            fig = create_prompt_dynamics_heatmap(
                filtered_prompt_metrics,
                filtered_learning_dynamics,
                metric='mean_V_i',
                sort_by='first_success'
            )
            charts['prompt_dynamics_vi'] = fig
            if output_dir:
                self._save_chart(fig, 'prompt_dynamics_vi', output_dir, format)

            print("Generating prompt dynamics heatmap (S_i)...")
            fig = create_prompt_dynamics_heatmap(
                filtered_prompt_metrics,
                filtered_learning_dynamics,
                metric='mean_S_i',
                sort_by='first_success'
            )
            charts['prompt_dynamics_si'] = fig
            if output_dir:
                self._save_chart(fig, 'prompt_dynamics_si', output_dir, format)

            print("Generating prompt strict accuracy heatmap...")
            fig = create_prompt_dynamics_heatmap(
                filtered_prompt_metrics,
                filtered_learning_dynamics,
                metric='prompt_strict_acc',
                sort_by='first_success'
            )
            charts['prompt_accuracy'] = fig
            if output_dir:
                self._save_chart(fig, 'prompt_accuracy', output_dir, format)
        else:
            print("⏭️  Skipping prompt dynamics heatmaps (--skip-heatmaps enabled)")

        if not skip_heatmaps:
            print("Generating exploration heatmap...")
            fig = create_exploration_heatmap(
                filtered_exploration,
                filtered_learning_dynamics
            )
            charts['exploration'] = fig
            if output_dir:
                self._save_chart(fig, 'exploration', output_dir, format)

            print("Generating forgetting analysis...")
            fig = create_forgetting_analysis(
                filtered_prompt_metrics,
                filtered_learning_dynamics
            )
            charts['forgetting_analysis'] = fig
            if output_dir:
                self._save_chart(fig, 'forgetting_analysis', output_dir, format)
        else:
            print("⏭️  Skipping exploration and forgetting heatmaps")

        print("Generating instruction type heatmap...")
        fig = create_instruction_type_heatmap(self.inst_type_metrics)
        charts['instruction_types'] = fig
        if output_dir:
            self._save_chart(fig, 'instruction_types', output_dir, format)

        print("Generating reward case evolution...")
        fig = create_reward_case_evolution(self.reward_dist)
        charts['reward_case_evolution'] = fig
        if output_dir:
            self._save_chart(fig, 'reward_case_evolution', output_dir, format)

        print("Generating sample efficiency chart...")
        fig = create_sample_efficiency_chart(self.learning_dynamics)
        charts['sample_efficiency'] = fig
        if output_dir:
            self._save_chart(fig, 'sample_efficiency', output_dir, format)

        print("Generating V_i vs S_i correlation...")
        fig = create_vi_si_correlation(self.vi_si_data)
        charts['vi_si_correlation'] = fig
        if output_dir:
            self._save_chart(fig, 'vi_si_correlation', output_dir, format)

        print("Generating plateau detection chart...")
        fig = create_plateau_detection_chart(self.learning_dynamics)
        charts['plateau_detection'] = fig
        if output_dir:
            self._save_chart(fig, 'plateau_detection', output_dir, format)

        print("Generating instruction interference matrix...")
        fig = create_instruction_interference_matrix(
            self.interference_data
        )
        charts['instruction_interference'] = fig
        if output_dir:
            self._save_chart(fig, 'instruction_interference', output_dir, format)

        print("Generating upsampling candidates table...")
        fig = create_upsampling_candidates_table(
            self.upsampling_data,
            top_n=20
        )
        charts['upsampling_candidates'] = fig
        if output_dir:
            self._save_chart(fig, 'upsampling_candidates', output_dir, format)

        print("Generating score distribution evolution...")
        fig = create_score_distribution_evolution(
            self.score_distributions
        )
        charts['score_distribution'] = fig
        if output_dir:
            self._save_chart(fig, 'score_distribution', output_dir, format)

        return charts

    def save_all_charts(self, output_dir: str | Path, format: str = 'html',
                        max_prompts: Optional[int] = None, skip_heatmaps: bool = False) -> None:
        """Save all charts to files (saves immediately as generated to avoid memory issues).

        Args:
            output_dir: Directory to save charts
            format: Output format ('html' or 'png')
            max_prompts: Maximum number of prompts to include in heatmaps
            skip_heatmaps: If True, skip computationally expensive heatmaps
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save charts incrementally (saves each chart as it's created)
        self.generate_all_charts(
            max_prompts=max_prompts,
            skip_heatmaps=skip_heatmaps,
            output_dir=output_dir,
            format=format
        )

        print(f"\n✓ All charts saved to {output_dir}")

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

    def save_combined_dashboard(self, output_path: str | Path,
                                 max_prompts: Optional[int] = None, skip_heatmaps: bool = False) -> None:
        """Save a combined dashboard with all charts.

        Args:
            output_path: Path to save the dashboard HTML
            max_prompts: Maximum number of prompts to include in heatmaps
            skip_heatmaps: If True, skip computationally expensive heatmaps
        """
        charts = self.generate_all_charts(max_prompts=max_prompts, skip_heatmaps=skip_heatmaps)

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
