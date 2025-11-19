"""Entry point script for running training analysis."""

import argparse
from pathlib import Path
from .dashboard import TrainingDashboard


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze IFEval+GRPO training rollouts and generate dashboard'
    )
    parser.add_argument(
        'rollout_dir',
        type=str,
        help='Directory containing {iteration}.jsonl files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output',
        help='Directory to save analysis outputs (default: analysis_output)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        nargs='+',
        default=None,
        help='Specific iterations to analyze (default: all)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['html', 'png'],
        default='html',
        help='Output format for individual charts (default: html)'
    )
    parser.add_argument(
        '--combined-dashboard',
        action='store_true',
        help='Generate a single combined dashboard HTML file'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary statistics without generating charts'
    )

    args = parser.parse_args()

    # Initialize dashboard
    print(f"Loading rollout data from {args.rollout_dir}...")
    dashboard = TrainingDashboard(args.rollout_dir, args.iterations)

    # Print summary statistics
    dashboard.print_summary_statistics()

    if args.summary_only:
        return

    # Generate outputs
    output_dir = Path(args.output_dir)

    if args.combined_dashboard:
        # Generate single combined dashboard
        output_path = output_dir / 'training_dashboard.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating combined dashboard...")
        dashboard.save_combined_dashboard(output_path)
        print(f"\n✓ Dashboard saved to: {output_path}")
    else:
        # Generate individual chart files
        print(f"\nGenerating individual charts...")
        dashboard.save_all_charts(output_dir, format=args.format)
        print(f"\n✓ Individual charts saved to: {output_dir}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
