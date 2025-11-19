"""Example usage of the analysis dashboard."""

from pathlib import Path
from analysis import TrainingDashboard

def main():
    """Example usage scenarios."""

    # Example 1: Basic usage - generate all charts
    print("Example 1: Generate combined dashboard")
    print("-" * 60)

    rollout_dir = Path("path/to/your/rollout_dir")  # Replace with actual path

    # Check if directory exists
    if not rollout_dir.exists():
        print(f"Warning: {rollout_dir} does not exist.")
        print("Please update the path in this example script.")
        return

    # Initialize dashboard
    dashboard = TrainingDashboard(rollout_dir)

    # Print summary statistics
    dashboard.print_summary_statistics()

    # Generate combined dashboard
    output_path = Path("training_dashboard.html")
    dashboard.save_combined_dashboard(output_path)
    print(f"\n✓ Dashboard saved to {output_path}")

    # Example 2: Access specific metrics for custom analysis
    print("\n\nExample 2: Custom analysis with metrics")
    print("-" * 60)

    # Get top 10 hardest prompts
    hard_prompts = dashboard.learning_dynamics.nlargest(10, 'iterations_to_learn')
    print("\nTop 10 hardest prompts (by iterations to learn):")
    for i, row in hard_prompts.iterrows():
        print(f"  {row['prompt_hash'][:12]}: {row['iterations_to_learn']:.0f} iterations, "
              f"final score: {row['final_score']:.2f}")

    # Get prompts with most forgetting
    forgetful_prompts = dashboard.learning_dynamics.nlargest(5, 'num_forgetting_events')
    print("\nTop 5 prompts with most forgetting:")
    for i, row in forgetful_prompts.iterrows():
        print(f"  {row['prompt_hash'][:12]}: {row['num_forgetting_events']} events, "
              f"final score: {row['final_score']:.2f}")

    # Example 3: Analyze specific instruction types
    print("\n\nExample 3: Instruction type analysis")
    print("-" * 60)

    # Get average success rate per instruction type
    inst_success = dashboard.inst_type_metrics.groupby('instruction_type')['success_rate'].mean()
    inst_success = inst_success.sort_values()

    print("\nTop 5 hardest instruction types:")
    for inst, rate in inst_success.head(5).items():
        print(f"  {inst}: {rate:.1%}")

    print("\nTop 5 easiest instruction types:")
    for inst, rate in inst_success.tail(5).items():
        print(f"  {inst}: {rate:.1%}")

    # Example 4: Generate specific charts
    print("\n\nExample 4: Generate individual charts")
    print("-" * 60)

    from analysis.charts import (
        create_sample_efficiency_chart,
        create_upsampling_candidates_table,
    )

    # Generate sample efficiency chart
    efficiency_chart = create_sample_efficiency_chart(dashboard.learning_dynamics)
    efficiency_chart.write_html("sample_efficiency.html")
    print("✓ Sample efficiency chart saved to sample_efficiency.html")

    # Generate upsampling candidates table
    upsampling_table = create_upsampling_candidates_table(dashboard.upsampling_data, top_n=10)
    upsampling_table.write_html("upsampling_candidates.html")
    print("✓ Upsampling candidates table saved to upsampling_candidates.html")

    # Example 5: Curriculum learning decisions
    print("\n\nExample 5: Curriculum learning recommendations")
    print("-" * 60)

    # Get prompts sorted by difficulty (for curriculum ordering)
    curriculum_order = dashboard.learning_dynamics.sort_values('iterations_to_learn')

    print("\nCurriculum ordering (easiest to hardest):")
    print("First 5 prompts to introduce:")
    for i, row in curriculum_order.head(5).iterrows():
        print(f"  {i+1}. {row['prompt_hash'][:12]}: {row['iterations_to_learn']:.0f} iters, "
              f"{row['num_instructions']} instructions")

    print("\nLast 5 prompts to introduce (hardest):")
    for i, row in curriculum_order.tail(5).iterrows():
        print(f"  {i+1}. {row['prompt_hash'][:12]}: {row['iterations_to_learn']:.0f} iters, "
              f"{row['num_instructions']} instructions")

    # Example 6: Upsampling decisions
    print("\n\nExample 6: Data upsampling recommendations")
    print("-" * 60)

    top_upsampling = dashboard.upsampling_data.head(10)
    print("\nTop 10 prompts to upsample in next run:")
    for i, row in top_upsampling.iterrows():
        print(f"  {i+1}. {row['prompt_hash'][:12]}: "
              f"upsampling_score={row['upsampling_score']:.2f}, "
              f"final_score={row['final_score']:.2f}, "
              f"learning_rate={row['learning_rate']:.4f}")

    # Example 7: Plateau analysis
    print("\n\nExample 7: Plateau analysis")
    print("-" * 60)

    plateaued = dashboard.learning_dynamics[dashboard.learning_dynamics['is_plateau']]
    print(f"\nTotal plateaued prompts: {len(plateaued)}")

    # Split by final score
    good_plateau = plateaued[plateaued['final_score'] >= 1.0]
    bad_plateau = plateaued[plateaued['final_score'] < 0.0]

    print(f"Plateaued at good performance: {len(good_plateau)} (can keep)")
    print(f"Plateaued at bad performance: {len(bad_plateau)} (consider removing or upsampling)")

    if len(bad_plateau) > 0:
        print("\nBad plateau prompts (candidates for removal/upsampling):")
        for i, row in bad_plateau.head(5).iterrows():
            print(f"  {row['prompt_hash'][:12]}: final_score={row['final_score']:.2f}")

    # Example 8: Exploration analysis
    print("\n\nExample 8: Exploration analysis")
    print("-" * 60)

    # Get latest iteration exploration metrics
    latest_iter = dashboard.exploration_metrics['iteration'].max()
    latest_exploration = dashboard.exploration_metrics[
        dashboard.exploration_metrics['iteration'] == latest_iter
    ]

    avg_variance = latest_exploration['std_score'].mean()
    avg_consistency = latest_exploration['consistency_score'].mean()

    print(f"\nIteration {latest_iter} exploration metrics:")
    print(f"Average score variance: {avg_variance:.3f}")
    print(f"Average consistency: {avg_consistency:.3f}")

    if avg_variance < 0.1:
        print("⚠️  Low variance - might be over-exploiting (not exploring enough)")
    elif avg_variance > 0.5:
        print("⚠️  High variance - might be under-exploiting (too random)")
    else:
        print("✓ Healthy exploration-exploitation balance")


if __name__ == '__main__':
    main()
