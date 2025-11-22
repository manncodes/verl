#!/usr/bin/env python3
"""
Example usage of the bucketing script as a library.
This shows how to use the RolloutBucketer class programmatically.
"""

from pathlib import Path
from bucket_rollouts_by_difficulty import RolloutBucketer, DifficultyMetrics, calculate_percentiles


def example_1_basic_bucketing():
    """Example 1: Basic percentile bucketing."""
    print("\n" + "="*60)
    print("Example 1: Basic Percentile Bucketing")
    print("="*60)

    bucketer = RolloutBucketer(
        input_dir="./rollouts",
        output_dir="./bucketed_output",
        metric="mean_reward"
    )

    bucketer.run_percentile_bucketing(
        num_buckets=3,
        bucket_names=["easy", "medium", "hard"]
    )


def example_2_threshold_bucketing():
    """Example 2: Threshold-based bucketing."""
    print("\n" + "="*60)
    print("Example 2: Threshold-based Bucketing")
    print("="*60)

    bucketer = RolloutBucketer(
        input_dir="./rollouts",
        output_dir="./threshold_output",
        metric="mean_reward"
    )

    bucketer.run_threshold_bucketing(
        thresholds=[0.3, 0.7],
        bucket_names=["low", "medium", "high"]
    )


def example_3_custom_processing():
    """Example 3: Custom processing with manual control."""
    print("\n" + "="*60)
    print("Example 3: Custom Processing")
    print("="*60)

    bucketer = RolloutBucketer(
        input_dir="./rollouts",
        output_dir="./custom_output",
        metric="pass_at_k"
    )

    # Load and process manually
    rollouts = bucketer.load_rollouts()
    print(f"Loaded {len(rollouts)} rollouts")

    # Calculate difficulties
    rollouts_with_difficulty = bucketer.calculate_difficulties(rollouts)

    # Analyze distribution
    difficulties = [d for _, d in rollouts_with_difficulty]
    print(f"\nDifficulty statistics:")
    print(f"  Min: {min(difficulties):.4f}")
    print(f"  Max: {max(difficulties):.4f}")
    print(f"  Mean: {sum(difficulties) / len(difficulties):.4f}")

    # Custom bucketing: Top 10%, middle 80%, bottom 10%
    p10, p90 = calculate_percentiles(difficulties, [10, 90])

    buckets = {
        "top_10_percent": [],
        "middle_80_percent": [],
        "bottom_10_percent": []
    }

    for rollout, difficulty in rollouts_with_difficulty:
        if difficulty >= p90:
            buckets["top_10_percent"].append(rollout)
        elif difficulty <= p10:
            buckets["bottom_10_percent"].append(rollout)
        else:
            buckets["middle_80_percent"].append(rollout)

    bucketer.save_buckets(buckets, prefix="custom_")

    print(f"\nCustom bucket sizes:")
    for name, rollout_list in buckets.items():
        print(f"  {name}: {len(rollout_list)} rollouts")


def example_4_multiple_metrics():
    """Example 4: Compare multiple difficulty metrics."""
    print("\n" + "="*60)
    print("Example 4: Multiple Metrics Comparison")
    print("="*60)

    metrics = ["mean_reward", "variance", "pass_at_k"]

    for metric in metrics:
        print(f"\n--- Bucketing by {metric} ---")
        bucketer = RolloutBucketer(
            input_dir="./rollouts",
            output_dir=f"./comparison_{metric}",
            metric=metric
        )

        bucketer.run_percentile_bucketing(
            num_buckets=3,
            output_prefix=f"{metric}_"
        )


def example_5_curriculum_stages():
    """Example 5: Create 5-stage curriculum."""
    print("\n" + "="*60)
    print("Example 5: 5-Stage Curriculum")
    print("="*60)

    bucketer = RolloutBucketer(
        input_dir="./rollouts",
        output_dir="./curriculum",
        metric="mean_reward"
    )

    bucketer.run_percentile_bucketing(
        num_buckets=5,
        bucket_names=[
            "stage_1_warmup",
            "stage_2_basic",
            "stage_3_intermediate",
            "stage_4_advanced",
            "stage_5_expert"
        ]
    )


def example_6_analyze_single_rollout():
    """Example 6: Analyze difficulty metrics for a single rollout."""
    print("\n" + "="*60)
    print("Example 6: Single Rollout Analysis")
    print("="*60)

    # Example rollout structure (like the one you provided)
    example_rollout = {
        "num": 4127,
        "reward": [
            {"score": 0},
            {"score": 1.25},
            {"score": 0},
            {"score": 0},
            {"score": 0},
            {"score": 1.25},
            {"score": 0},
            {"score": 0}
        ]
    }

    print("Rollout reward scores:", [r["score"] for r in example_rollout["reward"]])
    print("\nDifficulty metrics:")
    print(f"  mean_reward: {DifficultyMetrics.mean_reward(example_rollout):.4f}")
    print(f"  max_reward: {DifficultyMetrics.max_reward(example_rollout):.4f}")
    print(f"  min_reward: {DifficultyMetrics.min_reward(example_rollout):.4f}")
    print(f"  variance: {DifficultyMetrics.reward_variance(example_rollout):.4f}")
    print(f"  pass_at_k: {DifficultyMetrics.pass_at_k(example_rollout, threshold=0.5):.4f}")
    print(f"  success_rate: {DifficultyMetrics.success_rate(example_rollout, threshold=0.5):.4f}")
    print(f"  inverse_mean: {DifficultyMetrics.inverse_mean_reward(example_rollout):.4f}")


def example_7_fine_grained_buckets():
    """Example 7: Fine-grained bucketing with 10 levels."""
    print("\n" + "="*60)
    print("Example 7: Fine-Grained 10-Level Bucketing")
    print("="*60)

    bucketer = RolloutBucketer(
        input_dir="./rollouts",
        output_dir="./fine_grained",
        metric="mean_reward"
    )

    bucketer.run_percentile_bucketing(
        num_buckets=10,
        bucket_names=[f"difficulty_level_{i}" for i in range(10)]
    )


if __name__ == "__main__":
    import sys

    # Run specific example if provided, otherwise run example 6 (analysis)
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = {
            1: example_1_basic_bucketing,
            2: example_2_threshold_bucketing,
            3: example_3_custom_processing,
            4: example_4_multiple_metrics,
            5: example_5_curriculum_stages,
            6: example_6_analyze_single_rollout,
            7: example_7_fine_grained_buckets,
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example number: {example_num}")
            print(f"Available examples: {list(examples.keys())}")
    else:
        # Run the analysis example by default (doesn't require actual data)
        print("Running example 6 (single rollout analysis)")
        print("To run other examples: python example_bucketing_usage.py <example_number>")
        example_6_analyze_single_rollout()
