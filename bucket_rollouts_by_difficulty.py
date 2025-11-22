#!/usr/bin/env python3
"""
Bucket JSONL rollouts by difficulty into curriculum-based datasets.

Supports multiple difficulty metrics:
- mean_reward: Average reward score across all responses
- max_reward: Maximum reward score achieved
- variance: Variance in reward scores (higher = more inconsistent)
- pass_at_k: Percentage of responses passing a threshold
- success_rate: Binary success rate (score > threshold)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Callable
import statistics
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not found. Using native Python for percentile calculations.")


def calculate_percentiles(data: List[float], percentiles: List[float]) -> List[float]:
    """
    Calculate percentiles from data.
    Uses numpy if available, otherwise uses native Python implementation.

    Args:
        data: Sorted or unsorted list of values
        percentiles: List of percentile values (0-100)

    Returns:
        List of percentile values
    """
    if HAS_NUMPY:
        return np.percentile(data, percentiles).tolist()
    else:
        # Native Python implementation
        sorted_data = sorted(data)
        n = len(sorted_data)
        results = []

        for p in percentiles:
            if p == 0:
                results.append(sorted_data[0])
            elif p == 100:
                results.append(sorted_data[-1])
            else:
                # Linear interpolation
                rank = p / 100 * (n - 1)
                lower_idx = int(rank)
                upper_idx = min(lower_idx + 1, n - 1)
                fraction = rank - lower_idx

                value = sorted_data[lower_idx] + fraction * (sorted_data[upper_idx] - sorted_data[lower_idx])
                results.append(value)

        return results


class DifficultyMetrics:
    """Calculate various difficulty metrics for rollouts."""

    @staticmethod
    def mean_reward(rollout: Dict[str, Any]) -> float:
        """Calculate mean reward score across all responses."""
        if 'reward' not in rollout or not rollout['reward']:
            return 0.0
        scores = [r.get('score', 0) for r in rollout['reward']]
        return statistics.mean(scores) if scores else 0.0

    @staticmethod
    def max_reward(rollout: Dict[str, Any]) -> float:
        """Calculate maximum reward score."""
        if 'reward' not in rollout or not rollout['reward']:
            return 0.0
        scores = [r.get('score', 0) for r in rollout['reward']]
        return max(scores) if scores else 0.0

    @staticmethod
    def min_reward(rollout: Dict[str, Any]) -> float:
        """Calculate minimum reward score."""
        if 'reward' not in rollout or not rollout['reward']:
            return 0.0
        scores = [r.get('score', 0) for r in rollout['reward']]
        return min(scores) if scores else 0.0

    @staticmethod
    def reward_variance(rollout: Dict[str, Any]) -> float:
        """Calculate variance in reward scores (higher = more difficult/inconsistent)."""
        if 'reward' not in rollout or not rollout['reward']:
            return 0.0
        scores = [r.get('score', 0) for r in rollout['reward']]
        return statistics.variance(scores) if len(scores) > 1 else 0.0

    @staticmethod
    def pass_at_k(rollout: Dict[str, Any], threshold: float = 0.5, k: int = None) -> float:
        """
        Calculate pass@k: proportion of top-k responses that pass the threshold.
        If k is None, use all responses.
        """
        if 'reward' not in rollout or not rollout['reward']:
            return 0.0

        scores = [r.get('score', 0) for r in rollout['reward']]
        if not scores:
            return 0.0

        # Sort scores in descending order
        sorted_scores = sorted(scores, reverse=True)

        # Take top k
        if k is not None:
            sorted_scores = sorted_scores[:k]

        # Calculate pass rate
        passed = sum(1 for s in sorted_scores if s >= threshold)
        return passed / len(sorted_scores)

    @staticmethod
    def success_rate(rollout: Dict[str, Any], threshold: float = 0.5) -> float:
        """Calculate success rate: proportion of responses above threshold."""
        if 'reward' not in rollout or not rollout['reward']:
            return 0.0
        scores = [r.get('score', 0) for r in rollout['reward']]
        if not scores:
            return 0.0
        passed = sum(1 for s in scores if s >= threshold)
        return passed / len(scores)

    @staticmethod
    def inverse_mean_reward(rollout: Dict[str, Any]) -> float:
        """
        Inverse of mean reward (higher score = more difficult).
        Useful for curriculum learning where you want harder tasks to have higher difficulty scores.
        """
        mean_r = DifficultyMetrics.mean_reward(rollout)
        # Add small epsilon to avoid division by zero
        return 1.0 / (mean_r + 1e-6)


class RolloutBucketer:
    """Bucket rollouts by difficulty into curriculum datasets."""

    DIFFICULTY_FUNCTIONS = {
        'mean_reward': DifficultyMetrics.mean_reward,
        'max_reward': DifficultyMetrics.max_reward,
        'min_reward': DifficultyMetrics.min_reward,
        'variance': DifficultyMetrics.reward_variance,
        'pass_at_k': lambda r: DifficultyMetrics.pass_at_k(r, threshold=0.5),
        'success_rate': lambda r: DifficultyMetrics.success_rate(r, threshold=0.5),
        'inverse_mean': DifficultyMetrics.inverse_mean_reward,
    }

    def __init__(self, input_dir: Path, output_dir: Path, metric: str = 'mean_reward'):
        """
        Initialize the bucketer.

        Args:
            input_dir: Directory containing input JSONL files
            output_dir: Directory to save bucketed JSONL files
            metric: Difficulty metric to use
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metric = metric

        if metric not in self.DIFFICULTY_FUNCTIONS:
            raise ValueError(f"Unknown metric: {metric}. Choose from {list(self.DIFFICULTY_FUNCTIONS.keys())}")

        self.difficulty_fn = self.DIFFICULTY_FUNCTIONS[metric]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_rollouts(self) -> List[Dict[str, Any]]:
        """Load all rollouts from JSONL files in input directory."""
        rollouts = []
        jsonl_files = list(self.input_dir.glob('*.jsonl'))

        print(f"Found {len(jsonl_files)} JSONL files in {self.input_dir}")

        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        rollouts.append(json.loads(line))

        print(f"Loaded {len(rollouts)} rollouts")
        return rollouts

    def calculate_difficulties(self, rollouts: List[Dict[str, Any]]) -> List[tuple]:
        """
        Calculate difficulty scores for all rollouts.

        Returns:
            List of (rollout, difficulty_score) tuples
        """
        rollouts_with_difficulty = []
        for rollout in rollouts:
            difficulty = self.difficulty_fn(rollout)
            rollouts_with_difficulty.append((rollout, difficulty))

        return rollouts_with_difficulty

    def bucket_by_percentiles(
        self,
        rollouts_with_difficulty: List[tuple],
        num_buckets: int = 3,
        bucket_names: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Bucket rollouts by percentiles.

        Args:
            rollouts_with_difficulty: List of (rollout, difficulty) tuples
            num_buckets: Number of difficulty buckets
            bucket_names: Optional custom names for buckets (default: easy, medium, hard, etc.)

        Returns:
            Dictionary mapping bucket names to lists of rollouts
        """
        if not rollouts_with_difficulty:
            return {}

        # Default bucket names
        if bucket_names is None:
            if num_buckets == 3:
                bucket_names = ['easy', 'medium', 'hard']
            elif num_buckets == 5:
                bucket_names = ['very_easy', 'easy', 'medium', 'hard', 'very_hard']
            else:
                bucket_names = [f'bucket_{i}' for i in range(num_buckets)]

        if len(bucket_names) != num_buckets:
            raise ValueError(f"Number of bucket names ({len(bucket_names)}) must match num_buckets ({num_buckets})")

        # Sort by difficulty
        sorted_rollouts = sorted(rollouts_with_difficulty, key=lambda x: x[1])

        # Calculate percentiles
        difficulties = [d for _, d in sorted_rollouts]
        percentile_points = [i * 100 / num_buckets for i in range(num_buckets + 1)]
        percentiles = calculate_percentiles(difficulties, percentile_points)

        print(f"\nDifficulty percentiles ({self.metric}):")
        for i, p in enumerate(percentiles):
            print(f"  {i * 100 / num_buckets:.0f}%: {p:.4f}")

        # Bucket rollouts
        buckets = defaultdict(list)
        for rollout, difficulty in sorted_rollouts:
            # Find which bucket this belongs to
            bucket_idx = num_buckets - 1  # Default to hardest bucket
            for i in range(num_buckets):
                if difficulty <= percentiles[i + 1]:
                    bucket_idx = i
                    break

            buckets[bucket_names[bucket_idx]].append(rollout)

        return dict(buckets)

    def bucket_by_thresholds(
        self,
        rollouts_with_difficulty: List[tuple],
        thresholds: List[float],
        bucket_names: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Bucket rollouts by absolute thresholds.

        Args:
            rollouts_with_difficulty: List of (rollout, difficulty) tuples
            thresholds: List of threshold values (e.g., [0.3, 0.7] creates 3 buckets)
            bucket_names: Optional custom names for buckets

        Returns:
            Dictionary mapping bucket names to lists of rollouts
        """
        num_buckets = len(thresholds) + 1

        # Default bucket names
        if bucket_names is None:
            bucket_names = [f'bucket_{i}' for i in range(num_buckets)]

        if len(bucket_names) != num_buckets:
            raise ValueError(f"Number of bucket names must be {num_buckets} (thresholds + 1)")

        # Sort thresholds
        thresholds = sorted(thresholds)

        # Bucket rollouts
        buckets = defaultdict(list)
        for rollout, difficulty in rollouts_with_difficulty:
            bucket_idx = num_buckets - 1  # Default to last bucket
            for i, threshold in enumerate(thresholds):
                if difficulty <= threshold:
                    bucket_idx = i
                    break

            buckets[bucket_names[bucket_idx]].append(rollout)

        return dict(buckets)

    def save_buckets(self, buckets: Dict[str, List[Dict[str, Any]]], prefix: str = ""):
        """
        Save bucketed rollouts to JSONL files.

        Args:
            buckets: Dictionary mapping bucket names to rollouts
            prefix: Optional prefix for output filenames
        """
        for bucket_name, rollouts in buckets.items():
            output_file = self.output_dir / f"{prefix}{bucket_name}.jsonl"

            with open(output_file, 'w') as f:
                for rollout in rollouts:
                    f.write(json.dumps(rollout) + '\n')

            print(f"Saved {len(rollouts)} rollouts to {output_file}")

    def run_percentile_bucketing(
        self,
        num_buckets: int = 3,
        bucket_names: List[str] = None,
        output_prefix: str = ""
    ):
        """Run the full bucketing pipeline using percentiles."""
        print(f"\n{'='*60}")
        print(f"Bucketing by {self.metric} using {num_buckets} percentile buckets")
        print(f"{'='*60}\n")

        # Load rollouts
        rollouts = self.load_rollouts()

        # Calculate difficulties
        rollouts_with_difficulty = self.calculate_difficulties(rollouts)

        # Bucket by percentiles
        buckets = self.bucket_by_percentiles(rollouts_with_difficulty, num_buckets, bucket_names)

        # Print statistics
        print(f"\nBucket statistics:")
        for bucket_name, bucket_rollouts in buckets.items():
            print(f"  {bucket_name}: {len(bucket_rollouts)} rollouts")

        # Save buckets
        self.save_buckets(buckets, prefix=output_prefix)

        print(f"\n{'='*60}")
        print(f"Bucketing complete! Output saved to {self.output_dir}")
        print(f"{'='*60}\n")

    def run_threshold_bucketing(
        self,
        thresholds: List[float],
        bucket_names: List[str] = None,
        output_prefix: str = ""
    ):
        """Run the full bucketing pipeline using absolute thresholds."""
        print(f"\n{'='*60}")
        print(f"Bucketing by {self.metric} using thresholds: {thresholds}")
        print(f"{'='*60}\n")

        # Load rollouts
        rollouts = self.load_rollouts()

        # Calculate difficulties
        rollouts_with_difficulty = self.calculate_difficulties(rollouts)

        # Bucket by thresholds
        buckets = self.bucket_by_thresholds(rollouts_with_difficulty, thresholds, bucket_names)

        # Print statistics
        print(f"\nBucket statistics:")
        for bucket_name, bucket_rollouts in buckets.items():
            print(f"  {bucket_name}: {len(bucket_rollouts)} rollouts")

        # Save buckets
        self.save_buckets(buckets, prefix=output_prefix)

        print(f"\n{'='*60}")
        print(f"Bucketing complete! Output saved to {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bucket JSONL rollouts by difficulty for curriculum learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bucket by mean reward into 3 percentile buckets (easy, medium, hard)
  python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward -n 3

  # Bucket by pass@k into 5 percentile buckets
  python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m pass_at_k -n 5

  # Bucket by absolute thresholds
  python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m mean_reward --thresholds 0.3 0.7

  # Use variance as difficulty metric (higher variance = harder)
  python bucket_rollouts_by_difficulty.py -i ./rollouts -o ./bucketed -m variance -n 3

Available metrics:
  - mean_reward: Average reward score (higher = easier)
  - max_reward: Best response score (higher = easier)
  - min_reward: Worst response score (higher = easier)
  - variance: Variance in scores (higher = more inconsistent/harder)
  - pass_at_k: Pass@k rate (higher = easier)
  - success_rate: Success rate above threshold (higher = easier)
  - inverse_mean: 1 / mean_reward (higher = harder)
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Input directory containing JSONL files'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Output directory for bucketed JSONL files'
    )

    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='mean_reward',
        choices=list(RolloutBucketer.DIFFICULTY_FUNCTIONS.keys()),
        help='Difficulty metric to use (default: mean_reward)'
    )

    parser.add_argument(
        '-n', '--num-buckets',
        type=int,
        default=3,
        help='Number of buckets for percentile bucketing (default: 3)'
    )

    parser.add_argument(
        '--bucket-names',
        type=str,
        nargs='+',
        help='Custom names for buckets (e.g., --bucket-names easy medium hard)'
    )

    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        help='Absolute thresholds for bucketing (e.g., --thresholds 0.3 0.7). If provided, ignores --num-buckets'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefix for output filenames (default: empty)'
    )

    args = parser.parse_args()

    # Create bucketer
    bucketer = RolloutBucketer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metric=args.metric
    )

    # Run bucketing
    if args.thresholds:
        bucketer.run_threshold_bucketing(
            thresholds=args.thresholds,
            bucket_names=args.bucket_names,
            output_prefix=args.prefix
        )
    else:
        bucketer.run_percentile_bucketing(
            num_buckets=args.num_buckets,
            bucket_names=args.bucket_names,
            output_prefix=args.prefix
        )


if __name__ == '__main__':
    main()
