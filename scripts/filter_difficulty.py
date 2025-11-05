#!/usr/bin/env python3
"""
Difficulty Filtering and Bucketing Script

This script loads a model, calculates IFeval-style rewards on training problems,
computes mean@k metrics, and buckets problems based on difficulty using various strategies.

Usage:
    python scripts/filter_difficulty.py \
        --model_path <path_to_model> \
        --data_path <path_to_parquet> \
        --output_dir <output_directory> \
        --num_samples 5 \
        --bucketing_strategy percentile
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add verl to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verl.utils.reward_score import default_compute_score


@dataclass
class DifficultyMetrics:
    """Metrics for problem difficulty."""
    problem_id: str
    data_source: str
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    pass_rate: float  # Percentage of samples that passed
    mean_at_k: Dict[int, float]  # mean@1, mean@3, mean@5, etc.
    raw_rewards: List[float]
    difficulty_score: float  # Combined difficulty metric
    difficulty_bucket: Optional[str] = None


class DifficultyFilter:
    """Filter and bucket training problems by difficulty."""

    def __init__(
        self,
        model_path: str,
        data_path: str,
        output_dir: str,
        num_samples: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the difficulty filter.

        Args:
            model_path: Path to the model
            data_path: Path to the dataset (parquet format)
            output_dir: Directory to save results
            num_samples: Number of samples to generate per problem (for mean@k)
            device: Device to run on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

        print(f"Loading data from {data_path}...")
        self.data = pd.read_parquet(data_path)
        print(f"Loaded {len(self.data)} problems")

    def generate_response(self, prompt: str, num_samples: int = 1) -> List[str]:
        """Generate multiple responses for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        responses = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response)

        return responses

    def compute_reward(
        self,
        data_source: str,
        response: str,
        ground_truth: Any,
        extra_info: Optional[Dict] = None
    ) -> float:
        """Compute reward for a response."""
        try:
            result = default_compute_score(
                data_source=data_source,
                solution=response,
                ground_truth=ground_truth,
                extra_info=extra_info or {}
            )
            # The reward function returns a dict with 'score' key
            if isinstance(result, dict):
                return float(result.get('score', 0.0))
            return float(result)
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0

    def calculate_metrics_for_problem(
        self,
        problem_id: str,
        problem_data: Dict
    ) -> DifficultyMetrics:
        """Calculate difficulty metrics for a single problem."""
        # Extract problem information
        data_source = problem_data.get('data_source', '')

        # Handle different prompt formats
        if 'prompt' in problem_data:
            prompt_data = problem_data['prompt']
            if isinstance(prompt_data, list):
                # Chat format
                prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                                   for msg in prompt_data])
            else:
                prompt = str(prompt_data)
        elif 'input_ids' in problem_data:
            # Already tokenized
            prompt = self.tokenizer.decode(problem_data['input_ids'])
        else:
            raise ValueError(f"Problem {problem_id} has no prompt field")

        # Get ground truth
        reward_model_info = problem_data.get('reward_model', {})
        ground_truth = reward_model_info.get('ground_truth', '')
        extra_info = problem_data.get('extra_info', {})

        # Generate responses
        print(f"  Generating {self.num_samples} responses for problem {problem_id}...")
        responses = self.generate_response(prompt, self.num_samples)

        # Calculate rewards for each response
        rewards = []
        for response in responses:
            reward = self.compute_reward(data_source, response, ground_truth, extra_info)
            rewards.append(reward)

        # Calculate statistics
        rewards_array = np.array(rewards)
        mean_reward = float(np.mean(rewards_array))
        std_reward = float(np.std(rewards_array))
        max_reward = float(np.max(rewards_array))
        min_reward = float(np.min(rewards_array))
        pass_rate = float(np.mean(rewards_array > 0.0)) * 100  # Assuming reward > 0 means pass

        # Calculate mean@k for different k values
        mean_at_k = {}
        for k in [1, 3, 5, 10]:
            if k <= len(rewards):
                # mean@k: average of best k rewards
                top_k_rewards = np.sort(rewards_array)[-k:]
                mean_at_k[k] = float(np.mean(top_k_rewards))

        # Calculate difficulty score (lower mean reward = harder problem)
        # Also consider variance (high variance = inconsistent difficulty)
        difficulty_score = 1.0 - mean_reward + (std_reward * 0.3)

        return DifficultyMetrics(
            problem_id=problem_id,
            data_source=data_source,
            mean_reward=mean_reward,
            std_reward=std_reward,
            max_reward=max_reward,
            min_reward=min_reward,
            pass_rate=pass_rate,
            mean_at_k=mean_at_k,
            raw_rewards=rewards,
            difficulty_score=difficulty_score
        )

    def calculate_all_metrics(self) -> List[DifficultyMetrics]:
        """Calculate metrics for all problems in the dataset."""
        metrics_list = []

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Processing problems"):
            problem_id = str(idx)
            if 'id' in row:
                problem_id = str(row['id'])
            elif 'index' in row:
                problem_id = str(row['index'])

            try:
                metrics = self.calculate_metrics_for_problem(problem_id, row.to_dict())
                metrics_list.append(metrics)
            except Exception as e:
                print(f"Error processing problem {problem_id}: {e}")
                continue

        return metrics_list

    def bucket_by_percentile(
        self,
        metrics_list: List[DifficultyMetrics],
        percentiles: List[int] = [25, 50, 75]
    ) -> List[DifficultyMetrics]:
        """
        Bucket problems by percentile of difficulty score.

        Args:
            metrics_list: List of difficulty metrics
            percentiles: Percentile cutoffs (e.g., [25, 50, 75] for quartiles)

        Returns:
            Updated metrics list with difficulty_bucket assigned
        """
        difficulty_scores = [m.difficulty_score for m in metrics_list]
        thresholds = np.percentile(difficulty_scores, percentiles)

        for metrics in metrics_list:
            score = metrics.difficulty_score
            if score <= thresholds[0]:
                metrics.difficulty_bucket = "easy"
            elif score <= thresholds[1]:
                metrics.difficulty_bucket = "medium"
            elif score <= thresholds[2]:
                metrics.difficulty_bucket = "hard"
            else:
                metrics.difficulty_bucket = "very_hard"

        return metrics_list

    def bucket_by_pass_rate(
        self,
        metrics_list: List[DifficultyMetrics],
        thresholds: List[float] = [20.0, 50.0, 80.0]
    ) -> List[DifficultyMetrics]:
        """
        Bucket problems by pass rate.

        Args:
            metrics_list: List of difficulty metrics
            thresholds: Pass rate thresholds in percentage

        Returns:
            Updated metrics list with difficulty_bucket assigned
        """
        for metrics in metrics_list:
            pass_rate = metrics.pass_rate
            if pass_rate >= thresholds[2]:
                metrics.difficulty_bucket = "easy"
            elif pass_rate >= thresholds[1]:
                metrics.difficulty_bucket = "medium"
            elif pass_rate >= thresholds[0]:
                metrics.difficulty_bucket = "hard"
            else:
                metrics.difficulty_bucket = "very_hard"

        return metrics_list

    def bucket_by_mean_reward(
        self,
        metrics_list: List[DifficultyMetrics],
        thresholds: List[float] = [0.25, 0.50, 0.75]
    ) -> List[DifficultyMetrics]:
        """
        Bucket problems by mean reward thresholds.

        Args:
            metrics_list: List of difficulty metrics
            thresholds: Mean reward thresholds

        Returns:
            Updated metrics list with difficulty_bucket assigned
        """
        for metrics in metrics_list:
            mean_reward = metrics.mean_reward
            if mean_reward >= thresholds[2]:
                metrics.difficulty_bucket = "easy"
            elif mean_reward >= thresholds[1]:
                metrics.difficulty_bucket = "medium"
            elif mean_reward >= thresholds[0]:
                metrics.difficulty_bucket = "hard"
            else:
                metrics.difficulty_bucket = "very_hard"

        return metrics_list

    def bucket_by_adaptive_clustering(
        self,
        metrics_list: List[DifficultyMetrics],
        n_clusters: int = 4
    ) -> List[DifficultyMetrics]:
        """
        Bucket problems using K-means clustering on multiple features.

        Args:
            metrics_list: List of difficulty metrics
            n_clusters: Number of clusters

        Returns:
            Updated metrics list with difficulty_bucket assigned
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("scikit-learn not installed, falling back to percentile bucketing")
            return self.bucket_by_percentile(metrics_list)

        # Extract features for clustering
        features = []
        for m in metrics_list:
            feature_vec = [
                m.mean_reward,
                m.std_reward,
                m.pass_rate / 100.0,  # Normalize to [0, 1]
                m.difficulty_score
            ]
            features.append(feature_vec)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Calculate mean difficulty score per cluster to order them
        cluster_difficulties = {}
        for label in range(n_clusters):
            mask = cluster_labels == label
            cluster_scores = [m.difficulty_score for i, m in enumerate(metrics_list) if mask[i]]
            cluster_difficulties[label] = np.mean(cluster_scores)

        # Sort clusters by difficulty
        sorted_clusters = sorted(cluster_difficulties.items(), key=lambda x: x[1])
        cluster_to_bucket = {}
        bucket_names = ["easy", "medium", "hard", "very_hard"]
        for i, (label, _) in enumerate(sorted_clusters):
            if n_clusters == 4:
                cluster_to_bucket[label] = bucket_names[i]
            else:
                # Map to closest bucket name
                bucket_idx = int(i * len(bucket_names) / n_clusters)
                cluster_to_bucket[label] = bucket_names[min(bucket_idx, len(bucket_names) - 1)]

        # Assign buckets
        for i, metrics in enumerate(metrics_list):
            metrics.difficulty_bucket = cluster_to_bucket[cluster_labels[i]]

        return metrics_list

    def bucket_by_strategy(
        self,
        metrics_list: List[DifficultyMetrics],
        strategy: str = "percentile"
    ) -> List[DifficultyMetrics]:
        """
        Bucket problems using the specified strategy.

        Args:
            metrics_list: List of difficulty metrics
            strategy: One of 'percentile', 'pass_rate', 'mean_reward', 'adaptive'

        Returns:
            Updated metrics list with difficulty_bucket assigned
        """
        if strategy == "percentile":
            return self.bucket_by_percentile(metrics_list)
        elif strategy == "pass_rate":
            return self.bucket_by_pass_rate(metrics_list)
        elif strategy == "mean_reward":
            return self.bucket_by_mean_reward(metrics_list)
        elif strategy == "adaptive":
            return self.bucket_by_adaptive_clustering(metrics_list)
        else:
            raise ValueError(f"Unknown bucketing strategy: {strategy}")

    def save_results(
        self,
        metrics_list: List[DifficultyMetrics],
        strategy: str
    ):
        """Save results to files."""
        # Save detailed metrics
        metrics_dicts = []
        for m in metrics_list:
            metrics_dicts.append({
                'problem_id': m.problem_id,
                'data_source': m.data_source,
                'mean_reward': m.mean_reward,
                'std_reward': m.std_reward,
                'max_reward': m.max_reward,
                'min_reward': m.min_reward,
                'pass_rate': m.pass_rate,
                'mean_at_k': m.mean_at_k,
                'difficulty_score': m.difficulty_score,
                'difficulty_bucket': m.difficulty_bucket,
                'raw_rewards': m.raw_rewards,
            })

        # Save as JSON
        output_file = self.output_dir / f"difficulty_metrics_{strategy}.json"
        with open(output_file, 'w') as f:
            json.dump(metrics_dicts, f, indent=2)
        print(f"Saved detailed metrics to {output_file}")

        # Save as parquet for easy filtering
        df = pd.DataFrame(metrics_dicts)
        parquet_file = self.output_dir / f"difficulty_metrics_{strategy}.parquet"
        df.to_parquet(parquet_file)
        print(f"Saved metrics to {parquet_file}")

        # Save summary statistics
        summary = {
            'total_problems': len(metrics_list),
            'bucketing_strategy': strategy,
            'bucket_distribution': {},
            'bucket_statistics': {}
        }

        # Calculate distribution and statistics per bucket
        buckets = defaultdict(list)
        for m in metrics_list:
            buckets[m.difficulty_bucket].append(m)

        for bucket_name, bucket_metrics in buckets.items():
            summary['bucket_distribution'][bucket_name] = len(bucket_metrics)
            summary['bucket_statistics'][bucket_name] = {
                'count': len(bucket_metrics),
                'mean_reward': float(np.mean([m.mean_reward for m in bucket_metrics])),
                'mean_pass_rate': float(np.mean([m.pass_rate for m in bucket_metrics])),
                'mean_difficulty_score': float(np.mean([m.difficulty_score for m in bucket_metrics])),
            }

        summary_file = self.output_dir / f"summary_{strategy}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")

        # Save filtered datasets per bucket
        for bucket_name, bucket_metrics in buckets.items():
            problem_ids = [m.problem_id for m in bucket_metrics]
            # Filter original data
            bucket_data = self.data[self.data.index.isin([int(pid) if pid.isdigit() else pid
                                                          for pid in problem_ids])]
            bucket_file = self.output_dir / f"data_{strategy}_{bucket_name}.parquet"
            bucket_data.to_parquet(bucket_file)
            print(f"Saved {len(bucket_data)} problems to {bucket_file}")

        print("\n" + "="*80)
        print(f"Results saved to {self.output_dir}")
        print("="*80)
        print("\nBucket Distribution:")
        for bucket_name, stats in summary['bucket_statistics'].items():
            print(f"  {bucket_name:12s}: {stats['count']:4d} problems "
                  f"(mean_reward={stats['mean_reward']:.3f}, "
                  f"pass_rate={stats['mean_pass_rate']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Filter and bucket training problems by difficulty"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model for evaluation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset (parquet format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./difficulty_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate per problem for mean@k"
    )
    parser.add_argument(
        "--bucketing_strategy",
        type=str,
        default="percentile",
        choices=["percentile", "pass_rate", "mean_reward", "adaptive"],
        help="Strategy for bucketing problems by difficulty"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )

    args = parser.parse_args()

    # Initialize filter
    difficulty_filter = DifficultyFilter(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Calculate metrics
    print("\n" + "="*80)
    print("CALCULATING DIFFICULTY METRICS")
    print("="*80)
    metrics_list = difficulty_filter.calculate_all_metrics()

    # Bucket by difficulty
    print("\n" + "="*80)
    print(f"BUCKETING BY STRATEGY: {args.bucketing_strategy}")
    print("="*80)
    metrics_list = difficulty_filter.bucket_by_strategy(
        metrics_list,
        strategy=args.bucketing_strategy
    )

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    difficulty_filter.save_results(metrics_list, args.bucketing_strategy)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
