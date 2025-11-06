#!/usr/bin/env python3
"""
Difficulty Filtering and Bucketing Script

This script loads a model, calculates rewards on training problems using verl's
existing reward infrastructure, computes mean@k metrics, and buckets problems
based on difficulty using various strategies.

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
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

# Import verl components
from verl import DataProto
from verl.trainer.ppo.reward import load_reward_manager, compute_reward
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics


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
    """Filter and bucket training problems by difficulty using verl infrastructure."""

    def __init__(
        self,
        config: DictConfig,
        output_dir: str,
        num_samples: int = 5,
    ):
        """
        Initialize the difficulty filter.

        Args:
            config: Hydra/OmegaConf configuration for model, data, and generation
            output_dir: Directory to save results
            num_samples: Number of samples to generate per problem (for mean@k)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples

        # Load tokenizer
        model_path = copy_to_local(config.model.path, use_shm=config.model.get("use_shm", False))
        trust_remote_code = config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(model_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load reward manager using verl's infrastructure
        self.reward_fn = load_reward_manager(
            config, self.tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # Initialize Ray for parallel generation if using actor workers
        if not ray.is_initialized():
            ray.init(**OmegaConf.to_container(config.get("ray_kwargs", {}).get("ray_init", {})))

        # Load dataset using verl's RLHFDataset
        print(f"Loading data from {config.data.path}...")
        self.dataset = RLHFDataset(
            data_files=config.data.path,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=config.data,
        )
        print(f"Loaded {len(self.dataset)} problems")

        # Create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # For simplicity in offline mode
        )

        # Initialize actor/rollout for generation if config provided
        self.actor_worker = None
        if hasattr(config, 'actor_rollout_ref'):
            self._init_actor_worker()

    def _init_actor_worker(self):
        """Initialize actor worker for generation (optional, can use HF generate instead)."""
        # For simplicity, we'll use HuggingFace generation in this script
        # If you want to use verl's rollout workers, implement similar to ray_trainer.py
        from transformers import AutoModelForCausalLM

        model_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        print(f"Loading model from {model_path}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def generate_responses(self, data_batch: DataProto, num_samples: int = 1) -> List[DataProto]:
        """
        Generate multiple responses for a batch using verl-style generation.

        Args:
            data_batch: DataProto containing input prompts
            num_samples: Number of responses to generate per prompt

        Returns:
            List of DataProto objects with generated responses
        """
        # Repeat the batch for multiple samples
        repeated_batch = data_batch.repeat(repeat_times=num_samples, interleave=True)

        input_ids = repeated_batch.batch["input_ids"]
        attention_mask = repeated_batch.batch["attention_mask"]

        # Move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Generate responses
        gen_config = self.config.generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_config.get("max_new_tokens", 512),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9),
                do_sample=gen_config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        # Extract responses (remove prompt tokens)
        prompt_length = input_ids.shape[1]
        responses = outputs[:, prompt_length:]

        # Create output DataProto
        repeated_batch.batch["responses"] = responses
        repeated_batch.batch["response_mask"] = (responses != self.tokenizer.pad_token_id).long()

        return repeated_batch

    def calculate_metrics_for_batch(self, data_batch: Dict) -> List[DifficultyMetrics]:
        """Calculate difficulty metrics for a batch of problems."""
        # Convert to DataProto
        batch_proto = DataProto.from_single_dict(data_batch)

        # Add UIDs if not present
        if "uid" not in batch_proto.non_tensor_batch:
            batch_proto.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_proto.batch))], dtype=object
            )

        # Generate responses
        print(f"  Generating {self.num_samples} responses for batch...")
        gen_batch = self.generate_responses(batch_proto, self.num_samples)

        # Compute rewards using verl's reward manager
        print("  Computing rewards...")
        gen_batch.meta_info["validate"] = True
        result = self.reward_fn(gen_batch, return_dict=True)
        reward_tensor = result["reward_tensor"]

        # reward_tensor shape: (batch_size * num_samples, seq_len)
        # Sum across sequence to get per-sample rewards
        rewards = reward_tensor.sum(-1).cpu().numpy()

        # Reshape to (original_batch_size, num_samples)
        batch_size = len(batch_proto.batch)
        rewards_per_problem = rewards.reshape(batch_size, self.num_samples)

        # Calculate metrics for each problem
        metrics_list = []
        for i in range(batch_size):
            problem_rewards = rewards_per_problem[i]

            # Get problem info
            problem_id = batch_proto.non_tensor_batch["uid"][i]
            data_source = batch_proto.non_tensor_batch.get("data_source", np.array(["unknown"]))[i]

            # Calculate statistics
            mean_reward = float(np.mean(problem_rewards))
            std_reward = float(np.std(problem_rewards))
            max_reward = float(np.max(problem_rewards))
            min_reward = float(np.min(problem_rewards))
            pass_rate = float(np.mean(problem_rewards > 0.0)) * 100

            # Calculate mean@k for different k values
            mean_at_k = {}
            for k in [1, 3, 5, 10]:
                if k <= len(problem_rewards):
                    top_k_rewards = np.sort(problem_rewards)[-k:]
                    mean_at_k[k] = float(np.mean(top_k_rewards))

            # Calculate difficulty score (lower mean reward = harder problem)
            difficulty_score = 1.0 - mean_reward + (std_reward * 0.3)

            metrics = DifficultyMetrics(
                problem_id=problem_id,
                data_source=data_source,
                mean_reward=mean_reward,
                std_reward=std_reward,
                max_reward=max_reward,
                min_reward=min_reward,
                pass_rate=pass_rate,
                mean_at_k=mean_at_k,
                raw_rewards=problem_rewards.tolist(),
                difficulty_score=difficulty_score
            )
            metrics_list.append(metrics)

        return metrics_list

    def calculate_all_metrics(self) -> List[DifficultyMetrics]:
        """Calculate metrics for all problems in the dataset."""
        all_metrics = []

        for batch in tqdm(self.dataloader, desc="Processing batches"):
            try:
                batch_metrics = self.calculate_metrics_for_batch(batch)
                all_metrics.extend(batch_metrics)
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_metrics

    def dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, step=0):
        """
        Dump generations to JSONL file (reuses verl's _dump_generations pattern).

        Args:
            inputs: List of input prompts
            outputs: List of generated outputs
            gts: List of ground truths
            scores: List of scores
            reward_extra_infos_dict: Dictionary of extra reward info
            step: Global step (default 0 for offline evaluation)
        """
        filename = self.output_dir / f"generations_{step}.jsonl"

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [step] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def bucket_by_percentile(
        self,
        metrics_list: List[DifficultyMetrics],
        percentiles: List[int] = [25, 50, 75]
    ) -> List[DifficultyMetrics]:
        """Bucket problems by percentile of difficulty score."""
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
        """Bucket problems by pass rate."""
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
        """Bucket problems by mean reward thresholds."""
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
        """Bucket problems using K-means clustering on multiple features."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("scikit-learn not installed, falling back to percentile bucketing")
            return self.bucket_by_percentile(metrics_list)

        # Extract features
        features = []
        for m in metrics_list:
            feature_vec = [
                m.mean_reward,
                m.std_reward,
                m.pass_rate / 100.0,
                m.difficulty_score
            ]
            features.append(feature_vec)

        # Standardize and cluster
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Order clusters by difficulty
        cluster_difficulties = {}
        for label in range(n_clusters):
            mask = cluster_labels == label
            cluster_scores = [m.difficulty_score for i, m in enumerate(metrics_list) if mask[i]]
            cluster_difficulties[label] = np.mean(cluster_scores)

        sorted_clusters = sorted(cluster_difficulties.items(), key=lambda x: x[1])
        cluster_to_bucket = {}
        bucket_names = ["easy", "medium", "hard", "very_hard"]
        for i, (label, _) in enumerate(sorted_clusters):
            if n_clusters == 4:
                cluster_to_bucket[label] = bucket_names[i]
            else:
                bucket_idx = int(i * len(bucket_names) / n_clusters)
                cluster_to_bucket[label] = bucket_names[min(bucket_idx, len(bucket_names) - 1)]

        for i, metrics in enumerate(metrics_list):
            metrics.difficulty_bucket = cluster_to_bucket[cluster_labels[i]]

        return metrics_list

    def bucket_by_strategy(
        self,
        metrics_list: List[DifficultyMetrics],
        strategy: str = "percentile"
    ) -> List[DifficultyMetrics]:
        """Bucket problems using the specified strategy."""
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

    def save_results(self, metrics_list: List[DifficultyMetrics], strategy: str):
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

        # Save as parquet
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

        print("\n" + "="*80)
        print(f"Results saved to {self.output_dir}")
        print("="*80)
        print("\nBucket Distribution:")
        for bucket_name, stats in summary['bucket_statistics'].items():
            print(f"  {bucket_name:12s}: {stats['count']:4d} problems "
                  f"(mean_reward={stats['mean_reward']:.3f}, "
                  f"pass_rate={stats['mean_pass_rate']:.1f}%)")


@hydra.main(config_path="config", config_name="difficulty_filter", version_base=None)
def main(config: DictConfig):
    """Main entry point using Hydra configuration."""
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    # Initialize filter
    difficulty_filter = DifficultyFilter(
        config=config,
        output_dir=config.output_dir,
        num_samples=config.num_samples,
    )

    # Calculate metrics
    print("\n" + "="*80)
    print("CALCULATING DIFFICULTY METRICS")
    print("="*80)
    metrics_list = difficulty_filter.calculate_all_metrics()

    # Bucket by difficulty
    print("\n" + "="*80)
    print(f"BUCKETING BY STRATEGY: {config.bucketing_strategy}")
    print("="*80)
    metrics_list = difficulty_filter.bucket_by_strategy(
        metrics_list,
        strategy=config.bucketing_strategy
    )

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    difficulty_filter.save_results(metrics_list, config.bucketing_strategy)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
