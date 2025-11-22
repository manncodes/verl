#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess bucketed rollout JSONL files for curriculum learning.

This script converts bucketed rollouts (easy.jsonl, medium.jsonl, hard.jsonl)
from the bucket_rollouts_by_difficulty.py script into the proper training format
for verl training pipelines.

Features:
- Converts rollout format to training data format
- Preserves ground truth and constraint information
- Adds curriculum stage metadata
- Supports progressive curriculum training
- Creates train/val splits for each difficulty level
- Optional best-response filtering

Usage:
    # Basic: Process all buckets
    python preprocess_bucketed_rollouts.py \
        -i ./bucketed \
        -o ./preprocessed_curriculum

    # Process specific stages
    python preprocess_bucketed_rollouts.py \
        -i ./curriculum \
        -o ./preprocessed \
        --buckets stage1 stage2 stage3

    # Combine multiple stages into one dataset
    python preprocess_bucketed_rollouts.py \
        -i ./bucketed \
        -o ./preprocessed \
        --buckets easy medium \
        --combine \
        --output-name easy_medium_combined
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not found. Install with: pip install datasets")


DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> l </answer>. Question: "
)


class RolloutPreprocessor:
    """Preprocess bucketed rollouts for curriculum learning."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        filter_mode: str = "all",
        add_system_prompt: bool = True,
        train_val_split: float = 0.95,
    ):
        """
        Initialize the preprocessor.

        Args:
            input_dir: Directory containing bucketed JSONL files
            output_dir: Directory to save preprocessed parquet files
            filter_mode: How to filter rollouts ("all", "best_only", "passing_only")
            add_system_prompt: Whether to add system/instruction prompts
            train_val_split: Ratio for train/validation split
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.filter_mode = filter_mode
        self.add_system_prompt = add_system_prompt
        self.train_val_split = train_val_split

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_rollouts_from_jsonl(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Load rollouts from a JSONL file."""
        rollouts = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    rollouts.append(json.loads(line))
        return rollouts

    def extract_ground_truth(self, rollout: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract ground truth information from rollout.

        The rollout may have ground_truth in different formats:
        1. In reward_model.ground_truth
        2. In extra_info
        3. As a JSON string in ground_truth field
        """
        # Try reward_model first (from preprocessed data)
        if "reward_model" in rollout and "ground_truth" in rollout["reward_model"]:
            gt = rollout["reward_model"]["ground_truth"]
            if isinstance(gt, dict):
                return gt

        # Try extra_info
        if "extra_info" in rollout:
            extra = rollout["extra_info"]
            if "instruction_id_list" in extra and "kwargs" in extra:
                return {
                    "instruction_id_list": extra["instruction_id_list"],
                    "kwargs": extra["kwargs"],
                }

        # Try parsing ground_truth as JSON string
        if "ground_truth" in rollout:
            gt_str = rollout["ground_truth"]
            if isinstance(gt_str, str):
                try:
                    gt_dict = json.loads(gt_str)
                    if isinstance(gt_dict, list) and len(gt_dict) > 0:
                        return {
                            "instruction_id_list": gt_dict[0].get("instruction_id", []),
                            "kwargs": gt_dict[0].get("kwargs", []),
                        }
                except (json.JSONDecodeError, KeyError):
                    pass

        # Default empty ground truth
        return {"instruction_id_list": [], "kwargs": []}

    def extract_prompt(self, rollout: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract prompt messages from rollout.

        Returns messages in the format: [{"role": "user", "content": "..."}]
        """
        # If rollout already has prompt in message format
        if "prompt" in rollout and isinstance(rollout["prompt"], list):
            messages = rollout["prompt"]
            if messages and isinstance(messages[0], dict) and "role" in messages[0]:
                return messages

        # If rollout has messages field
        if "messages" in rollout and isinstance(rollout["messages"], list):
            # Filter out assistant messages (we only want prompts)
            return [msg for msg in rollout["messages"] if msg.get("role") != "assistant"]

        # Fallback: try to extract from extra_info or construct from constraint
        if "extra_info" in rollout and "prompt" in rollout["extra_info"]:
            content = rollout["extra_info"]["prompt"]
            return [{"role": "user", "content": content}]

        # Last resort: use constraint or empty
        if "constraint" in rollout:
            return [{"role": "user", "content": rollout["constraint"]}]

        return [{"role": "user", "content": ""}]

    def get_best_response_info(self, rollout: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get information about the best response from the rollout.

        Returns:
            Dict with best response info, or None if no responses
        """
        if "reward" not in rollout or not rollout["reward"]:
            return None

        rewards = rollout["reward"]
        if not rewards:
            return None

        # Find best response
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i].get("score", 0))
        best_reward = rewards[best_idx]

        # Get corresponding response if available
        response_content = None
        if "response" in rollout and "choices" in rollout["response"]:
            choices = rollout["response"]["choices"]
            if best_idx < len(choices):
                response_content = choices[best_idx].get("message", {}).get("content", "")

        return {
            "index": best_idx,
            "score": best_reward.get("score", 0),
            "response": response_content,
            "reward_info": best_reward,
        }

    def should_include_rollout(self, rollout: Dict[str, Any]) -> bool:
        """
        Determine if rollout should be included based on filter_mode.

        Args:
            rollout: Rollout data

        Returns:
            True if rollout should be included
        """
        if self.filter_mode == "all":
            return True

        if "reward" not in rollout or not rollout["reward"]:
            return False

        rewards = rollout["reward"]
        scores = [r.get("score", 0) for r in rewards]

        if self.filter_mode == "best_only":
            # Only include if best score is positive
            return max(scores) > 0

        elif self.filter_mode == "passing_only":
            # Only include if at least one response passes (score >= 0.5)
            return any(s >= 0.5 for s in scores)

        return True

    def process_rollout(
        self,
        rollout: Dict[str, Any],
        idx: int,
        stage: str,
        data_source: str = "bucketed_curriculum"
    ) -> Dict[str, Any]:
        """
        Convert a rollout into the training data format.

        Args:
            rollout: Original rollout data
            idx: Index for this example
            stage: Curriculum stage name (e.g., "easy", "medium", "hard")
            data_source: Name of the data source

        Returns:
            Processed data in verl training format
        """
        # Extract components
        ground_truth = self.extract_ground_truth(rollout)
        prompt_messages = self.extract_prompt(rollout)

        # Add system prompt if requested
        if self.add_system_prompt and prompt_messages:
            # Check if system message already exists
            has_system = any(msg.get("role") == "system" for msg in prompt_messages)
            if not has_system:
                # Add instruction prompt to first user message
                for msg in prompt_messages:
                    if msg.get("role") == "user":
                        instruction_prompt = DEFAULT_SYSTEM_CONTENT + '\n' + DEFAULT_USER_CONTENT_PREFIX
                        msg["content"] = instruction_prompt + "\n\n" + msg["content"]
                        break

        # Extract constraint information
        constraint_type = rollout.get("constraint_type", rollout.get("extra_info", {}).get("constraint_type", ""))
        constraint = rollout.get("constraint", rollout.get("extra_info", {}).get("constraint", ""))

        # Get best response info (useful for analysis)
        best_response_info = self.get_best_response_info(rollout)

        # Calculate difficulty metrics
        difficulty_metrics = {}
        if "reward" in rollout and rollout["reward"]:
            scores = [r.get("score", 0) for r in rollout["reward"]]
            difficulty_metrics = {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "num_responses": len(scores),
            }

        # Create data in verl format
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": rollout.get("ability", "instruction_following"),
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": "train",  # Will be updated during train/val split
                "index": idx,
                "key": f"{stage}_{idx}",
                "curriculum_stage": stage,
                "instruction_id_list": ground_truth.get("instruction_id_list", []),
                "kwargs": ground_truth.get("kwargs", []),
                "constraint_type": constraint_type,
                "constraint": constraint,
                "original_key": rollout.get("key", rollout.get("extra_info", {}).get("key", "")),
                "difficulty_metrics": difficulty_metrics,
            },
        }

        # Add best response info if available
        if best_response_info:
            data["extra_info"]["best_response"] = best_response_info

        return data

    def process_bucket(
        self,
        bucket_name: str,
        data_source: str = "bucketed_curriculum"
    ) -> Optional[datasets.Dataset]:
        """
        Process a single bucket file.

        Args:
            bucket_name: Name of the bucket (e.g., "easy", "medium")
            data_source: Name of the data source

        Returns:
            Processed dataset or None if file doesn't exist
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")

        jsonl_path = self.input_dir / f"{bucket_name}.jsonl"
        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} not found, skipping")
            return None

        print(f"\nProcessing {bucket_name}...")
        rollouts = self.load_rollouts_from_jsonl(jsonl_path)
        print(f"  Loaded {len(rollouts)} rollouts")

        # Filter rollouts
        filtered_rollouts = [r for r in rollouts if self.should_include_rollout(r)]
        print(f"  After filtering ({self.filter_mode}): {len(filtered_rollouts)} rollouts")

        if not filtered_rollouts:
            print(f"  Warning: No rollouts remaining after filtering")
            return None

        # Process each rollout
        processed_data = []
        for idx, rollout in enumerate(filtered_rollouts):
            processed = self.process_rollout(rollout, idx, bucket_name, data_source)
            processed_data.append(processed)

        # Convert to dataset
        dataset = datasets.Dataset.from_list(processed_data)
        return dataset

    def save_dataset(
        self,
        dataset: datasets.Dataset,
        name: str,
        create_val_split: bool = True
    ):
        """
        Save dataset to parquet files.

        Args:
            dataset: Dataset to save
            name: Name for the output files
            create_val_split: Whether to create train/val split
        """
        if create_val_split:
            # Split into train and validation
            split_dataset = dataset.train_test_split(
                test_size=1 - self.train_val_split,
                seed=42
            )
            train_dataset = split_dataset["train"]
            val_dataset = split_dataset["test"]

            # Update split field
            train_dataset = train_dataset.map(
                lambda x: {"extra_info": {**x["extra_info"], "split": "train"}}
            )
            val_dataset = val_dataset.map(
                lambda x: {"extra_info": {**x["extra_info"], "split": "val"}}
            )

            # Save
            train_path = self.output_dir / f"{name}_train.parquet"
            val_path = self.output_dir / f"{name}_val.parquet"

            train_dataset.to_parquet(train_path)
            val_dataset.to_parquet(val_path)

            print(f"  Saved {len(train_dataset)} train examples to {train_path}")
            print(f"  Saved {len(val_dataset)} val examples to {val_path}")

            return train_dataset, val_dataset
        else:
            # Save without split
            output_path = self.output_dir / f"{name}.parquet"
            dataset.to_parquet(output_path)
            print(f"  Saved {len(dataset)} examples to {output_path}")
            return dataset

    def process_buckets(
        self,
        bucket_names: Optional[List[str]] = None,
        combine: bool = False,
        combined_name: str = "combined",
        data_source: str = "bucketed_curriculum"
    ):
        """
        Process multiple buckets.

        Args:
            bucket_names: List of bucket names to process (None = auto-detect)
            combine: Whether to combine all buckets into one dataset
            combined_name: Name for combined dataset
            data_source: Name of the data source
        """
        # Auto-detect bucket files if not specified
        if bucket_names is None:
            jsonl_files = list(self.input_dir.glob("*.jsonl"))
            bucket_names = [f.stem for f in jsonl_files]
            print(f"Auto-detected buckets: {bucket_names}")

        if not bucket_names:
            print("No buckets found!")
            return

        # Process each bucket
        datasets_dict = {}
        for bucket_name in bucket_names:
            dataset = self.process_bucket(bucket_name, data_source)
            if dataset is not None:
                datasets_dict[bucket_name] = dataset

        if not datasets_dict:
            print("No datasets were created!")
            return

        # Save datasets
        if combine:
            # Combine all datasets
            combined_dataset = datasets.concatenate_datasets(list(datasets_dict.values()))
            print(f"\nCombined dataset: {len(combined_dataset)} examples")
            self.save_dataset(combined_dataset, combined_name)
        else:
            # Save each bucket separately
            for bucket_name, dataset in datasets_dict.items():
                self.save_dataset(dataset, bucket_name)

        # Print summary
        print("\n" + "="*60)
        print("Processing Complete!")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Buckets processed: {list(datasets_dict.keys())}")
        print(f"Total examples: {sum(len(d) for d in datasets_dict.values())}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess bucketed rollouts for curriculum learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all buckets in directory
  python preprocess_bucketed_rollouts.py -i ./bucketed -o ./preprocessed

  # Process specific stages
  python preprocess_bucketed_rollouts.py -i ./curriculum -o ./preprocessed \\
      --buckets stage1 stage2 stage3

  # Combine multiple stages for progressive training
  python preprocess_bucketed_rollouts.py -i ./bucketed -o ./preprocessed \\
      --buckets easy medium --combine --output-name easy_medium

  # Only include rollouts with positive scores
  python preprocess_bucketed_rollouts.py -i ./bucketed -o ./preprocessed \\
      --filter best_only

  # Only include rollouts with passing responses
  python preprocess_bucketed_rollouts.py -i ./bucketed -o ./preprocessed \\
      --filter passing_only
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Directory containing bucketed JSONL files'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directory to save preprocessed parquet files'
    )

    parser.add_argument(
        '--buckets',
        type=str,
        nargs='+',
        help='Specific bucket names to process (e.g., easy medium hard). If not provided, all .jsonl files will be processed'
    )

    parser.add_argument(
        '--combine',
        action='store_true',
        help='Combine all specified buckets into a single dataset'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default='combined',
        help='Name for output files (default: combined)'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        default='bucketed_curriculum',
        help='Name of the data source (default: bucketed_curriculum)'
    )

    parser.add_argument(
        '--filter',
        type=str,
        choices=['all', 'best_only', 'passing_only'],
        default='all',
        help='Filter mode: all (keep all), best_only (max score > 0), passing_only (any score >= 0.5)'
    )

    parser.add_argument(
        '--no-system-prompt',
        action='store_true',
        help='Do not add system/instruction prompts'
    )

    parser.add_argument(
        '--train-val-split',
        type=float,
        default=0.95,
        help='Train/validation split ratio (default: 0.95)'
    )

    parser.add_argument(
        '--no-val-split',
        action='store_true',
        help='Do not create validation split'
    )

    args = parser.parse_args()

    # Check for datasets library
    if not HAS_DATASETS:
        print("Error: datasets library is required.")
        print("Install with: pip install datasets")
        return

    # Create preprocessor
    preprocessor = RolloutPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filter_mode=args.filter,
        add_system_prompt=not args.no_system_prompt,
        train_val_split=args.train_val_split,
    )

    # Process buckets
    preprocessor.process_buckets(
        bucket_names=args.buckets,
        combine=args.combine,
        combined_name=args.output_name,
        data_source=args.data_source
    )


if __name__ == "__main__":
    main()
