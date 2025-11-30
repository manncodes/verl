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
Preprocess AIME 2025 and IFEval datasets into a combined test.parquet for evaluation.

This script combines:
1. AIME 2025: Advanced math reasoning problems
2. IFEval: Instruction following evaluation (Google's official version)

The output format is compatible with the batched IFEval reward function.

Usage:
    python aime2025_ifeval_test.py --local_save_dir ~/data/aime_ifeval_test
    python aime2025_ifeval_test.py --local_save_dir ~/data/aime_ifeval_test --aime_only
    python aime2025_ifeval_test.py --local_save_dir ~/data/aime_ifeval_test --ifeval_only
"""

import argparse
import json
import os
import shutil
from typing import Optional

import datasets


def process_aime_example(example, idx):
    """Process AIME 2025 example into standard format.

    AIME dataset typically has:
    - problem: The math problem text
    - solution/answer: Ground truth answer
    """
    problem = example.get("problem", "") or example.get("question", "")
    answer = example.get("answer", "") or example.get("solution", "")

    # Add reasoning instruction
    prompt_with_instruction = (
        f"{problem}\n\n"
        "Please solve this problem step by step. "
        "Put your reasoning in <think>...</think> tags and your final answer in <answer>...</answer> tags."
    )

    data = {
        "data_source": "aime_2025",
        "prompt": [
            {
                "role": "user",
                "content": prompt_with_instruction,
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer": answer,
            },
        },
        "extra_info": {
            "ability": "math",
            "split": "test",
            "index": idx,
            "original_problem": problem,
            "dataset_source": "aime_2025",
            "answer": answer,
            "prompt": problem,  # For consistency with IFEval
        },
    }
    return data


def process_ifeval_example(example, idx):
    """Process IFEval example into standard format.

    IFEval dataset has:
    - prompt: The instruction to follow
    - instruction_id_list: List of instruction constraint IDs
    - kwargs: List of kwargs for each instruction
    """
    prompt_text = example.get("prompt", "")
    instruction_id_list = example.get("instruction_id_list", [])
    kwargs = example.get("kwargs", [])

    # Add reasoning instruction with IFEval format
    prompt_with_instruction = (
        f"{prompt_text}\n\n"
        "Please follow the instructions carefully. "
        "Put your reasoning in <think>...</think> tags and your response in <answer>...</answer> tags."
    )

    data = {
        "data_source": "ifeval",
        "prompt": [
            {
                "role": "user",
                "content": prompt_with_instruction,
            }
        ],
        "ability": "instruction_following",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
            },
        },
        "extra_info": {
            "ability": "instruction_following",
            "split": "test",
            "index": idx,
            "original_prompt": prompt_text,
            "dataset_source": "ifeval",
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
            "prompt": prompt_text,  # Required for judge evaluation
        },
    }
    return data


def load_aime_dataset(local_dataset_path: Optional[str] = None):
    """Load AIME 2025 dataset.

    Tries multiple common dataset names/formats for AIME 2025.
    """
    possible_sources = [
        "Maxwell-Jia/AIME_2024",  # Common AIME dataset on HF
        "aime2024",
        "hendrycks/math",  # Contains AIME problems
    ]

    if local_dataset_path is not None:
        print(f"Loading AIME from local path: {local_dataset_path}")
        return datasets.load_dataset(local_dataset_path)

    for source in possible_sources:
        try:
            print(f"Trying to load AIME from {source}...")
            dataset = datasets.load_dataset(source)
            print(f"Successfully loaded AIME from {source}")
            return dataset
        except Exception as e:
            print(f"Failed to load from {source}: {e}")
            continue

    raise ValueError(
        "Could not load AIME dataset. Please specify --aime_local_path or "
        "ensure one of the standard AIME datasets is available."
    )


def load_ifeval_dataset(local_dataset_path: Optional[str] = None):
    """Load IFEval dataset (Google's official version).

    The official IFEval dataset is at google/IFEval.
    """
    if local_dataset_path is not None:
        print(f"Loading IFEval from local path: {local_dataset_path}")
        return datasets.load_dataset(local_dataset_path)

    try:
        print("Loading IFEval from google/IFEval...")
        dataset = datasets.load_dataset("google/IFEval")
        print("Successfully loaded IFEval")
        return dataset
    except Exception as e:
        # Fallback to other possible names
        try:
            print("Trying alternative: instruction-following-eval/ifeval...")
            dataset = datasets.load_dataset("instruction-following-eval/ifeval")
            print("Successfully loaded IFEval from alternative source")
            return dataset
        except:
            raise ValueError(
                f"Could not load IFEval dataset: {e}\n"
                "Please specify --ifeval_local_path or install the dataset manually."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess AIME 2025 and IFEval datasets for evaluation"
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/aime_ifeval_test",
        help="Directory to save the preprocessed test.parquet file.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Additional directory to copy processed data to (optional).",
    )
    parser.add_argument(
        "--aime_local_path",
        default=None,
        help="Local path to AIME dataset if already downloaded.",
    )
    parser.add_argument(
        "--ifeval_local_path",
        default=None,
        help="Local path to IFEval dataset if already downloaded.",
    )
    parser.add_argument(
        "--aime_only",
        action="store_true",
        help="Only include AIME examples (no IFEval).",
    )
    parser.add_argument(
        "--ifeval_only",
        action="store_true",
        help="Only include IFEval examples (no AIME).",
    )
    parser.add_argument(
        "--max_aime_samples",
        type=int,
        default=None,
        help="Maximum number of AIME samples to include (for testing).",
    )
    parser.add_argument(
        "--max_ifeval_samples",
        type=int,
        default=None,
        help="Maximum number of IFEval samples to include (for testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling. Default: 42",
    )

    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.aime_only and args.ifeval_only:
        parser.error("--aime_only and --ifeval_only are mutually exclusive")

    all_examples = []

    # Load and process AIME
    if not args.ifeval_only:
        print("=" * 80)
        print("Loading AIME 2025 dataset...")
        print("=" * 80)

        aime_dataset = load_aime_dataset(args.aime_local_path)

        # Get the test split (or train if test doesn't exist)
        if "test" in aime_dataset:
            aime_data = aime_dataset["test"]
        elif "validation" in aime_dataset:
            aime_data = aime_dataset["validation"]
        elif "train" in aime_dataset:
            aime_data = aime_dataset["train"]
        else:
            # Single dataset without splits
            aime_data = aime_dataset

        print(f"Loaded {len(aime_data)} AIME examples")

        # Limit samples if specified
        if args.max_aime_samples is not None:
            aime_data = aime_data.select(range(min(args.max_aime_samples, len(aime_data))))
            print(f"Limited to {len(aime_data)} AIME samples")

        # Process AIME examples
        print("Processing AIME examples...")
        for idx, example in enumerate(aime_data):
            processed = process_aime_example(example, idx)
            all_examples.append(processed)

        print(f"Processed {len(all_examples)} AIME examples")

    # Load and process IFEval
    if not args.aime_only:
        print("=" * 80)
        print("Loading IFEval dataset...")
        print("=" * 80)

        ifeval_dataset = load_ifeval_dataset(args.ifeval_local_path)

        # Get the test split (IFEval typically has 'train' which is the eval set)
        if "test" in ifeval_dataset:
            ifeval_data = ifeval_dataset["test"]
        elif "train" in ifeval_dataset:
            ifeval_data = ifeval_dataset["train"]
        else:
            # Single dataset without splits
            ifeval_data = ifeval_dataset

        print(f"Loaded {len(ifeval_data)} IFEval examples")

        # Limit samples if specified
        if args.max_ifeval_samples is not None:
            ifeval_data = ifeval_data.select(range(min(args.max_ifeval_samples, len(ifeval_data))))
            print(f"Limited to {len(ifeval_data)} IFEval samples")

        # Process IFEval examples
        print("Processing IFEval examples...")
        base_idx = len(all_examples)
        for idx, example in enumerate(ifeval_data):
            processed = process_ifeval_example(example, base_idx + idx)
            all_examples.append(processed)

        print(f"Processed {len(all_examples) - base_idx} IFEval examples")

    print("=" * 80)
    print(f"Total examples: {len(all_examples)}")
    print("=" * 80)

    # Convert to dataset
    combined_dataset = datasets.Dataset.from_list(all_examples)

    # Shuffle with seed for reproducibility
    combined_dataset = combined_dataset.shuffle(seed=args.seed)

    # Save to disk
    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    test_path = os.path.join(local_dir, "test.parquet")

    print(f"Saving combined test dataset to {test_path}...")
    combined_dataset.to_parquet(test_path)

    # Save example JSON for reference
    if len(combined_dataset) > 0:
        # Save one example from each dataset type if available
        examples_to_save = {}

        for example in combined_dataset:
            source = example["data_source"]
            if source not in examples_to_save:
                examples_to_save[source] = example

        # Save combined example
        example_path = os.path.join(local_dir, "test_example.json")
        with open(example_path, "w") as f:
            json.dump(examples_to_save, f, indent=2, default=str)
        print(f"Saved example(s) to {example_path}")

    # Copy to additional directory if specified
    if args.hdfs_dir is not None:
        print(f"Copying to additional location: {args.hdfs_dir}...")
        os.makedirs(args.hdfs_dir, exist_ok=True)
        shutil.copytree(local_dir, args.hdfs_dir, dirs_exist_ok=True)
        print("Copy complete.")

    # Print statistics
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    print(f"Total examples: {len(combined_dataset)}")

    # Count by data source
    from collections import Counter
    source_counts = Counter(example["data_source"] for example in combined_dataset)
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} examples")

    print(f"\nOutput: {test_path}")
    print("=" * 80)
