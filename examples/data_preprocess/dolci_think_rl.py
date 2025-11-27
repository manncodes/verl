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
Preprocess the allenai/Dolci-Think-RL dataset to parquet format for RL training.

This dataset contains 102,026 high-quality prompts for deliberate reasoning RL training,
covering diverse reasoning tasks including mathematics, logic, and problem-solving.

Dataset schema:
    - prompt: The input prompt/question (string, 11-12.3k chars)
    - ground_truth: Ground truth answer values (list with 1 element)
    - dataset: Dataset source information (list)
    - original_dataset: Original source dataset name (15 unique values)
    - dataset_source: Dataset source category (4 unique values: math, code, IF, chat)
    - outputs: Model outputs/responses (nullable list)
    - custom_id, id, key: Identifiers
    - constraint_type, constraint: Constraint info
    - conversation_hash, model, predicted_label: Metadata

Usage:
    python dolci_think_rl.py --local_save_dir ~/data/dolci_think_rl
    python dolci_think_rl.py --local_save_dir ~/data/dolci_think_rl --train_ratio 0.95
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def get_ability_from_source(dataset_source: str) -> str:
    """Map dataset_source to ability category."""
    source_to_ability = {
        "math": "math",
        "code": "code",
        "IF": "instruction_following",
        "chat": "chat",
    }
    return source_to_ability.get(dataset_source, "reasoning")


def get_reward_style(dataset_source: str) -> str:
    """Determine reward style based on dataset source.

    Math and code tasks typically have verifiable/rule-based rewards,
    while chat and IF may require model-based evaluation.
    """
    if dataset_source in ("math", "code"):
        return "rule"
    return "rule"  # Dolci-Think-RL uses RLVR (verifiable rewards) for all tasks


def extract_ground_truth(ground_truth_list):
    """Extract ground truth from the list format.

    The ground_truth field is a list with typically 1 element.
    """
    if ground_truth_list is None or len(ground_truth_list) == 0:
        return None
    return ground_truth_list[0]


def extract_dataset_info(dataset_list):
    """Extract dataset info from the list format."""
    if dataset_list is None or len(dataset_list) == 0:
        return None
    return dataset_list[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess allenai/Dolci-Think-RL dataset")
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy processed data to.")
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to the raw dataset if already downloaded.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dolci_think_rl",
        help="Directory to save the preprocessed dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Ratio of data to use for training (rest goes to test). Default: 0.95",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split. Default: 42",
    )
    parser.add_argument(
        "--filter_null_ground_truth",
        action="store_true",
        help="Filter out examples with null ground_truth values.",
    )

    args = parser.parse_args()

    data_source = "allenai/Dolci-Think-RL"

    print(f"Loading the {data_source} dataset from HuggingFace...", flush=True)

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # The dataset may have a single 'train' split - we'll create our own train/test split
    if "train" in dataset:
        full_dataset = dataset["train"]
    else:
        # If it's a DatasetDict with other splits, concatenate them
        full_dataset = datasets.concatenate_datasets(list(dataset.values()))

    print(f"Loaded {len(full_dataset)} examples from {data_source}", flush=True)

    # Optionally filter out examples with null ground_truth
    if args.filter_null_ground_truth:
        original_size = len(full_dataset)
        full_dataset = full_dataset.filter(
            lambda x: x.get("ground_truth") is not None and len(x.get("ground_truth", [])) > 0
        )
        print(f"Filtered out {original_size - len(full_dataset)} examples with null ground_truth", flush=True)

    # Create train/test split
    split_dataset = full_dataset.train_test_split(
        train_size=args.train_ratio,
        seed=args.seed,
    )
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}", flush=True)

    # Instruction for thinking/reasoning format
    # Dolci-Think models use <think>...</think> tags for reasoning
    instruction_following = "Think step by step, showing your reasoning in <think>...</think> tags, then provide your final answer."

    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract the prompt
            prompt_text = example.get("prompt", "")

            # Add instruction if not already present
            if "<think>" not in prompt_text.lower() and "think step by step" not in prompt_text.lower():
                prompt_with_instruction = prompt_text + "\n\n" + instruction_following
            else:
                prompt_with_instruction = prompt_text

            # Extract ground truth (it's a list with typically 1 element)
            ground_truth = extract_ground_truth(example.get("ground_truth"))

            # Get dataset source category for determining ability and reward style
            dataset_source = example.get("dataset_source", "")
            ability = get_ability_from_source(dataset_source)
            reward_style = get_reward_style(dataset_source)

            # Build the standardized data format for verl
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_with_instruction,
                    }
                ],
                "ability": ability,
                "reward_model": {
                    "style": reward_style,
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "original_prompt": prompt_text,
                    "original_dataset": example.get("original_dataset"),
                    "dataset_source": dataset_source,
                    "custom_id": example.get("custom_id"),
                    "constraint_type": example.get("constraint_type"),
                    "constraint": example.get("constraint"),
                },
            }
            return data

        return process_fn

    print("Processing train dataset...", flush=True)
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    print("Processing test dataset...", flush=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Handle save directory
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    hdfs_dir = args.hdfs_dir

    # Save to parquet format
    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")

    print(f"Saving train dataset to {train_path}...", flush=True)
    train_dataset.to_parquet(train_path)

    print(f"Saving test dataset to {test_path}...", flush=True)
    test_dataset.to_parquet(test_path)

    # Save example JSONs for reference
    if len(train_dataset) > 0:
        example = train_dataset[0]
        example_path = os.path.join(local_dir, "train_example.json")
        with open(example_path, "w") as f:
            json.dump(example, f, indent=2, default=str)
        print(f"Saved train example to {example_path}", flush=True)

    if len(test_dataset) > 0:
        example = test_dataset[0]
        example_path = os.path.join(local_dir, "test_example.json")
        with open(example_path, "w") as f:
            json.dump(example, f, indent=2, default=str)
        print(f"Saved test example to {example_path}", flush=True)

    # Copy to HDFS if specified
    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}...", flush=True)
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print("HDFS copy complete.", flush=True)

    print("Preprocessing complete!", flush=True)
    print(f"  Train: {len(train_dataset)} examples -> {train_path}")
    print(f"  Test: {len(test_dataset)} examples -> {test_path}")
