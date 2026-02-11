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
Preprocess the how2everything dataset to parquet format for VeRL training.

Downloads how2everything/how2train_rl_100k (train) and how2everything/how2bench (eval)
from HuggingFace and converts them to the VeRL-expected parquet schema.

Usage:
    python recipe/how2everything/data_preprocess.py \
        --local_save_dir ~/data/how2everything \
        --train_dataset how2everything/how2train_rl_100k \
        --test_dataset how2everything/how2bench
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION_TEMPLATE = (
    "You will be given a goal and a list of resources. "
    "Your task is to output a list of steps that complete "
    "the goal using the given resources.\n\n"
    "Goal:\n{goal}\n\n"
    "Resources:\n{resources}\n\n"
    "Output exactly {n_steps} steps to achieve the goal using the given resources. "
    "Each step should be a single, concise sentence describing one primary action."
)


def format_resources(resources):
    """Format a list of resources into a bracketed string."""
    if isinstance(resources, list):
        return "[" + ", ".join(str(r) for r in resources) + "]"
    return str(resources)


def format_steps(steps):
    """Format a list of steps into numbered lines."""
    if isinstance(steps, list):
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
    return str(steps)


def make_map_fn(split, data_source, goal_key="goal", resources_key="resources", steps_key="steps"):
    """Create a mapping function to convert raw examples to VeRL parquet schema.

    Args:
        split: "train" or "test"
        data_source: Identifier string for reward routing
        goal_key: Column name for the goal field
        resources_key: Column name for the resources field
        steps_key: Column name for the steps field
    """

    def process_fn(example, idx):
        goal = example[goal_key]
        resources = example[resources_key]
        steps = example[steps_key]
        n_steps = len(steps) if isinstance(steps, list) else 1

        resources_str = format_resources(resources)

        prompt_text = INSTRUCTION_TEMPLATE.format(
            goal=goal,
            resources=resources_str,
            n_steps=n_steps,
        )

        ground_truth = json.dumps(
            {
                "goal": goal,
                "resources": resources if isinstance(resources, list) else [resources],
                "steps": steps if isinstance(steps, list) else [steps],
                "n_steps": n_steps,
            }
        )

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "ability": "procedural",
            "reward_model": {"style": "genrm", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "topic": example.get("topic", "unknown"),
                "goal": goal,
                "resources": resources if isinstance(resources, list) else [resources],
                "reference_steps": steps if isinstance(steps, list) else [steps],
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess how2everything datasets for VeRL")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/how2everything",
        help="Directory to save preprocessed parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy outputs to.",
    )
    parser.add_argument(
        "--train_dataset",
        default="how2everything/how2train_rl_100k",
        help="HuggingFace dataset ID for training data.",
    )
    parser.add_argument(
        "--test_dataset",
        default="how2everything/how2bench",
        help="HuggingFace dataset ID for evaluation data.",
    )
    parser.add_argument(
        "--goal_key",
        default="goal",
        help="Column name for goal field in the source dataset.",
    )
    parser.add_argument(
        "--resources_key",
        default="resources",
        help="Column name for resources field in the source dataset.",
    )
    parser.add_argument(
        "--steps_key",
        default="steps",
        help="Column name for steps field in the source dataset.",
    )

    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Load training dataset
    print(f"Loading training dataset: {args.train_dataset}")
    train_dataset = datasets.load_dataset(args.train_dataset, split="train")
    print(f"  Loaded {len(train_dataset)} training examples")
    print(f"  Columns: {train_dataset.column_names}")

    # Load test/eval dataset
    print(f"Loading test dataset: {args.test_dataset}")
    test_dataset = datasets.load_dataset(args.test_dataset, split="train")
    print(f"  Loaded {len(test_dataset)} test examples")
    print(f"  Columns: {test_dataset.column_names}")

    # Transform to VeRL schema
    train_data_source = "how2everything/how2train"
    test_data_source = "how2everything/how2bench"

    train_dataset = train_dataset.map(
        function=make_map_fn(
            "train", train_data_source, args.goal_key, args.resources_key, args.steps_key
        ),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn(
            "test", test_data_source, args.goal_key, args.resources_key, args.steps_key
        ),
        with_indices=True,
        remove_columns=test_dataset.column_names,
    )

    # Save to parquet
    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"Saved train parquet: {train_path} ({len(train_dataset)} rows)")
    print(f"Saved test parquet:  {test_path} ({len(test_dataset)} rows)")

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
