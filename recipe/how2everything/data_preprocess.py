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

Handles two upstream data formats:
  - Flat: top-level {goal, steps, resources, topic, ...}
  - Nested (how2mine export): {source_example: {...}, final_procedure: {goal, steps, resources}}

Usage:
    python recipe/how2everything/data_preprocess.py \\
        --local_save_dir ~/data/how2everything \\
        --train_dataset how2everything/how2train_rl_100k \\
        --test_dataset how2everything/how2bench
"""

import argparse
import json
import os
import sys

import datasets

from verl.utils.hdfs_io import copy, makedirs

# Matches the upstream how2everything inference_inst.txt template.
INSTRUCTION_TEMPLATE = (
    "You will be given a goal and a list of resources. "
    "Your task is to output a list of steps that complete "
    "the goal using the given resources.\n\n"
    "Goal:\n{goal}\n\n"
    "Resources:\n{resources}\n\n"
    "Output exactly {n_steps} steps to achieve the goal using the given resources. "
    "Each step should be a single, concise sentence describing one primary action."
)


def _get_nested(example, key, default=None):
    """Get a value from an example dict, supporting dot-notation for nested access.

    Also handles the how2mine export format where fields are nested under
    'final_procedure' (goal, steps, resources) and 'source_example' (topic, url).
    """
    # Direct access first
    if key in example:
        return example[key]

    # Try nested under final_procedure (how2mine export format)
    fp = example.get("final_procedure")
    if isinstance(fp, dict) and key in fp:
        return fp[key]

    # Try dot-notation
    parts = key.split(".")
    obj = example
    for part in parts:
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return default
    return obj


def _format_resources(resources):
    """Format a list of resources into a bracketed string."""
    if isinstance(resources, list):
        return "[" + ", ".join(str(r) for r in resources) + "]"
    if resources:
        return str(resources)
    return "[]"


def _detect_schema(dataset):
    """Inspect the first example to determine which schema format is used.

    Returns a dict describing the detected format.
    """
    sample = dataset[0]
    columns = set(dataset.column_names)

    # Flat format: top-level goal, steps
    if "goal" in columns and "steps" in columns:
        has_resources = "resources" in columns
        has_topic = "topic" in columns
        return {"format": "flat", "has_resources": has_resources, "has_topic": has_topic}

    # Nested how2mine export format
    if "final_procedure" in columns:
        fp = sample.get("final_procedure", {})
        if isinstance(fp, dict) and "goal" in fp:
            has_resources = "resources" in fp
            se = sample.get("source_example", {})
            has_topic = isinstance(se, dict) and "topic" in se
            return {"format": "nested", "has_resources": has_resources, "has_topic": has_topic}

    return {"format": "unknown", "columns": list(columns)}


def make_map_fn(split, data_source):
    """Create a mapping function to convert raw examples to VeRL parquet schema."""

    def process_fn(example, idx):
        goal = _get_nested(example, "goal")
        steps = _get_nested(example, "steps", default=[])
        resources = _get_nested(example, "resources", default=[])

        # Get topic from top-level or source_example
        topic = example.get("topic")
        if topic is None:
            se = example.get("source_example")
            if isinstance(se, dict):
                topic = se.get("topic", "unknown")
            else:
                topic = "unknown"

        if not isinstance(steps, list):
            steps = [steps] if steps else []
        if not isinstance(resources, list):
            resources = [resources] if resources else []

        n_steps = len(steps)

        prompt_text = INSTRUCTION_TEMPLATE.format(
            goal=goal,
            resources=_format_resources(resources),
            n_steps=n_steps,
        )

        ground_truth = json.dumps({
            "goal": goal,
            "resources": resources,
            "steps": steps,
            "n_steps": n_steps,
        })

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "procedural",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "topic": topic,
                "goal": goal,
                "resources": resources,
                "reference_steps": steps,
            },
        }

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

    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Load training dataset
    print(f"Loading training dataset: {args.train_dataset}")
    train_dataset = datasets.load_dataset(args.train_dataset, split="train")
    print(f"  Loaded {len(train_dataset)} training examples")
    print(f"  Columns: {train_dataset.column_names}")

    schema_info = _detect_schema(train_dataset)
    print(f"  Detected schema: {schema_info}")
    if schema_info["format"] == "unknown":
        print(f"  ERROR: Could not detect schema. Columns: {schema_info['columns']}")
        print("  Expected either flat (goal, steps, resources) or nested (final_procedure.goal, ...)")
        sys.exit(1)

    # Load test/eval dataset
    print(f"Loading test dataset: {args.test_dataset}")
    test_dataset = datasets.load_dataset(args.test_dataset, split="train")
    print(f"  Loaded {len(test_dataset)} test examples")
    print(f"  Columns: {test_dataset.column_names}")

    test_schema = _detect_schema(test_dataset)
    print(f"  Detected schema: {test_schema}")
    if test_schema["format"] == "unknown":
        print(f"  ERROR: Could not detect schema. Columns: {test_schema['columns']}")
        sys.exit(1)

    # Transform to VeRL schema
    train_dataset = train_dataset.map(
        function=make_map_fn("train", "how2everything/how2train"),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", "how2everything/how2bench"),
        with_indices=True,
        remove_columns=test_dataset.column_names,
    )

    # Validate a sample
    sample = train_dataset[0]
    gt = json.loads(sample["reward_model"]["ground_truth"])
    print(f"\n  Sample goal: {gt['goal'][:80]}...")
    print(f"  Sample steps: {len(gt['steps'])} steps")
    print(f"  Sample resources: {len(gt['resources'])} resources")

    # Save to parquet
    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"\nSaved train parquet: {train_path} ({len(train_dataset)} rows)")
    print(f"Saved test parquet:  {test_path} ({len(test_dataset)} rows)")

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
