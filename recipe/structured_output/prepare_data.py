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
Preprocess the nvidia/Nemotron-RL-instruction_following-structured_outputs dataset
to veRL's parquet format for structured output RL training.

Usage:
    python -m recipe.structured_output.prepare_data \
        --local_dir ~/data/structured_output \
        --split train
"""

import argparse
import json
import os
from functools import partial

from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs


def process_nemotron_example(example: dict) -> dict:
    """Convert a Nemotron structured output example to veRL format.

    The Nemotron dataset has:
    - responses_create_params: dict with "input" containing the conversation
    - schema_str: JSON schema string
    - schema_type: always "json"
    - schema_fields_count: number of required fields

    We convert to veRL format:
    - prompt: list of message dicts
    - data_source: identifier string
    - reward_model: dict with ground_truth containing schema info
    - extra_info: additional metadata
    """
    # Extract the input messages from the dataset
    params = example.get("responses_create_params", {})
    messages = params.get("input", [])

    # The messages are already in the format [{"role": "user", "content": "..."}]
    if not messages:
        # Fallback: construct from raw content if structure differs
        messages = [{"role": "user", "content": str(params)}]

    schema_str = example.get("schema_str", "{}")
    schema_type = example.get("schema_type", "json")
    schema_fields_count = example.get("schema_fields_count", "0")

    # The ground truth for structured output is the schema itself
    # The reward function validates whether the model output conforms to this schema
    data = {
        "data_source": "structured_output",
        "prompt": messages,
        "ability": "structured_output",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({"schema_str": schema_str}),
        },
        "extra_info": {
            "schema_str": schema_str,
            "schema_type": schema_type,
            "schema_fields_count": str(schema_fields_count),
        },
    }
    return data


def build_nemotron_dataset(split: str = "train"):
    """Load and process the Nemotron structured output dataset.

    Args:
        split: Dataset split to load ("train" or "validation").
    """
    data_source = "nvidia/Nemotron-RL-instruction_following-structured_outputs"
    print(f"Loading the {data_source} dataset from HuggingFace (split={split})...", flush=True)

    dataset = load_dataset(data_source, split=split)
    print(f"Loaded {len(dataset)} examples", flush=True)

    dataset = dataset.map(
        lambda example, idx: {**process_nemotron_example(example), "extra_info": {**process_nemotron_example(example)["extra_info"], "index": idx, "split": split}},
        with_indices=True,
        remove_columns=dataset.column_names,
    )
    return dataset


def build_custom_schema_dataset(schema_file: str):
    """Build a dataset from a custom JSON file containing schemas.

    The file should contain a JSON array of objects with:
    - "prompt": The user prompt/document
    - "schema": The JSON schema to validate against
    - "expected_output": (optional) The expected output

    Args:
        schema_file: Path to the JSON file.
    """
    with open(schema_file) as f:
        examples = json.load(f)

    processed = []
    for idx, example in enumerate(examples):
        prompt_text = example.get("prompt", "")
        schema = example.get("schema", {})
        expected_output = example.get("expected_output")

        ground_truth = {"schema_str": json.dumps(schema)}
        if expected_output:
            ground_truth["answer"] = json.dumps(expected_output) if not isinstance(expected_output, str) else expected_output

        processed.append({
            "data_source": "structured_output",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "structured_output",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(ground_truth),
            },
            "extra_info": {
                "schema_str": json.dumps(schema),
                "schema_type": "json",
                "index": idx,
                "split": "custom",
            },
        })

    from datasets import Dataset

    return Dataset.from_list(processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare structured output dataset for veRL training")
    parser.add_argument("--local_dir", default="~/data/structured_output", help="Local output directory")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS output directory")
    parser.add_argument("--split", default="all", choices=["train", "validation", "all"], help="Dataset split to process")
    parser.add_argument("--custom_schema_file", default=None, help="Path to custom schema JSON file")
    parser.add_argument("--train_repeat", type=int, default=1, help="Number of times to repeat training data")

    args = parser.parse_args()
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    if args.custom_schema_file:
        custom_dataset = build_custom_schema_dataset(args.custom_schema_file)
        output_path = os.path.join(local_dir, "custom_structured_output.parquet")
        custom_dataset.to_parquet(output_path)
        print(f"Saved custom dataset ({len(custom_dataset)} examples) to {output_path}")
    else:
        if args.split in ("train", "all"):
            train_dataset = build_nemotron_dataset(split="train")
            if args.train_repeat > 1:
                from datasets import concatenate_datasets

                train_dataset = concatenate_datasets([train_dataset for _ in range(args.train_repeat)])
                print(f"Repeated training data {args.train_repeat}x -> {len(train_dataset)} examples")
            output_path = os.path.join(local_dir, "structured_output_train.parquet")
            train_dataset.to_parquet(output_path)
            print(f"Saved train dataset ({len(train_dataset)} examples) to {output_path}")

        if args.split in ("validation", "all"):
            val_dataset = build_nemotron_dataset(split="validation")
            output_path = os.path.join(local_dir, "structured_output_val.parquet")
            val_dataset.to_parquet(output_path)
            print(f"Saved validation dataset ({len(val_dataset)} examples) to {output_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
