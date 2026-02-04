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
to parquet format for RLVR training with verl.

The dataset tests the ability of models to follow output formatting instructions
under JSON schema constraints. Each example contains:
  - responses_create_params: OpenAI Responses API formatted input messages
  - schema_str: JSON schema the model output must conform to
  - schema_type: Type of schema (currently only "json")
  - schema_fields_count: Number of fields in the schema

The reward function validates JSON schema adherence (binary: 1.0 or 0.0).

Usage:
    python structured_output.py --local_save_dir ~/data/structured_output
    python structured_output.py --local_dataset_path /path/to/local/dataset --local_save_dir ~/data/structured_output
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


DATA_SOURCE = "nvidia/Nemotron-RL-instruction_following-structured_outputs"


def extract_prompt_from_responses_create_params(responses_create_params):
    """Convert responses_create_params (OpenAI Responses API format) to chat messages.

    The responses_create_params contains an 'input' field with messages in the format:
        [{"content": "...", "role": "user", "type": "message"}, ...]

    We convert this to the standard chat format:
        [{"role": "user", "content": "..."}, ...]

    Args:
        responses_create_params: Dict with 'input' field containing messages.

    Returns:
        List of chat messages in {"role": ..., "content": ...} format.
    """
    input_messages = responses_create_params.get("input", [])
    prompt = []
    for msg in input_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt.append({"role": role, "content": content})
    return prompt


def make_map_fn(split):
    """Create a map function for processing dataset examples.

    Args:
        split: The dataset split name ("train" or "validation").

    Returns:
        A function that processes individual examples.
    """

    def process_fn(example, idx):
        # Parse responses_create_params - it may be a string or dict
        responses_create_params = example["responses_create_params"]
        if isinstance(responses_create_params, str):
            responses_create_params = json.loads(responses_create_params)

        # Extract prompt messages from the API format
        prompt = extract_prompt_from_responses_create_params(responses_create_params)

        # The schema_str is the ground truth for validation
        schema_str = example["schema_str"]
        schema_type = example.get("schema_type", "json")
        schema_fields_count = example.get("schema_fields_count", None)

        data = {
            "data_source": DATA_SOURCE,
            "prompt": prompt,
            "ability": "structured_output",
            "reward_model": {"style": "rule", "ground_truth": schema_str},
            "extra_info": {
                "split": split,
                "index": idx,
                "schema_type": schema_type,
                "schema_fields_count": schema_fields_count,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Nemotron structured output dataset for RLVR training.")
    parser.add_argument(
        "--local_dataset_path", default=None, help="Local path to the raw dataset, if it exists."
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the processed data to.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/structured_output",
        help="The local save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(DATA_SOURCE)

    train_dataset = dataset["train"]

    # The dataset has a validation split
    if "validation" in dataset:
        val_dataset = dataset["validation"]
    else:
        val_dataset = None

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    print(f"Saved {len(train_dataset)} train examples to {os.path.join(local_save_dir, 'train.parquet')}")

    if val_dataset is not None:
        val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
        val_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))
        print(f"Saved {len(val_dataset)} validation examples to {os.path.join(local_save_dir, 'validation.parquet')}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"Copied processed data to {args.hdfs_dir}")
