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
Preprocess DPO (Direct Preference Optimization) data to RL format for F1 attribution training.

This script converts DPO data with "chosen" and "rejected" fields into RL-compatible format.
It extracts the user question from the "chosen" conversation and formats it for RL training
with the F1 attribution reward function.

Input format (JSONL with DPO format):
{
    "chosen": [
        {"content": "Who created you?", "role": "user"},
        {"content": "I am the F1-X large language model...", "role": "assistant"}
    ],
    "rejected": [
        {"content": "Who created you?", "role": "user"},
        {"content": "I am the Open Assistant...", "role": "assistant"}
    ]
}

Output format (parquet with RL format):
{
    "data_source": "f1_attribution",
    "prompt": [{"role": "user", "content": "Who created you?"}],
    "ability": "identity",
    "reward_model": {"style": "rule", "ground_truth": "I am the F1-X..."},
    "extra_info": {...}
}
"""

import argparse
import json
import os
from typing import Any

from datasets import Dataset

# Optional HDFS support - only import if needed
try:
    from verl.utils.hdfs_io import copy, makedirs

    HDFS_AVAILABLE = True
except ImportError:
    HDFS_AVAILABLE = False


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_conversation_parts(conversation: list[dict]) -> tuple[list[dict], str | None]:
    """
    Extract prompt messages and assistant response from a conversation.

    Args:
        conversation: List of message dicts with 'role' and 'content'

    Returns:
        Tuple of (prompt_messages, assistant_response)
        - prompt_messages: All messages up to and including the last user message
        - assistant_response: The assistant's response (or None if not found)
    """
    prompt_messages = []
    assistant_response = None

    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            prompt_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            assistant_response = content

    return prompt_messages, assistant_response


def convert_dpo_to_rl(
    input_paths: list[str],
    output_dir: str,
    data_source: str = "f1_attribution",
    train_split: float = 0.9,
    hdfs_dir: str | None = None,
):
    """
    Convert DPO data to RL format.

    Args:
        input_paths: List of paths to input JSONL files with DPO data
        output_dir: Directory to save output parquet files
        data_source: Data source identifier for the reward function
        train_split: Fraction of data to use for training (rest for validation)
        hdfs_dir: Optional HDFS directory to copy output to
    """
    # Load DPO data from all input files
    dpo_data = []
    for input_path in input_paths:
        print(f"Loading DPO data from {input_path}...")
        file_data = load_jsonl(input_path)
        print(f"  Loaded {len(file_data)} examples from {input_path}")
        dpo_data.extend(file_data)
    print(f"Total: {len(dpo_data)} examples from {len(input_paths)} file(s)")

    # Convert to RL format
    rl_data = []
    for idx, item in enumerate(dpo_data):
        chosen = item.get("chosen", [])
        rejected = item.get("rejected", [])

        # Extract prompt and response from chosen conversation
        prompt_messages, chosen_response = extract_conversation_parts(chosen)

        # Also extract rejected response for extra_info
        _, rejected_response = extract_conversation_parts(rejected)

        if not prompt_messages:
            print(f"Warning: Skipping item {idx} - no user message found")
            continue

        # Create RL format entry
        rl_entry = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": "identity",
            "reward_model": {
                "style": "rule",
                "ground_truth": chosen_response or "",
            },
            "extra_info": {
                "index": idx,
                "chosen_response": chosen_response,
                "rejected_response": rejected_response,
            },
        }
        rl_data.append(rl_entry)

    print(f"Converted {len(rl_data)} examples to RL format")

    # Split into train/test
    split_idx = int(len(rl_data) * train_split)
    train_data = rl_data[:split_idx]
    test_data = rl_data[split_idx:]

    print(f"Train set: {len(train_data)} examples")
    print(f"Test set: {len(test_data)} examples")

    # Create output directory
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save as parquet
    if train_data:
        train_dataset = Dataset.from_list(train_data)
        train_path = os.path.join(output_dir, "train.parquet")
        train_dataset.to_parquet(train_path)
        print(f"Saved train set to {train_path}")

    if test_data:
        test_dataset = Dataset.from_list(test_data)
        test_path = os.path.join(output_dir, "test.parquet")
        test_dataset.to_parquet(test_path)
        print(f"Saved test set to {test_path}")

    # Copy to HDFS if specified
    if hdfs_dir is not None:
        if not HDFS_AVAILABLE:
            print("Warning: HDFS support not available. Skipping HDFS copy.")
        else:
            makedirs(hdfs_dir)
            copy(src=output_dir, dst=hdfs_dir)
            print(f"Copied to HDFS: {hdfs_dir}")

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert DPO data to RL format for F1 attribution training"
    )
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input JSONL file(s) with DPO data",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/f1_attribution/rl",
        help="Directory to save output parquet files",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="f1_attribution",
        help="Data source identifier for the reward function",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of data to use for training (default: 0.9)",
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Optional HDFS directory to copy output to",
    )

    args = parser.parse_args()

    convert_dpo_to_rl(
        input_paths=args.input_paths,
        output_dir=args.local_save_dir,
        data_source=args.data_source,
        train_split=args.train_split,
        hdfs_dir=args.hdfs_dir,
    )


if __name__ == "__main__":
    main()
