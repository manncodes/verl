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
Preprocess the RLVR-IFeval dataset for GRPO/PPO training.

RLVR-IFeval is a synthetic instruction-following dataset from AllenAI designed for
reinforcement learning with verifiable rewards. It contains prompts with constraints
sampled from the Tulu 2 SFT mixture with randomly added IFEval-style constraints.

Key differences from google/IFEval:
- Contains training data (not just evaluation)
- Uses "messages" format (chat-style) instead of single "prompt"
- ground_truth is a JSON string
- Includes constraint_type and constraint description

Dataset: https://huggingface.co/datasets/allenai/RLVR-IFeval
Reference: Part of the Tulu 3 release (2024)
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists."
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/rlvr_ifeval", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the dataset to.")
    parser.add_argument(
        "--add_instruction_prompt",
        action="store_true",
        help="Whether to add instruction following guidance to the prompt.",
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.95,
        help="Ratio of data to use for training (default: 0.95, rest for validation)",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "allenai/RLVR-IFeval"

    # Load dataset
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # RLVR-IFeval has a train split - this is actual training data
    train_dataset = dataset["train"]

    # Optional: Add instruction following guidance
    instruction_following_prompt = (
        "Please follow all the instructions in the prompt carefully and precisely."
        if args.add_instruction_prompt
        else ""
    )

    def process_fn(example, idx):
        """Process each example into the verl format.

        The RLVR-IFeval dataset has the following fields:
        - messages: List of message dicts with 'role' and 'content' (chat format)
        - ground_truth: JSON string containing constraint parameters
        - constraint_type: Type of constraint (e.g., "keywords:existence", "length_constraints:number_words")
        - constraint: Plain English description of the constraint
        """
        # Extract messages (chat format)
        messages = example.get("messages", [])

        # Parse ground_truth JSON string
        ground_truth_str = example.get("ground_truth", "{}")
        try:
            ground_truth_dict = json.loads(ground_truth_str)
        except json.JSONDecodeError:
            ground_truth_dict = {}

        # Extract constraint information
        constraint_type = example.get("constraint_type", "")
        constraint = example.get("constraint", "")

        # Convert ground_truth dict to instruction_id_list and kwargs format
        # that matches the IFEval reward function expectations
        instruction_id_list = [constraint_type] if constraint_type else []
        kwargs_list = [ground_truth_dict] if ground_truth_dict else []

        # Build prompt from messages
        # Typically last user message or concatenate all
        prompt_messages = []
        user_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                user_content = content
                prompt_messages.append({"role": "user", "content": content})
            elif role == "system":
                # Prepend system message if exists
                prompt_messages.insert(0, {"role": "system", "content": content})

        # Add optional instruction following guidance to user message
        if instruction_following_prompt and prompt_messages:
            for msg in prompt_messages:
                if msg["role"] == "user":
                    msg["content"] = instruction_following_prompt + "\n\n" + msg["content"]
                    break

        # If no messages, create a default structure
        if not prompt_messages:
            prompt_messages = [{"role": "user", "content": user_content or constraint}]

        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": "instruction_following",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs_list,
                },
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "key": f"rlvr_ifeval_{idx}",
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs_list,
                "constraint_type": constraint_type,
                "constraint": constraint,
                "prompt": user_content,  # Store the actual user prompt
            },
        }
        return data

    # Process dataset
    processed_dataset = train_dataset.map(function=process_fn, with_indices=True)

    # Split into train and validation
    split_ratio = args.train_split_ratio
    split_dataset = processed_dataset.train_test_split(test_size=1 - split_ratio, seed=42)
    final_train_dataset = split_dataset["train"]
    final_val_dataset = split_dataset["test"]

    # Save datasets
    hdfs_dir = args.hdfs_dir
    local_save_dir = os.path.expanduser(args.local_save_dir)

    # Create directory if it doesn't exist
    os.makedirs(local_save_dir, exist_ok=True)

    # Save train and validation sets
    final_train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    final_val_dataset.to_parquet(os.path.join(local_save_dir, "val.parquet"))

    print(f"Dataset saved to {local_save_dir}")
    print(f"Train examples: {len(final_train_dataset)}")
    print(f"Validation examples: {len(final_val_dataset)}")

    # Optional: Copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
        print(f"Dataset copied to HDFS: {hdfs_dir}")

    # Print example
    if len(final_train_dataset) > 0:
        print("\n=== Example Training Data ===")
        example = final_train_dataset[0]
        print(f"Prompt: {example['prompt']}")
        print(f"Constraint: {example['extra_info']['constraint']}")
        print(f"Constraint Type: {example['extra_info']['constraint_type']}")
        print(f"Ground Truth: {example['reward_model']['ground_truth']}")
