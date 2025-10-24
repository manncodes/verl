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
Preprocess the IFEval (Instruction Following Evaluation) dataset to parquet format.

IFEval is a benchmark for evaluating instruction following capabilities of LLMs.
It contains verifiable instructions that can be checked programmatically.

Reference: Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models."
arXiv preprint arXiv:2311.07911 (2023).

Dataset: https://huggingface.co/datasets/google/IFEval
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists."
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/ifeval", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the dataset to.")
    parser.add_argument(
        "--add_instruction_prompt",
        action="store_true",
        help="Whether to add instruction following guidance to the prompt.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "google/IFEval"

    # Load dataset
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # IFEval only has a train split, we'll use it as test for evaluation
    # For GRPO training, you may want to create synthetic examples or use a subset
    test_dataset = dataset["train"]

    # Optional: Add instruction following guidance
    instruction_following_prompt = (
        "Please follow all the instructions in the prompt carefully and precisely." if args.add_instruction_prompt else ""
    )

    def process_fn(example, idx):
        """Process each example into the verl format.

        The IFEval dataset has the following fields:
        - prompt: The input prompt with instructions
        - instruction_id_list: List of instruction IDs to verify
        - kwargs: List of kwargs for each instruction
        - key: Unique identifier for the example
        """
        prompt_text = example["prompt"]

        # Add optional instruction following guidance
        if instruction_following_prompt:
            prompt_text = instruction_following_prompt + "\n\n" + prompt_text

        # Extract instruction verification info
        instruction_id_list = example.get("instruction_id_list", [])
        kwargs_list = example.get("kwargs", [])

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "ability": "instruction_following",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs_list,
                },
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "key": example.get("key", f"ifeval_{idx}"),
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs_list,
                "prompt": example["prompt"],  # Store original prompt
            },
        }
        return data

    # Process dataset
    test_dataset = test_dataset.map(function=process_fn, with_indices=True)

    # For GRPO training, you may want to create a smaller train subset
    # Here we create a simple 80/20 split for demonstration
    split_dataset = test_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Save datasets
    hdfs_dir = args.hdfs_dir
    local_save_dir = os.path.expanduser(args.local_save_dir)

    # Create directory if it doesn't exist
    os.makedirs(local_save_dir, exist_ok=True)

    # Save train and test sets
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    eval_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Dataset saved to {local_save_dir}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(eval_dataset)}")

    # Optional: Copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
        print(f"Dataset copied to HDFS: {hdfs_dir}")
