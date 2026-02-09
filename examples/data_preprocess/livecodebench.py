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
Preprocess LiveCodeBench dataset for use with verl PPO training.

LiveCodeBench (https://github.com/LiveCodeBench/LiveCodeBench) provides competitive
programming problems collected from LeetCode, AtCoder, and CodeForces.

This script loads the dataset from HuggingFace, constructs prompts, compresses test
cases, and outputs a parquet file suitable for verl training pipelines.

Usage:
    python examples/data_preprocess/livecodebench.py \
        --local_dir ~/data/livecodebench \
        --start_date 2024-08-01 \
        --end_date 2025-01-01

    # Use all available problems (no date filter):
    python examples/data_preprocess/livecodebench.py \
        --local_dir ~/data/livecodebench \
        --no_date_filter
"""

import argparse
import base64
import json
import os
import pickle
import zlib

from datasets import load_dataset


def process_livecodebench(example):
    """Process a single LiveCodeBench example into prompt + compressed test cases.

    Constructs the query prompt following the LiveCodeBench format and combines
    public + private test cases into a compressed ground truth payload.
    """
    # Construct query prompt following LiveCodeBench format
    # Reference: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    query_prompt = (
        "You will be given a question (problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests.\n\n"
        f"Question: {example['question_content']}\n\n"
    )

    if example["starter_code"]:
        query_prompt += (
            "You will use the following starter code to write the solution to the problem and enclose your "
            f"code within delimiters.\n```python\n{example['starter_code']}\n```"
        )
    else:
        query_prompt += (
            "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
            "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
            "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```"
        )

    # Parse test cases
    public_test_cases = json.loads(example["public_test_cases"])
    try:
        private_test_cases = json.loads(example["private_test_cases"])
    except Exception:
        private_test_cases = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8"))))
        )
    full_test_cases = public_test_cases + private_test_cases

    metadata = json.loads(example["metadata"])
    test_cases = {
        "inputs": [t["input"] for t in full_test_cases],
        "outputs": [t["output"] for t in full_test_cases],
        "fn_name": metadata.get("func_name", None),
    }
    compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")

    return query_prompt, compressed


def build_dataset(data_source, start_date=None, end_date=None):
    """Build the LiveCodeBench dataset with optional date filtering.

    Args:
        data_source: HuggingFace dataset identifier.
        start_date: Start date filter (inclusive), e.g. "2024-08-01".
        end_date: End date filter (exclusive), e.g. "2025-01-01".

    Returns:
        A HuggingFace Dataset in verl's expected format.
    """
    print(f"Loading {data_source} from HuggingFace...", flush=True)
    dataset = load_dataset(data_source, split="test")
    print(f"Loaded {len(dataset)} problems.", flush=True)

    if start_date and end_date:
        # Convert to ISO format for string comparison
        start_iso = f"{start_date}T00:00:00"
        end_iso = f"{end_date}T00:00:00"
        dataset = dataset.filter(lambda x: start_iso <= x["contest_date"] < end_iso)
        print(f"After date filtering [{start_date}, {end_date}): {len(dataset)} problems.", flush=True)

    def map_fn(example, idx):
        question, solution = process_livecodebench(example)
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "Code",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "test",
                "index": idx,
                "platform": example.get("platform", ""),
                "question_id": example.get("question_id", ""),
                "contest_id": example.get("contest_id", ""),
                "contest_date": example.get("contest_date", ""),
            },
        }

    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names, num_proc=8)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LiveCodeBench for verl training")
    parser.add_argument(
        "--local_dir",
        default="~/data/livecodebench",
        help="Local directory to save the processed parquet file.",
    )
    parser.add_argument(
        "--data_source",
        default="livecodebench/code_generation_lite",
        choices=["livecodebench/code_generation_lite", "livecodebench/code_generation"],
        help="HuggingFace dataset to use. 'lite' version has pre-decompressed test cases.",
    )
    parser.add_argument(
        "--start_date",
        default=None,
        help="Start date filter (inclusive), e.g. '2024-08-01'. Use with --end_date.",
    )
    parser.add_argument(
        "--end_date",
        default=None,
        help="End date filter (exclusive), e.g. '2025-01-01'. Use with --start_date.",
    )
    parser.add_argument(
        "--no_date_filter",
        action="store_true",
        help="Disable date filtering and use all available problems.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy output to.",
    )

    args = parser.parse_args()
    local_dir = os.path.expanduser(args.local_dir)

    if args.no_date_filter:
        start_date, end_date = None, None
    else:
        start_date = args.start_date
        end_date = args.end_date

    dataset = build_dataset(args.data_source, start_date, end_date)
    print(f"Final dataset size: {len(dataset)}", flush=True)

    os.makedirs(local_dir, exist_ok=True)
    output_path = os.path.join(local_dir, "test.parquet")
    dataset.to_parquet(output_path)
    print(f"Saved to {output_path}", flush=True)

    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}", flush=True)
