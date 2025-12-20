# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The verl Authors
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
Data preprocessing script for Dolci-Think-RL and similar multi-task datasets.

This script processes datasets for Cascaded RL training following formats similar to:
- AllenAI Dolci-Think-RL: https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B
- NVIDIA Nemotron-Cascade datasets: https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-Math

The script organizes data by domain (math, code, instruction-following, etc.) and
prepares it for the Cascaded RL training pipeline.

Usage:
    python preprocess_dolci_dataset.py \
        --input_path allenai/Dolci-Think-RL-32B \
        --output_dir ./processed_data \
        --split_by_domain

Output format (parquet):
    - data_source: str - Task/domain identifier
    - prompt: list[dict] - Chat format messages
    - ability: str - Task ability category
    - reward_model: dict - Contains style and ground_truth for verification
    - extra_info: dict - Additional metadata
"""

import os
import argparse
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm


def normalize_domain(raw_domain: str) -> str:
    """Normalize domain/ability names to standard categories."""
    raw_lower = raw_domain.lower().strip()

    # Math-related
    if any(x in raw_lower for x in ["math", "gsm", "aime", "olympiad", "geometry", "algebra", "calculus", "competition_math"]):
        return "math"

    # Code-related
    if any(x in raw_lower for x in ["code", "programming", "python", "java", "coding", "livecodebench", "humaneval", "mbpp"]):
        return "code"

    # Instruction following
    if any(x in raw_lower for x in ["instruction", "ifeval", "following", "constraint"]):
        return "instruction_following"

    # General/Chat
    if any(x in raw_lower for x in ["chat", "general", "conversation", "helpfulness"]):
        return "general"

    # SWE
    if any(x in raw_lower for x in ["swe", "software", "github", "bug", "patch"]):
        return "swe"

    # Science/Reasoning
    if any(x in raw_lower for x in ["science", "reasoning", "logic", "gpqa", "mmlu"]):
        return "reasoning"

    return raw_lower


def extract_ground_truth(example: Dict[str, Any]) -> Optional[Any]:
    """Extract ground truth answer from various possible fields."""
    # Try common ground truth field names
    gt_fields = [
        "answer", "ground_truth", "solution", "correct_answer",
        "target", "label", "expected_output", "reference_answer",
        "gold_answer", "canonical_solution"
    ]

    for field in gt_fields:
        if field in example and example[field] is not None:
            return example[field]

    # Check nested structures
    if "metadata" in example and isinstance(example["metadata"], dict):
        for field in gt_fields:
            if field in example["metadata"]:
                return example["metadata"][field]

    return None


def format_prompt_as_chat(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert various prompt formats to chat format."""

    # Already in chat format
    if "messages" in example:
        messages = example["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict) and "role" in messages[0]:
                return messages

    # Single prompt field
    prompt_fields = ["prompt", "question", "input", "query", "problem", "instruction"]

    for field in prompt_fields:
        if field in example and example[field]:
            prompt = example[field]
            if isinstance(prompt, str):
                return [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                # Could be multi-turn
                messages = []
                for i, msg in enumerate(prompt):
                    if isinstance(msg, str):
                        role = "user" if i % 2 == 0 else "assistant"
                        messages.append({"role": role, "content": msg})
                    elif isinstance(msg, dict):
                        messages.append(msg)
                return messages

    # Fallback: combine instruction + input
    if "instruction" in example and "input" in example:
        content = example["instruction"]
        if example["input"]:
            content = f"{content}\n\nInput: {example['input']}"
        return [{"role": "user", "content": content}]

    return []


def get_reward_style(domain: str) -> str:
    """Determine reward style based on domain."""
    if domain in ["math", "code", "instruction_following"]:
        return "rule"  # Rule-based verification
    elif domain in ["general", "chat"]:
        return "model"  # Reward model based
    else:
        return "rule"  # Default to rule-based


def process_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Process a single example into the standard format."""

    # Extract domain/ability
    raw_domain = example.get("domain", example.get("ability", example.get("task", example.get("category", "general"))))
    domain = normalize_domain(str(raw_domain))

    # Create data source identifier
    source = example.get("source", example.get("dataset", example.get("data_source", f"{domain}_data")))
    data_source = f"{source}_{domain}" if source != f"{domain}_data" else source

    # Format prompt
    prompt = format_prompt_as_chat(example)

    # Extract ground truth
    ground_truth = extract_ground_truth(example)

    # Determine reward style
    reward_style = get_reward_style(domain)

    # Build reward model config
    reward_model = {
        "style": reward_style,
        "ground_truth": ground_truth,
    }

    # Extra info
    extra_info = {
        "split": example.get("split", "train"),
        "index": idx,
        "original_domain": str(raw_domain),
    }

    # Add test cases for code if available
    if domain == "code":
        if "test_cases" in example:
            reward_model["test_cases"] = example["test_cases"]
        elif "tests" in example:
            reward_model["test_cases"] = example["tests"]
        elif "public_tests" in example:
            reward_model["test_cases"] = example["public_tests"]

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": domain,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def load_and_process_dataset(
    input_path: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Load dataset from HuggingFace and process it."""

    print(f"Loading dataset from: {input_path}")

    # Try to load from HuggingFace Hub
    try:
        if os.path.isdir(input_path):
            # Local directory
            dataset = load_dataset("parquet", data_dir=input_path, split=split)
        elif input_path.endswith(".parquet"):
            # Single parquet file
            dataset = load_dataset("parquet", data_files=input_path, split=split)
        elif input_path.endswith(".jsonl") or input_path.endswith(".json"):
            # JSON file
            dataset = load_dataset("json", data_files=input_path, split=split)
        else:
            # HuggingFace Hub dataset
            try:
                dataset = load_dataset(input_path, split=split)
            except:
                # Try with trust_remote_code
                dataset = load_dataset(input_path, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    print(f"Loaded {len(dataset)} examples")

    if max_samples and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} examples")

    # Process examples
    processed_data = []
    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        processed = process_example(example, idx)
        if processed["prompt"]:  # Only include if we have a valid prompt
            processed_data.append(processed)

    print(f"Processed {len(processed_data)} valid examples")

    return pd.DataFrame(processed_data)


def split_by_domain(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split dataframe by domain/ability."""
    domain_dfs = {}

    for domain in df["ability"].unique():
        domain_df = df[df["ability"] == domain].reset_index(drop=True)
        domain_dfs[domain] = domain_df
        print(f"  {domain}: {len(domain_df)} examples")

    return domain_dfs


def save_to_parquet(df: pd.DataFrame, output_path: str):
    """Save dataframe to parquet format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for Cascaded RL training")

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input dataset (HuggingFace Hub ID or local path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Output directory for processed parquet files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--split_by_domain",
        action="store_true",
        help="Split output by domain into separate files",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dolci",
        help="Name prefix for output files",
    )

    args = parser.parse_args()

    # Process the dataset
    df = load_and_process_dataset(
        input_path=args.input_path,
        split=args.split,
        max_samples=args.max_samples,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.split_by_domain:
        # Split and save by domain
        print("\nSplitting by domain...")
        domain_dfs = split_by_domain(df)

        for domain, domain_df in domain_dfs.items():
            output_path = os.path.join(
                args.output_dir,
                f"{args.dataset_name}_{domain}_{args.split}.parquet"
            )
            save_to_parquet(domain_df, output_path)
    else:
        # Save as single file
        output_path = os.path.join(
            args.output_dir,
            f"{args.dataset_name}_{args.split}.parquet"
        )
        save_to_parquet(df, output_path)

    # Print summary
    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)
    print(f"Total examples: {len(df)}")
    print(f"Domain distribution:")
    for domain, count in df["ability"].value_counts().items():
        print(f"  {domain}: {count}")
    print("="*50)


if __name__ == "__main__":
    main()
