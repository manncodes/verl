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
Includes robust normalization for compressed code/IO data blobs.
"""

import argparse
import json
import os
import random
import base64
import zlib
import gzip
import pickle
import ast
import re
from typing import Optional, Set, Any

import numpy as np
import datasets
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs

# Valid abilities that can be filtered
VALID_ABILITIES = frozenset({"math", "code", "instruction_following", "chat", "reasoning"})

# =============================================================================
# Normalization Logic (Robust v3)
# =============================================================================

def recursive_normalize_fixed(val: Any) -> Any:
    """
    Robustly handles:
    1. Numpy Arrays
    2. Nested lists/tuples/sequences
    3. JSON Strings
    4. Compressed Blobs (eJy...) - Handles Strings, Bytes, and various compression formats
    """
    # STEP 0: Handle Numpy arrays and scalars
    if isinstance(val, np.ndarray):
        val = val.tolist()
    elif isinstance(val, (np.str_, np.bytes_)):
        val = str(val) if isinstance(val, np.str_) else bytes(val)

    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []

    # Helper: Robust Decompress - tries multiple decompression methods
    def robust_decompress(data_bytes):
        # Try standard zlib (wbits=15)
        try:
            return zlib.decompress(data_bytes)
        except:
            pass
        # Try auto-detect (wbits=0)
        try:
            return zlib.decompress(data_bytes, wbits=0)
        except:
            pass
        # Try raw deflate (wbits=-15)
        try:
            return zlib.decompress(data_bytes, wbits=-15)
        except:
            pass
        # Try zlib with max window (wbits=31 for gzip)
        try:
            return zlib.decompress(data_bytes, wbits=31)
        except:
            pass
        # Try gzip
        try:
            return gzip.decompress(data_bytes)
        except:
            pass
        raise ValueError("All decompression attempts failed")

    # Helper: Decode a potential blob string
    def try_decode_blob(s):
        if s is None:
            return None

        # Normalize to string
        if isinstance(s, (bytes, np.bytes_)):
            try:
                s = s.decode('utf-8', errors='ignore')
            except:
                return None

        # Handle numpy string types
        if isinstance(s, np.str_):
            s = str(s)

        if not isinstance(s, str):
            return None

        # Clean the string - remove ALL whitespace (including internal newlines)
        s = s.strip()

        # Handle python string representation of bytes "b'...'"
        if s.startswith("b'") and s.endswith("'"):
            s = s[2:-1]
        elif s.startswith('b"') and s.endswith('"'):
            s = s[2:-1]

        # Check signature (eJ = 78 9C = Default Zlib, eJw/eJx/eJy/eJz variants)
        if s.startswith('eJ'):
            try:
                # Remove any internal whitespace from base64 string
                s_clean = re.sub(r'\s+', '', s)

                # Fix Base64 Padding
                pad = len(s_clean) % 4
                if pad:
                    s_clean += '=' * (4 - pad)

                # Try standard Base64 decode
                try:
                    compressed = base64.b64decode(s_clean)
                except:
                    # Try URL-safe Base64 as fallback
                    compressed = base64.urlsafe_b64decode(s_clean)

                # Robust Decompress
                decoded = robust_decompress(compressed)

                # Try Unpickle
                try:
                    return pickle.loads(decoded)
                except:
                    # Try JSON
                    try:
                        return json.loads(decoded)
                    except:
                        return decoded.decode('utf-8', errors='ignore')
            except Exception:
                return None
        return None

    # Helper: Parse JSON string
    def try_decode_json(s):
        if isinstance(s, (bytes, np.bytes_)):
            try:
                s = s.decode('utf-8')
            except:
                return None
        if isinstance(s, np.str_):
            s = str(s)
        if not isinstance(s, str):
            return None

        s = s.strip()
        if s.startswith('[') or s.startswith('{'):
            try:
                return json.loads(s)
            except:
                return None
        return None

    # Helper: Check if object is a sequence (list, tuple, or array-like)
    def is_sequence(obj):
        if isinstance(obj, (str, bytes, dict)):
            return False
        return isinstance(obj, (list, tuple)) or hasattr(obj, '__iter__') and hasattr(obj, '__getitem__')

    # Helper: Convert sequence to list
    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return list(obj)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, '__iter__'):
            return list(obj)
        return obj

    # STEP 1: Check if the input itself is a blob (String or Bytes)
    blob_res = try_decode_blob(val)
    if blob_res is not None:
        return blob_res

    # STEP 2: Outer Parsing (if it's a string representation of a list)
    data = val
    if isinstance(data, str):
        data = data.strip()
        try:
            data = json.loads(data)
        except:
            try:
                data = ast.literal_eval(data)
            except:
                pass

    # STEP 3: Drill Down Loop
    max_iterations = 100  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Handle numpy arrays at each iteration
        if isinstance(data, np.ndarray):
            data = data.tolist()

        # Check if data is a sequence with elements
        if is_sequence(data):
            data_list = to_list(data)
            if len(data_list) == 0:
                return data_list

            first = data_list[0]

            # Handle numpy types in first element
            if isinstance(first, np.ndarray):
                first = first.tolist()
            elif isinstance(first, np.str_):
                first = str(first)

            # Scenario A: First element is a String or Bytes
            if isinstance(first, (str, bytes)):
                # Convert bytes to string for checking
                first_str = first.decode('utf-8', errors='ignore') if isinstance(first, bytes) else first

                # Assertions -> DONE
                if first_str.strip().startswith("assert"):
                    return data_list

                # JSON string -> Unpack and loop again
                res = try_decode_json(first)
                if res is not None:
                    data = res
                    continue

                # Blob -> Unpack and DONE
                res = try_decode_blob(first)
                if res is not None:
                    return res

                # Generic strings (chat/reasoning) -> DONE
                return data_list

            # Scenario B: First element is a sequence -> Drill down
            elif is_sequence(first):
                data = to_list(first)
                continue

            # Scenario C: First element is a Dict (IO Objects) -> DONE
            elif isinstance(first, dict):
                return data_list

            else:
                # Unknown structure, return as list
                return data_list
        else:
            return data

    # If we hit max iterations, return what we have
    return data

# =============================================================================
# Standard Utils
# =============================================================================

def get_ability_from_source(dataset_source: str) -> str:
    """Map dataset_source to ability category."""
    source_to_ability = {
        "math": "math",
        "code_stdio": "code",
        "code": "code",
        "ifeval": "instruction_following",
        "general-quality": "chat",
        "general-quality_ref": "chat",
    }
    return source_to_ability.get(dataset_source, "reasoning")

def get_reward_style(dataset_source: str) -> str:
    if dataset_source in ("math", "code", "code_stdio"):
        return "rule"
    return "model"

def validate_abilities(abilities: Optional[list], arg_name: str) -> Optional[Set[str]]:
    if abilities is None:
        return None
    abilities_set = set(abilities)
    invalid = abilities_set - VALID_ABILITIES
    if invalid:
        raise ValueError(f"Invalid abilities: {sorted(invalid)}")
    return abilities_set

def create_ability_filter(include_abilities: Optional[Set[str]], exclude_abilities: Optional[Set[str]]):
    def filter_fn(example) -> bool:
        dataset_source = example.get("dataset", [None])[0]
        ability = get_ability_from_source(dataset_source)
        if include_abilities is not None and ability not in include_abilities:
            return False
        if exclude_abilities is not None and ability in exclude_abilities:
            return False
        return True
    return filter_fn

def split_uniform_test_from_dataset(dataset: datasets.Dataset, samples_per_ability: Optional[int] = 50, seed: int = 42):
    if len(dataset) == 0: return dataset, dataset

    ability_to_indices: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        example = dataset[idx]
        dataset_source = example.get("dataset", [None])[0]
        ability = get_ability_from_source(dataset_source)
        if ability not in ability_to_indices: ability_to_indices[ability] = []
        ability_to_indices[ability].append(idx)

    min_count = min(len(indices) for indices in ability_to_indices.values())
    if samples_per_ability is None: samples_per_ability = min_count
    else: samples_per_ability = min(samples_per_ability, min_count)

    rng = random.Random(seed)
    test_indices_set: set[int] = set()
    for ability in sorted(ability_to_indices.keys()):
        indices = ability_to_indices[ability]
        sampled = rng.sample(indices, samples_per_ability)
        test_indices_set.update(sampled)

    all_indices = set(range(len(dataset)))
    train_indices = sorted(all_indices - test_indices_set)
    test_indices = sorted(test_indices_set)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return dataset.select(train_indices), dataset.select(test_indices)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess allenai/Dolci-Think-RL dataset")
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/dolci_think_rl")
    parser.add_argument("--test_samples_per_ability", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_null_ground_truth", action="store_true")
    parser.add_argument("--exclude_abilities", nargs="+", default=None)
    parser.add_argument("--include_abilities", nargs="+", default=None)

    args = parser.parse_args()
    include_abilities = validate_abilities(args.include_abilities, "include_abilities")
    exclude_abilities = validate_abilities(args.exclude_abilities, "exclude_abilities")

    data_source = "allenai/Dolci-Think-RL"
    print(f"Loading {data_source}...", flush=True)

    if args.local_dataset_path:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    if "train" in dataset:
        full_dataset = dataset["train"].shuffle(42)
    else:
        full_dataset = datasets.concatenate_datasets(list(dataset.values()))

    # Filter Nulls
    if args.filter_null_ground_truth:
        full_dataset = full_dataset.filter(lambda x: x.get("ground_truth") is not None and len(x.get("ground_truth", [])) > 0)

    # Filter Abilities
    if include_abilities or exclude_abilities:
        ability_filter = create_ability_filter(include_abilities, exclude_abilities)
        full_dataset = full_dataset.filter(ability_filter)

    train_dataset, test_dataset = split_uniform_test_from_dataset(
        full_dataset, samples_per_ability=args.test_samples_per_ability, seed=args.seed
    )
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}", flush=True)

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_text = example.get("prompt", "")

            # --- NORMALIZATION ---
            raw_gt = example.get("ground_truth")
            clean_gt = recursive_normalize_fixed(raw_gt)

            # --- FIX: SERIALIZATION ---
            # Serialize to JSON string to ensure Parquet schema consistency
            try:
                ground_truth_str = json.dumps(clean_gt)
            except Exception:
                # Fallback for rare non-serializable edge cases
                ground_truth_str = str(clean_gt)

            dataset_source = example.get("dataset")[0]
            ability = get_ability_from_source(dataset_source)
            reward_style = get_reward_style(dataset_source)

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": ability,
                "reward_model": {
                    "style": reward_style,
                    "ground_truth": ground_truth_str, # Now strictly a string
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "original_prompt": prompt_text,
                    "dataset_source": dataset_source,
                    "custom_id": example.get("custom_id"),
                },
            }
            return data
        return process_fn

    print("Processing datasets...", flush=True)
    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir if args.local_dir else args.local_save_dir
    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Save examples
    if len(train_dataset) > 0:
        with open(os.path.join(local_dir, "train_example.json"), "w") as f:
            json.dump(train_dataset[0], f, indent=2, default=str)

    print("Done.")
