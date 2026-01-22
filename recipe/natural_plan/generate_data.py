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
Data Generation Script for Natural Plan Environment

This script generates synthetic training and test data for the Natural Plan
RLVR environment. The data is generated programmatically to avoid test data
leakage - all instances are freshly synthesized.

Output format is compatible with verl's RLHFDataset (parquet files).

Usage:
    python generate_data.py --output_dir ~/data/natural_plan --num_train 10000 --num_test 1000
    python generate_data.py --task calendar --num_train 5000 --seed 42
"""

import argparse
import json
import os
from typing import Optional

import datasets

from recipe.natural_plan.tasks.calendar_scheduling import CalendarSchedulingTask
from recipe.natural_plan.tasks.meeting_planning import MeetingPlanningTask
from recipe.natural_plan.tasks.trip_planning import TripPlanningTask


def generate_calendar_data(
    num_samples: int,
    seed: int,
    split: str,
    difficulty: str = "mixed",
) -> list[dict]:
    """
    Generate calendar scheduling instances.

    Args:
        num_samples: Number of instances to generate
        seed: Random seed
        split: "train" or "test"
        difficulty: "easy", "medium", "hard", or "mixed"

    Returns:
        List of dicts in verl data format
    """
    task = CalendarSchedulingTask(seed=seed)
    data = []

    # Difficulty settings
    difficulty_configs = {
        "easy": {"min_people": 2, "max_people": 3, "min_days": 1, "max_days": 1},
        "medium": {"min_people": 3, "max_people": 4, "min_days": 1, "max_days": 3},
        "hard": {"min_people": 4, "max_people": 6, "min_days": 3, "max_days": 5},
    }

    for i in range(num_samples):
        # Select difficulty
        if difficulty == "mixed":
            diff = ["easy", "medium", "hard"][i % 3]
        else:
            diff = difficulty

        config = difficulty_configs[diff]
        task.min_people = config["min_people"]
        task.max_people = config["max_people"]
        task.min_days = config["min_days"]
        task.max_days = config["max_days"]

        instance = task.generate()
        prompt = instance.get_prompt()

        data.append({
            "data_source": "natural_plan/calendar_scheduling",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "planning",
            "reward_model": {
                "style": "rule",
                "ground_truth": instance.to_dict()["solution"]
            },
            "extra_info": {
                "split": split,
                "index": i,
                "task": "calendar_scheduling",
                "difficulty": diff,
                "num_people": len(instance.participants),
                "num_days": instance.num_days,
                "duration": instance.meeting_duration,
                "participants": instance.to_dict()["participants"],
            }
        })

    return data


def generate_meeting_data(
    num_samples: int,
    seed: int,
    split: str,
    difficulty: str = "mixed",
) -> list[dict]:
    """
    Generate meeting planning instances.

    Args:
        num_samples: Number of instances to generate
        seed: Random seed
        split: "train" or "test"
        difficulty: "easy", "medium", "hard", or "mixed"

    Returns:
        List of dicts in verl data format
    """
    task = MeetingPlanningTask(seed=seed)
    data = []

    difficulty_configs = {
        "easy": {"min_people": 2, "max_people": 3},
        "medium": {"min_people": 3, "max_people": 5},
        "hard": {"min_people": 5, "max_people": 8},
    }

    for i in range(num_samples):
        if difficulty == "mixed":
            diff = ["easy", "medium", "hard"][i % 3]
        else:
            diff = difficulty

        config = difficulty_configs[diff]
        task.min_people = config["min_people"]
        task.max_people = config["max_people"]

        instance = task.generate()
        prompt = instance.get_prompt()
        instance_dict = instance.to_dict()

        data.append({
            "data_source": "natural_plan/meeting_planning",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "planning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"solution_score": instance.solution_score}
            },
            "extra_info": {
                "split": split,
                "index": i,
                "task": "meeting_planning",
                "difficulty": diff,
                "num_people": len(instance.people),
                "optimal_meetings": instance.solution_score,
                "start_location": instance_dict["start_location"],
                "start_time": instance_dict["start_time"],
                "people": instance_dict["people"],
                "distance_matrix": instance_dict["distance_matrix"],
            }
        })

    return data


def generate_trip_data(
    num_samples: int,
    seed: int,
    split: str,
    difficulty: str = "mixed",
) -> list[dict]:
    """
    Generate trip planning instances.

    Args:
        num_samples: Number of instances to generate
        seed: Random seed
        split: "train" or "test"
        difficulty: "easy", "medium", "hard", or "mixed"

    Returns:
        List of dicts in verl data format
    """
    task = TripPlanningTask(seed=seed)
    data = []

    difficulty_configs = {
        "easy": {"min_cities": 3, "max_cities": 4, "min_constraints": 0, "max_constraints": 1},
        "medium": {"min_cities": 4, "max_cities": 5, "min_constraints": 1, "max_constraints": 2},
        "hard": {"min_cities": 5, "max_cities": 7, "min_constraints": 2, "max_constraints": 4},
    }

    for i in range(num_samples):
        if difficulty == "mixed":
            diff = ["easy", "medium", "hard"][i % 3]
        else:
            diff = difficulty

        config = difficulty_configs[diff]
        task.min_cities = config["min_cities"]
        task.max_cities = config["max_cities"]
        task.min_constraints = config["min_constraints"]
        task.max_constraints = config["max_constraints"]

        instance = task.generate()
        prompt = instance.get_prompt()
        instance_dict = instance.to_dict()

        data.append({
            "data_source": "natural_plan/trip_planning",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "planning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "solution_cities": instance_dict["solution_cities"],
                    "solution_durations": instance_dict["solution_durations"]
                }
            },
            "extra_info": {
                "split": split,
                "index": i,
                "task": "trip_planning",
                "difficulty": diff,
                "num_cities": len(instance.cities),
                "total_days": instance.total_days,
                "region": instance.region,
                "cities": instance_dict["cities"],
                "durations": instance_dict["durations"],
                "constraints": instance_dict["constraints"],
            }
        })

    return data


def generate_mixed_data(
    num_samples: int,
    seed: int,
    split: str,
    task_weights: Optional[dict] = None,
) -> list[dict]:
    """
    Generate mixed data from all three tasks.

    Args:
        num_samples: Total number of instances
        seed: Random seed
        split: "train" or "test"
        task_weights: Dict mapping task name to weight (defaults to equal)

    Returns:
        List of dicts in verl data format
    """
    if task_weights is None:
        task_weights = {"calendar": 1, "meeting": 1, "trip": 1}

    total_weight = sum(task_weights.values())
    task_counts = {
        task: int(num_samples * weight / total_weight)
        for task, weight in task_weights.items()
    }

    # Adjust for rounding
    remaining = num_samples - sum(task_counts.values())
    for task in task_counts:
        if remaining <= 0:
            break
        task_counts[task] += 1
        remaining -= 1

    data = []

    if task_counts.get("calendar", 0) > 0:
        data.extend(generate_calendar_data(
            task_counts["calendar"], seed, split
        ))

    if task_counts.get("meeting", 0) > 0:
        data.extend(generate_meeting_data(
            task_counts["meeting"], seed + 1000, split
        ))

    if task_counts.get("trip", 0) > 0:
        data.extend(generate_trip_data(
            task_counts["trip"], seed + 2000, split
        ))

    # Shuffle the data
    import random
    rng = random.Random(seed)
    rng.shuffle(data)

    # Re-index
    for i, item in enumerate(data):
        item["extra_info"]["index"] = i

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Natural Plan training data for RLVR"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/natural_plan",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["calendar", "meeting", "trip", "all"],
        default="all",
        help="Task to generate data for"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=10000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=1000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "mixed"],
        default="mixed",
        help="Difficulty level"
    )

    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating Natural Plan data...")
    print(f"  Task: {args.task}")
    print(f"  Train samples: {args.num_train}")
    print(f"  Test samples: {args.num_test}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Output: {output_dir}")
    print()

    # Generate data based on task
    if args.task == "calendar":
        train_data = generate_calendar_data(
            args.num_train, args.seed, "train", args.difficulty
        )
        test_data = generate_calendar_data(
            args.num_test, args.seed + 10000, "test", args.difficulty
        )
    elif args.task == "meeting":
        train_data = generate_meeting_data(
            args.num_train, args.seed, "train", args.difficulty
        )
        test_data = generate_meeting_data(
            args.num_test, args.seed + 10000, "test", args.difficulty
        )
    elif args.task == "trip":
        train_data = generate_trip_data(
            args.num_train, args.seed, "train", args.difficulty
        )
        test_data = generate_trip_data(
            args.num_test, args.seed + 10000, "test", args.difficulty
        )
    else:  # all
        train_data = generate_mixed_data(
            args.num_train, args.seed, "train"
        )
        test_data = generate_mixed_data(
            args.num_test, args.seed + 10000, "test"
        )

    # Convert to HuggingFace dataset and save
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(test_data)} test samples to {test_path}")

    # Print sample
    print("\n--- Sample Training Instance ---")
    sample = train_data[0]
    print(f"Task: {sample['extra_info']['task']}")
    print(f"Difficulty: {sample['extra_info']['difficulty']}")
    print(f"\nPrompt:\n{sample['prompt'][0]['content'][:500]}...")
    print(f"\nGround Truth: {json.dumps(sample['reward_model']['ground_truth'], indent=2)}")


if __name__ == "__main__":
    main()
