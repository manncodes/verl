# Natural Plan RLVR Environment

This recipe implements an RLVR (Reinforcement Learning with Verifiable Rewards) environment based on the [Natural Plan benchmark](https://github.com/google-deepmind/natural-plan). Unlike the original benchmark, this implementation synthesizes data programmatically, avoiding test data leakage and enabling unlimited training data generation.

## Overview

Natural Plan consists of three planning tasks that require models to reason about constraints and produce structured plans:

### 1. Calendar Scheduling
Find a common meeting time among multiple participants with different availability schedules.

**Example:**
```
Find a meeting time of 1 hour that works for all 3 participants.

The meeting can be scheduled on: Monday
Working hours are 9:00 - 17:00 each day.

Alice's schedule:
  - Busy: Monday, 9:00 - 10:30
  - Busy: Monday, 14:00 - 15:00

Bob's schedule:
  - Busy: Monday, 10:00 - 11:30

Charlie's schedule:
  - Busy: Monday, 9:30 - 10:30
  - Busy: Monday, 15:00 - 16:00

Format your answer as: The proposed time is [Day], [Start Time] - [End Time]
```

### 2. Meeting Planning
Schedule meetings with people at various locations, considering travel times and availability windows.

**Example:**
```
You are at Downtown at 9:00AM.
You want to meet with 3 people today.

People to meet:
- Alice: at Marina District, available 9:00AM-12:00PM, meeting duration: 30 minutes
- Bob: at Financial District, available 10:00AM-2:00PM, meeting duration: 45 minutes
- Charlie: at North Beach, available 11:00AM-3:00PM, meeting duration: 30 minutes

Travel times between locations:
- Downtown <-> Marina District: 15 minutes
- Downtown <-> Financial District: 10 minutes
- Marina District <-> North Beach: 20 minutes
...

Plan a schedule to meet as many people as possible.
```

### 3. Trip Planning
Plan an optimal trip itinerary visiting multiple cities with specified durations and constraints.

**Example:**
```
Plan a 10-day trip visiting 4 European cities:
Cities to visit: Paris, London, Rome, Amsterdam

Requirements:
- Spend exactly 3 days in Paris
- Spend exactly 3 days in London
- Spend exactly 2 days in Rome
- Spend exactly 2 days in Amsterdam

Additional constraints:
- Visit Paris before London
- Visit Rome and Amsterdam consecutively

Provide the optimal itinerary.
Format: List each city visit as 'Days X-Y: [City]' or 'Day X: [City]'
```

## Key Features

- **No Test Data Leakage**: All data is synthesized programmatically with configurable random seeds
- **Verifiable Rewards**: Deterministic verification of model responses
- **Configurable Difficulty**: Easy, medium, and hard difficulty levels
- **Scalable**: Generate unlimited training data
- **verl Integration**: Ready-to-use with verl's PPO and GRPO trainers

## Installation

The Natural Plan environment is included in the verl repository. No additional installation is required.

## Quick Start

### 1. Generate Training Data

```bash
# Generate mixed data from all three tasks
python -m recipe.natural_plan.generate_data \
    --output_dir ~/data/natural_plan \
    --num_train 10000 \
    --num_test 1000 \
    --seed 42

# Generate data for a specific task
python -m recipe.natural_plan.generate_data \
    --task calendar \
    --num_train 5000 \
    --difficulty medium
```

### 2. Run RLVR Training

**Using PPO:**
```bash
python -m verl.trainer.main_ppo \
    --config-path recipe/natural_plan/config \
    --config-name natural_plan_trainer \
    data.train_files=~/data/natural_plan/train.parquet \
    data.val_files=~/data/natural_plan/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
```

**Using GRPO (no critic):**
```bash
python -m verl.trainer.main_ppo \
    --config-path recipe/natural_plan/config \
    --config-name natural_plan_grpo \
    data.train_files=~/data/natural_plan/train.parquet \
    data.val_files=~/data/natural_plan/test.parquet
```

## Data Format

The generated data follows verl's standard format:

```python
{
    "data_source": "natural_plan/calendar_scheduling",  # Task identifier
    "prompt": [{"role": "user", "content": "..."}],     # Chat format
    "ability": "planning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {...}  # Task-specific ground truth
    },
    "extra_info": {
        "split": "train",
        "task": "calendar_scheduling",
        "difficulty": "medium",
        ...  # Full instance info for validation
    }
}
```

## Reward Function

The reward function (`recipe/natural_plan/reward_fn.py`) provides:

1. **compute_score(data_source, solution_str, ground_truth, extra_info)**: Main entry point returning float score (0.0 or 1.0)

2. **compute_score_with_details(...)**: Returns detailed dict with parsing status, validity checks, and error messages

The reward is deterministic and based on:
- **Calendar**: Exact match of day, start time, and end time
- **Meeting**: Ratio of valid meetings to optimal solution
- **Trip**: Exact match of city sequence and durations

## Customization

### Adjusting Difficulty

In `generate_data.py`, difficulty configurations are:

```python
# Calendar
"easy": {"min_people": 2, "max_people": 3, "min_days": 1, "max_days": 1}
"hard": {"min_people": 4, "max_people": 6, "min_days": 3, "max_days": 5}

# Meeting
"easy": {"min_people": 2, "max_people": 3}
"hard": {"min_people": 5, "max_people": 8}

# Trip
"easy": {"min_cities": 3, "max_cities": 4, "min_constraints": 0}
"hard": {"min_cities": 5, "max_cities": 7, "min_constraints": 2}
```

### Custom Task Weights

Generate data with custom task distribution:

```python
from recipe.natural_plan.generate_data import generate_mixed_data

data = generate_mixed_data(
    num_samples=10000,
    seed=42,
    split="train",
    task_weights={"calendar": 2, "meeting": 1, "trip": 1}  # 2x calendar
)
```

### Using Individual Task Synthesizers

```python
from recipe.natural_plan.tasks import CalendarSchedulingTask

task = CalendarSchedulingTask(
    seed=42,
    min_people=3,
    max_people=5,
    min_days=1,
    max_days=3
)

# Generate single instance
instance = task.generate(num_people=4, num_days=2)
print(instance.get_prompt())
print(instance.get_solution_str())

# Generate batch
instances = task.generate_batch(n=100)
```

## Verification Details

### Calendar Scheduling
- Parses response for pattern: `[Day], [HH:MM] - [HH:MM]`
- Validates meeting duration matches requirement
- Checks all participants are available (not busy during proposed slot)

### Meeting Planning
- Parses meeting entries: `Meet [Name] at [Location] from [Time] to [Time]`
- Validates travel times between consecutive meetings
- Checks each meeting is within person's availability window
- Counts valid meetings and compares to optimal solution

### Trip Planning
- Parses itinerary: `Days X-Y: [City]` or `Day X: [City]`
- Validates all cities are visited with correct durations
- Checks all constraints are satisfied (ordering, consecutiveness, etc.)

## File Structure

```
recipe/natural_plan/
├── __init__.py              # Package exports
├── README.md                # This file
├── reward_fn.py             # Unified reward function
├── generate_data.py         # Data generation script
├── config/
│   ├── natural_plan_trainer.yaml   # PPO config
│   └── natural_plan_grpo.yaml      # GRPO config
└── tasks/
    ├── __init__.py
    ├── calendar_scheduling.py   # Calendar task synthesizer
    ├── meeting_planning.py      # Meeting task synthesizer
    └── trip_planning.py         # Trip task synthesizer
```

## Citation

If you use this environment, please cite both the original Natural Plan paper and verl:

```bibtex
@article{zheng2024natural,
  title={NATURAL PLAN: Benchmarking LLMs on Natural Language Planning},
  author={Zheng, Huaixiu Steven and others},
  journal={arXiv preprint arXiv:2406.04520},
  year={2024}
}

@misc{verl2024,
  title={verl: A Flexible and Efficient Library for RL Training of LLMs},
  author={ByteDance Seed Team},
  year={2024},
  url={https://github.com/volcengine/verl}
}
```

## License

This code is licensed under the Apache 2.0 License. The Natural Plan benchmark concepts are from Google DeepMind's work (Apache 2.0 / CC-BY 4.0).
