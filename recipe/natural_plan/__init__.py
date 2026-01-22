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
Natural Plan Environment for RLVR

This module implements a reinforcement learning environment based on the Natural Plan
benchmark (https://github.com/google-deepmind/natural-plan). It provides:

1. Programmatic data synthesis for three planning tasks:
   - Calendar Scheduling: Find common meeting times among participants
   - Meeting Planning: Schedule meetings with people at various locations
   - Trip Planning: Plan optimal trip sequences with durations

2. Verifiable reward functions that deterministically check plan correctness

3. Integration with verl's RLVR training pipeline

Key features:
- No test data leakage: All data is synthesized programmatically
- Configurable difficulty levels (number of people, days, constraints)
- Deterministic verification for reward computation
"""

from recipe.natural_plan.tasks.calendar_scheduling import CalendarSchedulingTask
from recipe.natural_plan.tasks.meeting_planning import MeetingPlanningTask
from recipe.natural_plan.tasks.trip_planning import TripPlanningTask
from recipe.natural_plan.reward_fn import compute_score

__all__ = [
    "CalendarSchedulingTask",
    "MeetingPlanningTask",
    "TripPlanningTask",
    "compute_score",
]
