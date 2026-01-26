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
Calendar Scheduling Task for Natural Plan Environment

This task involves finding a common meeting time among multiple participants
with different availability schedules across one or more days.

The task is fully synthesizable and verifiable:
- Calendars are generated programmatically with guaranteed solutions
- Verification checks exact match of day, start time, and end time
"""

import random
import re
from dataclasses import dataclass, field
from typing import Optional


# Common first names for generating participant names
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zach", "Emma", "Liam", "Sophia", "Mason", "Ava", "Lucas"
]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class TimeSlot:
    """Represents a time slot with start and end hours (in decimal format)."""
    day: str
    start_hour: float  # e.g., 9.5 = 9:30 AM
    end_hour: float    # e.g., 10.0 = 10:00 AM

    def overlaps(self, other: "TimeSlot") -> bool:
        """Check if this slot overlaps with another slot on the same day."""
        if self.day != other.day:
            return False
        return not (self.end_hour <= other.start_hour or other.end_hour <= self.start_hour)

    def contains(self, other: "TimeSlot") -> bool:
        """Check if this slot fully contains another slot."""
        if self.day != other.day:
            return False
        return self.start_hour <= other.start_hour and self.end_hour >= other.end_hour

    def to_time_str(self, hour: float) -> str:
        """Convert decimal hour to HH:MM format."""
        h = int(hour)
        m = int((hour - h) * 60)
        return f"{h}:{m:02d}"

    def __str__(self) -> str:
        return f"{self.day}, {self.to_time_str(self.start_hour)} - {self.to_time_str(self.end_hour)}"


@dataclass
class Participant:
    """Represents a meeting participant with availability windows."""
    name: str
    available_slots: list[TimeSlot] = field(default_factory=list)
    busy_slots: list[TimeSlot] = field(default_factory=list)

    def is_available(self, slot: TimeSlot) -> bool:
        """Check if participant is available during the given slot."""
        # Must be within an available window
        in_available = any(avail.contains(slot) for avail in self.available_slots)
        if not in_available:
            return False
        # Must not overlap with any busy slots
        return not any(busy.overlaps(slot) for busy in self.busy_slots)


@dataclass
class CalendarSchedulingInstance:
    """A single calendar scheduling problem instance."""
    participants: list[Participant]
    num_days: int
    meeting_duration: float  # in hours
    solution: TimeSlot

    def get_prompt(self) -> str:
        """Generate the natural language prompt for this instance."""
        lines = []
        lines.append(f"Find a meeting time of {self._duration_to_str(self.meeting_duration)} "
                    f"that works for all {len(self.participants)} participants.")
        lines.append("")

        days_str = ", ".join(DAYS_OF_WEEK[:self.num_days])
        lines.append(f"The meeting can be scheduled on: {days_str}")
        lines.append(f"Working hours are 9:00 - 17:00 each day.")
        lines.append("")

        for p in self.participants:
            lines.append(f"{p.name}'s schedule:")
            if p.busy_slots:
                for slot in p.busy_slots:
                    lines.append(f"  - Busy: {slot}")
            else:
                lines.append(f"  - No meetings scheduled")
            lines.append("")

        lines.append("Please find a time slot that works for everyone.")
        lines.append("Format your answer as: The proposed time is [Day], [Start Time] - [End Time]")

        return "\n".join(lines)

    def _duration_to_str(self, duration: float) -> str:
        """Convert duration in hours to human-readable string."""
        if duration == 0.5:
            return "30 minutes"
        elif duration == 1.0:
            return "1 hour"
        elif duration == 1.5:
            return "1 hour and 30 minutes"
        elif duration == 2.0:
            return "2 hours"
        else:
            hours = int(duration)
            mins = int((duration - hours) * 60)
            if mins == 0:
                return f"{hours} hours"
            return f"{hours} hours and {mins} minutes"

    def get_solution_str(self) -> str:
        """Get the solution in the expected format."""
        return f"The proposed time is {self.solution}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "num_people": len(self.participants),
            "num_days": self.num_days,
            "duration": self.meeting_duration,
            "participants": [
                {
                    "name": p.name,
                    "busy_slots": [
                        {"day": s.day, "start": s.start_hour, "end": s.end_hour}
                        for s in p.busy_slots
                    ]
                }
                for p in self.participants
            ],
            "solution": {
                "day": self.solution.day,
                "start": self.solution.start_hour,
                "end": self.solution.end_hour
            }
        }


class CalendarSchedulingTask:
    """
    Synthesizer for Calendar Scheduling problems.

    Generates instances where participants have various busy slots,
    and exactly one valid meeting time exists (or multiple, with one designated as solution).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        min_people: int = 2,
        max_people: int = 5,
        min_days: int = 1,
        max_days: int = 5,
        min_duration: float = 0.5,
        max_duration: float = 2.0,
        working_hours: tuple[float, float] = (9.0, 17.0),
        min_busy_slots_per_person: int = 1,
        max_busy_slots_per_person: int = 4,
    ):
        """
        Initialize the calendar scheduling task synthesizer.

        Args:
            seed: Random seed for reproducibility
            min_people: Minimum number of participants
            max_people: Maximum number of participants
            min_days: Minimum number of available days
            max_days: Maximum number of available days
            min_duration: Minimum meeting duration in hours
            max_duration: Maximum meeting duration in hours
            working_hours: Tuple of (start, end) working hours
            min_busy_slots_per_person: Minimum busy slots per participant
            max_busy_slots_per_person: Maximum busy slots per participant
        """
        self.rng = random.Random(seed)
        self.min_people = min_people
        self.max_people = max_people
        self.min_days = min_days
        self.max_days = max_days
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.working_hours = working_hours
        self.min_busy_slots = min_busy_slots_per_person
        self.max_busy_slots = max_busy_slots_per_person

    def _sample_duration(self) -> float:
        """Sample a meeting duration (in 30-min increments)."""
        options = [d for d in [0.5, 1.0, 1.5, 2.0]
                   if self.min_duration <= d <= self.max_duration]
        return self.rng.choice(options)

    def _sample_names(self, n: int) -> list[str]:
        """Sample n unique participant names."""
        return self.rng.sample(FIRST_NAMES, min(n, len(FIRST_NAMES)))

    def _generate_busy_slot(self, day: str, exclude_slot: TimeSlot) -> Optional[TimeSlot]:
        """Generate a random busy slot that doesn't overlap with the solution."""
        start_hour, end_hour = self.working_hours

        # Generate random slot duration (0.5 to 2 hours)
        duration = self.rng.choice([0.5, 1.0, 1.5, 2.0])

        # Try to find a non-overlapping slot
        for _ in range(20):  # Limited attempts
            slot_start = start_hour + self.rng.random() * (end_hour - start_hour - duration)
            slot_start = round(slot_start * 2) / 2  # Round to nearest 30 min
            slot_end = slot_start + duration

            if slot_end > end_hour:
                continue

            candidate = TimeSlot(day, slot_start, slot_end)

            # Make sure it doesn't overlap with the solution
            if not candidate.overlaps(exclude_slot):
                return candidate

        return None

    def generate(
        self,
        num_people: Optional[int] = None,
        num_days: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> CalendarSchedulingInstance:
        """
        Generate a single calendar scheduling instance.

        The generation ensures at least one valid solution exists by:
        1. First choosing a solution slot
        2. Generating busy slots that don't overlap with the solution

        Args:
            num_people: Number of participants (random if None)
            num_days: Number of available days (random if None)
            duration: Meeting duration in hours (random if None)

        Returns:
            A CalendarSchedulingInstance with guaranteed solution
        """
        # Determine parameters
        if num_people is None:
            num_people = self.rng.randint(self.min_people, self.max_people)
        if num_days is None:
            num_days = self.rng.randint(self.min_days, self.max_days)
        if duration is None:
            duration = self._sample_duration()

        # Get available days
        days = DAYS_OF_WEEK[:num_days]

        # Choose solution slot first (guarantees at least one valid answer)
        start_hour, end_hour = self.working_hours
        solution_day = self.rng.choice(days)
        max_start = end_hour - duration
        solution_start = start_hour + self.rng.random() * (max_start - start_hour)
        solution_start = round(solution_start * 2) / 2  # Round to nearest 30 min
        solution = TimeSlot(solution_day, solution_start, solution_start + duration)

        # Generate participants
        names = self._sample_names(num_people)
        participants = []

        for name in names:
            # Generate available slots (all working hours on all days)
            available_slots = [
                TimeSlot(day, start_hour, end_hour) for day in days
            ]

            # Generate busy slots
            num_busy = self.rng.randint(self.min_busy_slots, self.max_busy_slots)
            busy_slots = []

            for _ in range(num_busy):
                # Pick a random day
                busy_day = self.rng.choice(days)
                slot = self._generate_busy_slot(busy_day, solution)
                if slot is not None:
                    # Check it doesn't overlap with existing busy slots
                    if not any(slot.overlaps(existing) for existing in busy_slots):
                        busy_slots.append(slot)

            participants.append(Participant(
                name=name,
                available_slots=available_slots,
                busy_slots=busy_slots
            ))

        return CalendarSchedulingInstance(
            participants=participants,
            num_days=num_days,
            meeting_duration=duration,
            solution=solution
        )

    def generate_batch(
        self,
        n: int,
        num_people: Optional[int] = None,
        num_days: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> list[CalendarSchedulingInstance]:
        """Generate a batch of instances."""
        return [self.generate(num_people, num_days, duration) for _ in range(n)]


def parse_calendar_response(response: str) -> Optional[TimeSlot]:
    """
    Parse a calendar scheduling response to extract the proposed time.

    Expected format: "The proposed time is [Day], [Start Time] - [End Time]"
    Also handles variations like "Monday, 9:00 - 10:30"

    Returns:
        TimeSlot if parsed successfully, None otherwise
    """
    # Pattern to match day and time range
    pattern = r"(?:proposed time(?:\s+is)?:?\s*)?(\w+day),?\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})"

    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        day = match.group(1).capitalize()
        start_hour = int(match.group(2)) + int(match.group(3)) / 60
        end_hour = int(match.group(4)) + int(match.group(5)) / 60

        # Validate day
        if day not in DAYS_OF_WEEK:
            return None

        return TimeSlot(day, start_hour, end_hour)

    return None


def verify_calendar_solution(
    response: str,
    instance: CalendarSchedulingInstance,
    exact_match: bool = True,
) -> dict:
    """
    Verify if a response correctly solves the calendar scheduling problem.

    Args:
        response: The model's response string
        instance: The problem instance
        exact_match: If True, requires exact match with solution;
                     if False, accepts any valid slot

    Returns:
        Dict with score and detailed metrics
    """
    parsed = parse_calendar_response(response)

    if parsed is None:
        return {
            "score": 0.0,
            "parsed": False,
            "valid": False,
            "exact_match": False,
            "error": "Could not parse response"
        }

    # Check if the parsed slot has correct duration
    parsed_duration = parsed.end_hour - parsed.start_hour
    duration_match = abs(parsed_duration - instance.meeting_duration) < 0.01

    if not duration_match:
        return {
            "score": 0.0,
            "parsed": True,
            "valid": False,
            "exact_match": False,
            "error": f"Wrong duration: expected {instance.meeting_duration}h, got {parsed_duration}h"
        }

    # Check if all participants are available
    all_available = all(p.is_available(parsed) for p in instance.participants)

    if not all_available:
        return {
            "score": 0.0,
            "parsed": True,
            "valid": False,
            "exact_match": False,
            "error": "Proposed time conflicts with participant schedules"
        }

    # Check exact match with solution if required
    if exact_match:
        is_exact = (
            parsed.day == instance.solution.day and
            abs(parsed.start_hour - instance.solution.start_hour) < 0.01 and
            abs(parsed.end_hour - instance.solution.end_hour) < 0.01
        )
        return {
            "score": 1.0 if is_exact else 0.5,  # Partial credit for valid but different slot
            "parsed": True,
            "valid": True,
            "exact_match": is_exact,
            "error": None
        }
    else:
        # Any valid slot is acceptable
        return {
            "score": 1.0,
            "parsed": True,
            "valid": True,
            "exact_match": parsed.day == instance.solution.day and
                          abs(parsed.start_hour - instance.solution.start_hour) < 0.01,
            "error": None
        }


def compute_score(solution_str: str, ground_truth: dict, extra_info: dict = None) -> float:
    """
    Compute reward score for calendar scheduling task.

    Validates that:
    1. The proposed slot has the correct duration
    2. The proposed slot doesn't conflict with any participant's busy slots

    This allows multiple valid solutions - any time slot where all participants
    are available receives full credit.

    Args:
        solution_str: Model's response
        ground_truth: Dict containing solution info (day, start, end)
        extra_info: Optional dict with full instance info for validation

    Returns:
        Float score (0.0 or 1.0)
    """
    parsed = parse_calendar_response(solution_str)

    if parsed is None:
        return 0.0

    # Extract expected duration
    expected_duration = ground_truth["end"] - ground_truth["start"]
    actual_duration = parsed.end_hour - parsed.start_hour

    # Check 1: Duration must match
    if abs(actual_duration - expected_duration) > 0.01:
        return 0.0

    # Check 2: Verify slot is valid (doesn't conflict with busy slots)
    if extra_info and "participants" in extra_info:
        for p in extra_info["participants"]:
            for busy in p.get("busy_slots", []):
                busy_slot = TimeSlot(busy["day"], busy["start"], busy["end"])
                if parsed.overlaps(busy_slot):
                    return 0.0  # Conflicts with a busy slot
        # All participants available - valid solution!
        return 1.0

    # No extra_info available - fall back to exact match
    expected_day = ground_truth["day"]
    expected_start = ground_truth["start"]
    expected_end = ground_truth["end"]

    if (parsed.day == expected_day and
        abs(parsed.start_hour - expected_start) < 0.01 and
        abs(parsed.end_hour - expected_end) < 0.01):
        return 1.0

    return 0.0


if __name__ == "__main__":
    # Demo usage
    task = CalendarSchedulingTask(seed=42)

    print("=" * 60)
    print("Calendar Scheduling Task Demo")
    print("=" * 60)

    for i in range(3):
        instance = task.generate(num_people=3, num_days=1)
        print(f"\n--- Instance {i+1} ---")
        print(instance.get_prompt())
        print(f"\nSolution: {instance.get_solution_str()}")
        print("-" * 40)
