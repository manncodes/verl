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
Meeting Planning Task for Natural Plan Environment

This task involves scheduling meetings with multiple people at various locations,
considering travel times between locations and each person's availability.

The task is fully synthesizable and verifiable:
- Locations, distances, and schedules are generated programmatically
- A valid solution is computed using constraint satisfaction
- Verification counts valid meetings in the plan
"""

import random
import re
from dataclasses import dataclass, field
from typing import Optional
from heapq import heappush, heappop


# Location names (neighborhoods/districts)
LOCATIONS = [
    "Downtown", "Marina District", "Financial District", "Chinatown",
    "North Beach", "Russian Hill", "Pacific Heights", "Nob Hill",
    "Mission District", "Castro", "Haight-Ashbury", "Sunset District",
    "Richmond District", "SoMa", "Potrero Hill", "Bernal Heights"
]

# Common first names
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier"
]


def time_to_minutes(time_str: str) -> int:
    """Convert time string (e.g., '9:30AM') to minutes from midnight."""
    match = re.match(r"(\d{1,2}):(\d{2})(AM|PM)?", time_str, re.IGNORECASE)
    if not match:
        # Try 24-hour format
        match = re.match(r"(\d{1,2}):(\d{2})", time_str)
        if not match:
            return -1
        hours, mins = int(match.group(1)), int(match.group(2))
        return hours * 60 + mins

    hours, mins = int(match.group(1)), int(match.group(2))
    period = match.group(3)

    if period:
        period = period.upper()
        if period == "PM" and hours != 12:
            hours += 12
        elif period == "AM" and hours == 12:
            hours = 0

    return hours * 60 + mins


def minutes_to_time(minutes: int) -> str:
    """Convert minutes from midnight to time string (e.g., '9:30AM')."""
    hours = minutes // 60
    mins = minutes % 60
    period = "AM" if hours < 12 else "PM"
    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12
    return f"{hours}:{mins:02d}{period}"


@dataclass
class Person:
    """A person to meet with."""
    name: str
    location: str
    available_start: int  # minutes from midnight
    available_end: int    # minutes from midnight
    meeting_duration: int  # minutes

    def is_available_at(self, start_time: int) -> bool:
        """Check if person is available at the given start time."""
        end_time = start_time + self.meeting_duration
        return self.available_start <= start_time and end_time <= self.available_end

    def __str__(self) -> str:
        return (f"{self.name} at {self.location}, "
                f"available {minutes_to_time(self.available_start)}-{minutes_to_time(self.available_end)}, "
                f"meeting: {self.meeting_duration} min")


@dataclass
class MeetingStep:
    """A step in the meeting plan."""
    person: Optional[Person]  # None for travel-only steps
    location: str
    arrival_time: int
    departure_time: int
    action: str  # "travel", "wait", "meet"

    def __str__(self) -> str:
        if self.action == "meet":
            return (f"Meet {self.person.name} at {self.location} "
                   f"from {minutes_to_time(self.arrival_time)} to {minutes_to_time(self.departure_time)}")
        elif self.action == "travel":
            return f"Travel to {self.location}, arrive at {minutes_to_time(self.arrival_time)}"
        else:
            return f"Wait at {self.location} until {minutes_to_time(self.departure_time)}"


@dataclass
class MeetingPlanningInstance:
    """A single meeting planning problem instance."""
    start_location: str
    start_time: int
    people: list[Person]
    distance_matrix: dict[tuple[str, str], int]  # (from, to) -> travel time in minutes
    solution_plan: list[MeetingStep]
    solution_score: int  # number of meetings in optimal solution

    def get_prompt(self) -> str:
        """Generate the natural language prompt for this instance."""
        lines = []
        lines.append(f"You are at {self.start_location} at {minutes_to_time(self.start_time)}.")
        lines.append(f"You want to meet with {len(self.people)} people today.")
        lines.append("")
        lines.append("People to meet:")
        for p in self.people:
            lines.append(f"- {p.name}: at {p.location}, "
                        f"available {minutes_to_time(p.available_start)}-{minutes_to_time(p.available_end)}, "
                        f"meeting duration: {p.meeting_duration} minutes")
        lines.append("")
        lines.append("Travel times between locations:")
        # Show unique location pairs
        locations_used = {self.start_location} | {p.location for p in self.people}
        for loc1 in sorted(locations_used):
            for loc2 in sorted(locations_used):
                if loc1 < loc2:
                    time = self.distance_matrix.get((loc1, loc2), self.distance_matrix.get((loc2, loc1)))
                    if time:
                        lines.append(f"- {loc1} <-> {loc2}: {time} minutes")
        lines.append("")
        lines.append("Plan a schedule to meet as many people as possible.")
        lines.append("Format each step as:")
        lines.append("1. Travel to [Location], arrive at [Time]")
        lines.append("2. Meet [Name] at [Location] from [Start Time] to [End Time]")
        lines.append("")
        lines.append("End with: Total meetings: [Number]")

        return "\n".join(lines)

    def get_solution_str(self) -> str:
        """Get the solution in the expected format."""
        lines = []
        step_num = 1
        for step in self.solution_plan:
            if step.action == "travel":
                lines.append(f"{step_num}. Travel to {step.location}, arrive at {minutes_to_time(step.arrival_time)}")
            elif step.action == "meet":
                lines.append(f"{step_num}. Meet {step.person.name} at {step.location} "
                           f"from {minutes_to_time(step.arrival_time)} to {minutes_to_time(step.departure_time)}")
            step_num += 1
        lines.append(f"\nTotal meetings: {self.solution_score}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "start_location": self.start_location,
            "start_time": self.start_time,
            "people": [
                {
                    "name": p.name,
                    "location": p.location,
                    "available_start": p.available_start,
                    "available_end": p.available_end,
                    "meeting_duration": p.meeting_duration
                }
                for p in self.people
            ],
            "distance_matrix": {f"{k[0]}|{k[1]}": v for k, v in self.distance_matrix.items()},
            "solution_score": self.solution_score
        }


class MeetingPlanningTask:
    """
    Synthesizer for Meeting Planning problems.

    Generates instances with people at various locations, travel times,
    and computes optimal meeting schedules.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        min_people: int = 2,
        max_people: int = 5,
        min_travel_time: int = 5,
        max_travel_time: int = 30,
        min_meeting_duration: int = 15,
        max_meeting_duration: int = 60,
        day_start: int = 9 * 60,   # 9 AM
        day_end: int = 17 * 60,    # 5 PM
        min_availability_window: int = 60,  # 1 hour
        max_availability_window: int = 240,  # 4 hours
    ):
        """Initialize the meeting planning task synthesizer."""
        self.rng = random.Random(seed)
        self.min_people = min_people
        self.max_people = max_people
        self.min_travel_time = min_travel_time
        self.max_travel_time = max_travel_time
        self.min_meeting_duration = min_meeting_duration
        self.max_meeting_duration = max_meeting_duration
        self.day_start = day_start
        self.day_end = day_end
        self.min_availability = min_availability_window
        self.max_availability = max_availability_window

    def _generate_distance_matrix(self, locations: list[str]) -> dict[tuple[str, str], int]:
        """Generate symmetric travel times between locations."""
        matrix = {}
        for i, loc1 in enumerate(locations):
            for loc2 in locations[i+1:]:
                time = self.rng.randint(self.min_travel_time, self.max_travel_time)
                # Round to 5 minute increments
                time = (time // 5) * 5
                if time < self.min_travel_time:
                    time = self.min_travel_time
                matrix[(loc1, loc2)] = time
                matrix[(loc2, loc1)] = time
        return matrix

    def _solve_meeting_planning(
        self,
        start_location: str,
        start_time: int,
        people: list[Person],
        distance_matrix: dict,
    ) -> tuple[list[MeetingStep], int]:
        """
        Find optimal meeting schedule using dynamic programming / branch and bound.

        Returns tuple of (plan, num_meetings).
        """
        n = len(people)
        if n == 0:
            return [], 0

        # State: (current_time, current_location, met_mask)
        # Use priority queue with negative meetings count for max-heap behavior
        # Format: (neg_meetings, time, location, met_mask, path)

        best_score = 0
        best_path = []

        # BFS/DFS with pruning
        from collections import deque
        queue = deque()
        # Start state
        queue.append((start_time, start_location, 0, []))  # time, loc, met_mask, path

        visited = {}  # (time, loc, mask) -> best meetings count at this state

        while queue:
            current_time, current_loc, met_mask, path = queue.popleft()

            # Count current meetings
            current_meetings = bin(met_mask).count("1")

            # Check if we've seen better at this state
            state_key = (current_time // 5, current_loc, met_mask)  # discretize time
            if state_key in visited and visited[state_key] >= current_meetings:
                continue
            visited[state_key] = current_meetings

            # Update best if needed
            if current_meetings > best_score:
                best_score = current_meetings
                best_path = path.copy()

            # If all met, we're done
            if met_mask == (1 << n) - 1:
                continue

            # Pruning: if even meeting all remaining can't beat best, skip
            remaining = n - current_meetings
            if current_meetings + remaining <= best_score:
                continue

            # Try to meet each unmet person
            for i, person in enumerate(people):
                if met_mask & (1 << i):
                    continue  # Already met

                # Calculate arrival time at person's location
                if current_loc == person.location:
                    travel_time = 0
                else:
                    travel_time = distance_matrix.get(
                        (current_loc, person.location),
                        distance_matrix.get((person.location, current_loc), float('inf'))
                    )

                arrival_time = current_time + travel_time

                # Check if we can meet within their availability
                if arrival_time > person.available_end - person.meeting_duration:
                    continue  # Too late

                # Wait if we arrive before they're available
                meeting_start = max(arrival_time, person.available_start)
                meeting_end = meeting_start + person.meeting_duration

                if meeting_end > person.available_end:
                    continue  # Meeting would extend past their availability

                if meeting_end > self.day_end:
                    continue  # Past end of day

                # Valid meeting, add to queue
                new_mask = met_mask | (1 << i)
                new_path = path + [
                    MeetingStep(None, person.location, arrival_time, meeting_start, "travel"),
                    MeetingStep(person, person.location, meeting_start, meeting_end, "meet")
                ]
                queue.append((meeting_end, person.location, new_mask, new_path))

        return best_path, best_score

    def generate(
        self,
        num_people: Optional[int] = None,
    ) -> MeetingPlanningInstance:
        """
        Generate a single meeting planning instance.

        Args:
            num_people: Number of people to meet (random if None)

        Returns:
            A MeetingPlanningInstance with computed optimal solution
        """
        if num_people is None:
            num_people = self.rng.randint(self.min_people, self.max_people)

        # Select locations
        num_locations = min(num_people + 2, len(LOCATIONS))
        locations = self.rng.sample(LOCATIONS, num_locations)

        # Start location
        start_location = locations[0]
        start_time = self.day_start + self.rng.randint(0, 60)  # 9:00-10:00 AM
        start_time = (start_time // 5) * 5  # Round to 5 min

        # Generate distance matrix
        distance_matrix = self._generate_distance_matrix(locations)

        # Generate people
        names = self.rng.sample(FIRST_NAMES, num_people)
        people = []

        for i, name in enumerate(names):
            # Assign location (not start location usually)
            person_loc = locations[min(i + 1, len(locations) - 1)]

            # Generate availability window
            window_size = self.rng.randint(self.min_availability, self.max_availability)
            window_start = self.rng.randint(self.day_start, self.day_end - window_size)
            window_start = (window_start // 15) * 15  # Round to 15 min

            # Meeting duration
            duration = self.rng.randint(self.min_meeting_duration, self.max_meeting_duration)
            duration = (duration // 15) * 15  # Round to 15 min
            if duration < self.min_meeting_duration:
                duration = self.min_meeting_duration

            people.append(Person(
                name=name,
                location=person_loc,
                available_start=window_start,
                available_end=window_start + window_size,
                meeting_duration=duration
            ))

        # Solve for optimal schedule
        solution_plan, solution_score = self._solve_meeting_planning(
            start_location, start_time, people, distance_matrix
        )

        return MeetingPlanningInstance(
            start_location=start_location,
            start_time=start_time,
            people=people,
            distance_matrix=distance_matrix,
            solution_plan=solution_plan,
            solution_score=solution_score
        )

    def generate_batch(self, n: int, num_people: Optional[int] = None) -> list[MeetingPlanningInstance]:
        """Generate a batch of instances."""
        return [self.generate(num_people) for _ in range(n)]


def parse_meeting_response(response: str) -> list[dict]:
    """
    Parse a meeting planning response to extract the schedule.

    Returns list of meeting dicts with person name, location, start_time, end_time.
    """
    meetings = []

    # Pattern for meeting steps: "Meet [Name] at [Location] from [Time] to [Time]"
    meeting_pattern = r"[Mm]eet\s+(\w+)\s+at\s+([^,]+?)(?:\s+from)?\s+(\d{1,2}:\d{2}(?:AM|PM)?)\s*(?:to|-)\s*(\d{1,2}:\d{2}(?:AM|PM)?)"

    for match in re.finditer(meeting_pattern, response, re.IGNORECASE):
        name = match.group(1)
        location = match.group(2).strip()
        start_time = time_to_minutes(match.group(3))
        end_time = time_to_minutes(match.group(4))

        if start_time >= 0 and end_time >= 0:
            meetings.append({
                "name": name,
                "location": location,
                "start_time": start_time,
                "end_time": end_time
            })

    return meetings


def verify_meeting_plan(
    response: str,
    instance: MeetingPlanningInstance,
) -> dict:
    """
    Verify if a response correctly solves the meeting planning problem.

    Validates:
    1. Meetings are at correct locations
    2. Times are within availability windows
    3. Travel times are respected
    4. No duplicate meetings

    Returns:
        Dict with score and detailed metrics
    """
    parsed_meetings = parse_meeting_response(response)

    if not parsed_meetings:
        return {
            "score": 0.0,
            "valid_meetings": 0,
            "total_meetings": 0,
            "optimal_meetings": instance.solution_score,
            "error": "No meetings parsed from response"
        }

    # Build name -> Person lookup
    person_lookup = {p.name.lower(): p for p in instance.people}

    valid_meetings = 0
    met_people = set()
    current_time = instance.start_time
    current_location = instance.start_location
    errors = []

    for meeting in parsed_meetings:
        name = meeting["name"].lower()

        # Check if person exists
        if name not in person_lookup:
            errors.append(f"Unknown person: {meeting['name']}")
            continue

        person = person_lookup[name]

        # Check for duplicate
        if name in met_people:
            errors.append(f"Duplicate meeting with {person.name}")
            continue

        # Check location
        if meeting["location"].lower() != person.location.lower():
            errors.append(f"Wrong location for {person.name}")
            continue

        # Check travel time
        if current_location != person.location:
            travel_time = instance.distance_matrix.get(
                (current_location, person.location),
                instance.distance_matrix.get((person.location, current_location), 0)
            )
            arrival_time = current_time + travel_time
        else:
            arrival_time = current_time

        # Check if meeting time is valid
        if meeting["start_time"] < arrival_time:
            errors.append(f"Meeting with {person.name} starts before arrival")
            continue

        if meeting["start_time"] < person.available_start:
            errors.append(f"{person.name} not available at meeting start time")
            continue

        if meeting["end_time"] > person.available_end:
            errors.append(f"Meeting with {person.name} extends past availability")
            continue

        # Check meeting duration
        actual_duration = meeting["end_time"] - meeting["start_time"]
        if actual_duration < person.meeting_duration:
            errors.append(f"Meeting with {person.name} too short")
            continue

        # Valid meeting
        valid_meetings += 1
        met_people.add(name)
        current_time = meeting["end_time"]
        current_location = person.location

    # Score based on ratio to optimal
    if instance.solution_score > 0:
        score = valid_meetings / instance.solution_score
    else:
        score = 1.0 if valid_meetings == 0 else 0.0

    return {
        "score": score,
        "valid_meetings": valid_meetings,
        "total_meetings": len(parsed_meetings),
        "optimal_meetings": instance.solution_score,
        "errors": errors
    }


def compute_score(solution_str: str, ground_truth: dict, extra_info: dict = None) -> float:
    """
    Compute reward score for meeting planning task.

    Args:
        solution_str: Model's response
        ground_truth: Dict containing solution_score (optimal number of meetings)
        extra_info: Dict with full instance info for validation

    Returns:
        Float score (0.0 to 1.0)
    """
    if extra_info is None:
        # Can't fully validate without instance info
        # Just check if response mentions meetings
        parsed = parse_meeting_response(solution_str)
        optimal = ground_truth.get("solution_score", 1)
        if optimal == 0:
            return 1.0 if len(parsed) == 0 else 0.0
        return min(1.0, len(parsed) / optimal)

    # Reconstruct instance for full validation
    people = [
        Person(
            name=p["name"],
            location=p["location"],
            available_start=p["available_start"],
            available_end=p["available_end"],
            meeting_duration=p["meeting_duration"]
        )
        for p in extra_info["people"]
    ]

    # Reconstruct distance matrix
    distance_matrix = {}
    for key, value in extra_info["distance_matrix"].items():
        loc1, loc2 = key.split("|")
        distance_matrix[(loc1, loc2)] = value

    instance = MeetingPlanningInstance(
        start_location=extra_info["start_location"],
        start_time=extra_info["start_time"],
        people=people,
        distance_matrix=distance_matrix,
        solution_plan=[],
        solution_score=ground_truth["solution_score"]
    )

    result = verify_meeting_plan(solution_str, instance)
    return result["score"]


if __name__ == "__main__":
    # Demo usage
    task = MeetingPlanningTask(seed=42)

    print("=" * 60)
    print("Meeting Planning Task Demo")
    print("=" * 60)

    for i in range(2):
        instance = task.generate(num_people=3)
        print(f"\n--- Instance {i+1} ---")
        print(instance.get_prompt())
        print(f"\n--- Solution (Optimal: {instance.solution_score} meetings) ---")
        print(instance.get_solution_str())
        print("-" * 40)
