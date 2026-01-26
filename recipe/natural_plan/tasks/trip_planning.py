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
Trip Planning Task for Natural Plan Environment

This task involves planning an optimal trip itinerary, visiting multiple cities
with specified durations, subject to various constraints.

The task is fully synthesizable and verifiable:
- Cities, durations, and constraints are generated programmatically
- The optimal solution is computed during generation
- Verification checks exact match of city sequence and durations
"""

import random
import re
from dataclasses import dataclass
from typing import Optional
from itertools import permutations


# City names for trip planning
CITIES = {
    "europe": [
        "Paris", "London", "Rome", "Barcelona", "Amsterdam",
        "Berlin", "Prague", "Vienna", "Munich", "Florence",
        "Venice", "Zurich", "Brussels", "Madrid", "Lisbon",
        "Budapest", "Copenhagen", "Stockholm", "Dublin", "Athens"
    ],
    "asia": [
        "Tokyo", "Seoul", "Singapore", "Bangkok", "Hong Kong",
        "Taipei", "Osaka", "Kyoto", "Bali", "Kuala Lumpur",
        "Shanghai", "Beijing", "Hanoi", "Ho Chi Minh City", "Manila"
    ],
    "americas": [
        "New York", "Los Angeles", "Chicago", "Miami", "San Francisco",
        "Boston", "Seattle", "Denver", "Austin", "Nashville",
        "Toronto", "Vancouver", "Montreal", "Mexico City", "Cancun"
    ]
}

# City attractions/highlights for generating constraints
ATTRACTIONS = {
    "Paris": ["Eiffel Tower", "Louvre", "Notre-Dame"],
    "London": ["Big Ben", "Tower of London", "British Museum"],
    "Rome": ["Colosseum", "Vatican", "Trevi Fountain"],
    "Barcelona": ["Sagrada Familia", "Park Guell", "La Rambla"],
    "Amsterdam": ["Anne Frank House", "Van Gogh Museum", "canals"],
    "Tokyo": ["Shibuya", "Senso-ji", "Tokyo Tower"],
    "New York": ["Statue of Liberty", "Central Park", "Times Square"],
}


@dataclass
class TripConstraint:
    """A constraint on the trip itinerary."""
    constraint_type: str  # "before", "after", "consecutive", "duration_min", "duration_max"
    city1: str
    city2: Optional[str] = None
    value: Optional[int] = None

    def __str__(self) -> str:
        if self.constraint_type == "before":
            return f"Visit {self.city1} before {self.city2}"
        elif self.constraint_type == "after":
            return f"Visit {self.city1} after {self.city2}"
        elif self.constraint_type == "consecutive":
            return f"Visit {self.city1} and {self.city2} consecutively"
        elif self.constraint_type == "duration_min":
            return f"Spend at least {self.value} days in {self.city1}"
        elif self.constraint_type == "duration_max":
            return f"Spend at most {self.value} days in {self.city1}"
        elif self.constraint_type == "first":
            return f"Start the trip in {self.city1}"
        elif self.constraint_type == "last":
            return f"End the trip in {self.city1}"
        return str(self.__dict__)

    def is_satisfied(self, cities: list[str], durations: list[int]) -> bool:
        """Check if constraint is satisfied by the given itinerary."""
        if self.city1 not in cities:
            return False

        idx1 = cities.index(self.city1)

        if self.constraint_type == "before":
            if self.city2 not in cities:
                return False
            idx2 = cities.index(self.city2)
            return idx1 < idx2

        elif self.constraint_type == "after":
            if self.city2 not in cities:
                return False
            idx2 = cities.index(self.city2)
            return idx1 > idx2

        elif self.constraint_type == "consecutive":
            if self.city2 not in cities:
                return False
            idx2 = cities.index(self.city2)
            return abs(idx1 - idx2) == 1

        elif self.constraint_type == "duration_min":
            return durations[idx1] >= self.value

        elif self.constraint_type == "duration_max":
            return durations[idx1] <= self.value

        elif self.constraint_type == "first":
            return idx1 == 0

        elif self.constraint_type == "last":
            return idx1 == len(cities) - 1

        return True


@dataclass
class TripPlanningInstance:
    """A single trip planning problem instance."""
    cities: list[str]
    durations: list[int]
    total_days: int
    constraints: list[TripConstraint]
    region: str
    solution_cities: list[str]
    solution_durations: list[int]

    def get_prompt(self) -> str:
        """Generate the natural language prompt for this instance."""
        lines = []

        region_name = self.region.replace("_", " ").title()
        lines.append(f"Plan a {self.total_days}-day trip visiting {len(self.cities)} {region_name} cities:")
        lines.append(f"Cities to visit: {', '.join(self.cities)}")
        lines.append("")

        lines.append("Requirements:")
        for i, (city, duration) in enumerate(zip(self.cities, self.durations), 1):
            lines.append(f"- Spend exactly {duration} day{'s' if duration > 1 else ''} in {city}")

        if self.constraints:
            lines.append("")
            lines.append("Additional constraints:")
            for constraint in self.constraints:
                lines.append(f"- {constraint}")

        lines.append("")
        lines.append("Provide the optimal itinerary.")
        lines.append("Format: List each city visit as 'Days X-Y: [City]' or 'Day X: [City]'")
        lines.append("Example: Days 1-3: Paris, Days 4-5: London, Day 6: Amsterdam")

        return "\n".join(lines)

    def get_solution_str(self) -> str:
        """Get the solution in the expected format."""
        lines = []
        current_day = 1
        for city, duration in zip(self.solution_cities, self.solution_durations):
            if duration == 1:
                lines.append(f"Day {current_day}: {city}")
            else:
                end_day = current_day + duration - 1
                lines.append(f"Days {current_day}-{end_day}: {city}")
            current_day += duration
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cities": self.cities,
            "durations": self.durations,
            "total_days": self.total_days,
            "region": self.region,
            "constraints": [
                {
                    "type": c.constraint_type,
                    "city1": c.city1,
                    "city2": c.city2,
                    "value": c.value
                }
                for c in self.constraints
            ],
            "solution_cities": self.solution_cities,
            "solution_durations": self.solution_durations
        }


class TripPlanningTask:
    """
    Synthesizer for Trip Planning problems.

    Generates instances with cities to visit, durations, and constraints,
    then finds valid orderings.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        min_cities: int = 3,
        max_cities: int = 6,
        min_duration_per_city: int = 1,
        max_duration_per_city: int = 4,
        min_constraints: int = 0,
        max_constraints: int = 3,
        regions: Optional[list[str]] = None,
    ):
        """Initialize the trip planning task synthesizer."""
        self.rng = random.Random(seed)
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.min_duration = min_duration_per_city
        self.max_duration = max_duration_per_city
        self.min_constraints = min_constraints
        self.max_constraints = max_constraints
        self.regions = regions or list(CITIES.keys())

    def _generate_constraints(
        self,
        cities: list[str],
        durations: list[int],
        num_constraints: int
    ) -> list[TripConstraint]:
        """Generate random valid constraints."""
        constraints = []
        constraint_types = ["before", "consecutive", "first", "last"]

        attempts = 0
        while len(constraints) < num_constraints and attempts < 50:
            attempts += 1

            ctype = self.rng.choice(constraint_types)

            if ctype == "before":
                if len(cities) < 2:
                    continue
                city1, city2 = self.rng.sample(cities, 2)
                constraint = TripConstraint("before", city1, city2)

            elif ctype == "consecutive":
                if len(cities) < 2:
                    continue
                city1, city2 = self.rng.sample(cities, 2)
                constraint = TripConstraint("consecutive", city1, city2)

            elif ctype == "first":
                city = self.rng.choice(cities)
                # Don't add if we already have a first constraint
                if any(c.constraint_type == "first" for c in constraints):
                    continue
                constraint = TripConstraint("first", city)

            elif ctype == "last":
                city = self.rng.choice(cities)
                # Don't add if we already have a last constraint
                if any(c.constraint_type == "last" for c in constraints):
                    continue
                constraint = TripConstraint("last", city)

            else:
                continue

            # Check for conflicts with existing constraints
            conflict = False
            for existing in constraints:
                # Check direct conflicts
                if (constraint.constraint_type == "before" and
                    existing.constraint_type == "before" and
                    constraint.city1 == existing.city2 and
                    constraint.city2 == existing.city1):
                    conflict = True
                    break

            if not conflict:
                constraints.append(constraint)

        return constraints

    def _find_valid_ordering(
        self,
        cities: list[str],
        durations: list[int],
        constraints: list[TripConstraint],
    ) -> Optional[tuple[list[str], list[int]]]:
        """Find a valid ordering of cities satisfying all constraints."""
        # For small number of cities, try all permutations
        if len(cities) <= 8:
            city_duration = dict(zip(cities, durations))

            for perm in permutations(cities):
                perm_list = list(perm)
                perm_durations = [city_duration[c] for c in perm_list]

                if all(c.is_satisfied(perm_list, perm_durations) for c in constraints):
                    return perm_list, perm_durations

        # For larger instances, use constraint propagation
        # Start with any ordering and swap to satisfy constraints
        city_duration = dict(zip(cities, durations))
        current = list(cities)
        self.rng.shuffle(current)

        # Apply hard constraints first
        for constraint in constraints:
            if constraint.constraint_type == "first":
                idx = current.index(constraint.city1)
                current[0], current[idx] = current[idx], current[0]
            elif constraint.constraint_type == "last":
                idx = current.index(constraint.city1)
                current[-1], current[idx] = current[idx], current[-1]

        # Try to fix other constraints
        for _ in range(100):  # Max iterations
            all_satisfied = True
            for constraint in constraints:
                current_durations = [city_duration[c] for c in current]
                if not constraint.is_satisfied(current, current_durations):
                    all_satisfied = False
                    # Try to fix
                    if constraint.constraint_type == "before":
                        idx1 = current.index(constraint.city1)
                        idx2 = current.index(constraint.city2)
                        if idx1 > idx2:
                            # Swap
                            current[idx1], current[idx2] = current[idx2], current[idx1]
                    elif constraint.constraint_type == "consecutive":
                        idx1 = current.index(constraint.city1)
                        idx2 = current.index(constraint.city2)
                        if abs(idx1 - idx2) != 1:
                            # Move city2 next to city1
                            current.remove(constraint.city2)
                            new_idx = idx1 + 1 if idx1 < len(current) else idx1
                            current.insert(new_idx, constraint.city2)

            if all_satisfied:
                return current, [city_duration[c] for c in current]

        return None

    def generate(
        self,
        num_cities: Optional[int] = None,
        region: Optional[str] = None,
        num_constraints: Optional[int] = None,
    ) -> TripPlanningInstance:
        """
        Generate a single trip planning instance.

        Args:
            num_cities: Number of cities (random if None)
            region: Region for cities (random if None)
            num_constraints: Number of constraints (random if None)

        Returns:
            A TripPlanningInstance with valid solution
        """
        if num_cities is None:
            num_cities = self.rng.randint(self.min_cities, self.max_cities)
        if region is None:
            region = self.rng.choice(self.regions)
        if num_constraints is None:
            num_constraints = self.rng.randint(self.min_constraints, self.max_constraints)

        # Select cities
        available_cities = CITIES.get(region, CITIES["europe"])
        cities = self.rng.sample(available_cities, min(num_cities, len(available_cities)))

        # Generate durations
        durations = [
            self.rng.randint(self.min_duration, self.max_duration)
            for _ in cities
        ]
        total_days = sum(durations)

        # Generate constraints and find valid ordering
        for attempt in range(10):  # Retry with different constraints if needed
            constraints = self._generate_constraints(cities, durations, num_constraints)
            result = self._find_valid_ordering(cities, durations, constraints)

            if result is not None:
                solution_cities, solution_durations = result
                break
        else:
            # No valid ordering found, generate without constraints
            constraints = []
            solution_cities = cities.copy()
            self.rng.shuffle(solution_cities)
            city_duration = dict(zip(cities, durations))
            solution_durations = [city_duration[c] for c in solution_cities]

        return TripPlanningInstance(
            cities=cities,
            durations=durations,
            total_days=total_days,
            constraints=constraints,
            region=region,
            solution_cities=solution_cities,
            solution_durations=solution_durations
        )

    def generate_batch(
        self,
        n: int,
        num_cities: Optional[int] = None,
        region: Optional[str] = None,
    ) -> list[TripPlanningInstance]:
        """Generate a batch of instances."""
        return [self.generate(num_cities, region) for _ in range(n)]


def _consolidate_consecutive_cities(cities: list[str], durations: list[int]) -> tuple[list[str], list[int]]:
    """Consolidate consecutive entries for the same city."""
    if not cities:
        return cities, durations

    consolidated_cities = []
    consolidated_durations = []

    current_city = cities[0]
    current_duration = durations[0]

    for i in range(1, len(cities)):
        if cities[i].lower() == current_city.lower():
            # Same city, add duration
            current_duration += durations[i]
        else:
            # Different city, save current and start new
            consolidated_cities.append(current_city)
            consolidated_durations.append(current_duration)
            current_city = cities[i]
            current_duration = durations[i]

    # Don't forget the last city
    consolidated_cities.append(current_city)
    consolidated_durations.append(current_duration)

    return consolidated_cities, consolidated_durations


def parse_trip_response(response: str) -> Optional[tuple[list[str], list[int]]]:
    """
    Parse a trip planning response to extract the itinerary.

    Expected formats:
    - "Days 1-3: Paris" or "Day 1: Paris"
    - "Paris (3 days)" or "Paris: 3 days"
    - Comma-separated list with durations

    Returns:
        Tuple of (cities, durations) if parsed successfully, None otherwise
    """
    cities = []
    durations = []

    # Pattern 1: "Days X-Y: City" or "Day X: City"
    pattern1 = r"Days?\s+(\d+)(?:-(\d+))?:?\s+([A-Za-z][A-Za-z\s]+?)(?:,|$|\n)"
    for match in re.finditer(pattern1, response, re.IGNORECASE):
        start_day = int(match.group(1))
        end_day = int(match.group(2)) if match.group(2) else start_day
        city = match.group(3).strip()

        duration = end_day - start_day + 1
        cities.append(city)
        durations.append(duration)

    if cities:
        # Consolidate consecutive same-city entries (e.g., Day 1: Paris, Day 2: Paris -> Paris: 2 days)
        return _consolidate_consecutive_cities(cities, durations)

    # Pattern 2: "City (N days)" or "City: N days"
    pattern2 = r"([A-Za-z][A-Za-z\s]+?)\s*[\(:]?\s*(\d+)\s*days?\s*[\)]?"
    for match in re.finditer(pattern2, response, re.IGNORECASE):
        city = match.group(1).strip()
        duration = int(match.group(2))

        # Skip if city looks like a number or common word
        if city.lower() in ["spend", "visit", "stay", "for", "day", "days"]:
            continue

        cities.append(city)
        durations.append(duration)

    if cities:
        return cities, durations

    return None


def verify_trip_solution(
    response: str,
    instance: TripPlanningInstance,
    exact_order: bool = True,
) -> dict:
    """
    Verify if a response correctly solves the trip planning problem.

    Args:
        response: The model's response string
        instance: The problem instance
        exact_order: If True, requires exact match with solution order

    Returns:
        Dict with score and detailed metrics
    """
    parsed = parse_trip_response(response)

    if parsed is None:
        return {
            "score": 0.0,
            "parsed": False,
            "cities_match": False,
            "durations_match": False,
            "constraints_satisfied": False,
            "error": "Could not parse response"
        }

    parsed_cities, parsed_durations = parsed

    # Normalize city names for comparison
    def normalize(s):
        return s.lower().strip()

    expected_cities = [normalize(c) for c in instance.solution_cities]
    expected_durations = instance.solution_durations
    actual_cities = [normalize(c) for c in parsed_cities]

    # Check cities match (allowing for order differences if not exact_order)
    if exact_order:
        cities_match = actual_cities == expected_cities
    else:
        cities_match = sorted(actual_cities) == sorted(expected_cities)

    # Check durations match
    if exact_order:
        durations_match = parsed_durations == expected_durations
    else:
        # Match durations to cities
        expected_map = dict(zip(expected_cities, expected_durations))
        durations_match = all(
            parsed_durations[i] == expected_map.get(c, -1)
            for i, c in enumerate(actual_cities)
        )

    # Check constraints
    constraints_satisfied = all(
        c.is_satisfied(parsed_cities, parsed_durations)
        for c in instance.constraints
    )

    # Calculate score
    if cities_match and durations_match and constraints_satisfied:
        score = 1.0
    elif cities_match and constraints_satisfied:
        score = 0.5  # Partial credit for correct cities but wrong durations
    else:
        score = 0.0

    return {
        "score": score,
        "parsed": True,
        "cities_match": cities_match,
        "durations_match": durations_match,
        "constraints_satisfied": constraints_satisfied,
        "error": None
    }


def compute_score(solution_str: str, ground_truth: dict, extra_info: dict = None) -> float:
    """
    Compute reward score for trip planning task.

    Args:
        solution_str: Model's response
        ground_truth: Dict containing solution_cities and solution_durations
        extra_info: Optional dict with constraints

    Returns:
        Float score (0.0 or 1.0)
    """
    parsed = parse_trip_response(solution_str)

    if parsed is None:
        return 0.0

    parsed_cities, parsed_durations = parsed

    # Normalize for comparison
    def normalize(s):
        return s.lower().strip()

    expected_cities = [normalize(c) for c in ground_truth["solution_cities"]]
    expected_durations = ground_truth["solution_durations"]
    actual_cities = [normalize(c) for c in parsed_cities]

    # Exact match required
    if actual_cities == expected_cities and parsed_durations == expected_durations:
        return 1.0

    return 0.0


if __name__ == "__main__":
    # Demo usage
    task = TripPlanningTask(seed=42)

    print("=" * 60)
    print("Trip Planning Task Demo")
    print("=" * 60)

    for i in range(3):
        instance = task.generate(num_cities=4, num_constraints=2)
        print(f"\n--- Instance {i+1} ---")
        print(instance.get_prompt())
        print(f"\n--- Solution ---")
        print(instance.get_solution_str())
        print("-" * 40)
