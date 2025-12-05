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
Verifiable reward functions for the NaturalPlan benchmark.
Reference: https://github.com/google-deepmind/natural-plan
Paper: "NATURAL PLAN: Benchmarking LLMs on Natural Language Planning" (arXiv:2406.04520)

This module implements verification for three planning tasks:
1. Trip Planning - Plan trips with flight connections and city visits
2. Meeting Planning - Schedule meetings with people at various locations
3. Calendar Scheduling - Find meeting times for multiple attendees
"""

import re
from typing import Any


# =============================================================================
# Trip Planning Verification
# =============================================================================

def _parse_trip_plan(solution_str: str) -> list[tuple[str, int]] | None:
    """
    Parse a trip planning response to extract city visits and durations.

    Expected formats:
    - "Day 1-3: City A" or "Days 1-3: City A"
    - "Day 1 to Day 3: City A"
    - "Day X from CityA to CityB" (for flights)

    Returns:
        List of (city, num_days) tuples or None if parsing fails
    """
    if not solution_str:
        return None

    visits = []

    # Pattern 1: "Day X-Y: City" or "Days X-Y: City"
    pattern1 = r"[Dd]ays?\s*(\d+)\s*[-–to]+\s*(\d+)\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$|Day|,)"
    matches = re.findall(pattern1, solution_str)
    for match in matches:
        start_day = int(match[0])
        end_day = int(match[1])
        city = match[2].strip().rstrip('.')
        if city:
            duration = end_day - start_day + 1
            visits.append((city, duration))

    if visits:
        return visits

    # Pattern 2: "Day X: City" for single day visits
    pattern2 = r"[Dd]ay\s*(\d+)\s*[:\-]\s*([A-Za-z\s]+?)(?:\n|$|Day|,)"
    matches = re.findall(pattern2, solution_str)

    # Group consecutive days in the same city
    if matches:
        current_city = None
        current_count = 0
        for _, city in matches:
            city = city.strip().rstrip('.')
            if city == current_city:
                current_count += 1
            else:
                if current_city:
                    visits.append((current_city, current_count))
                current_city = city
                current_count = 1
        if current_city:
            visits.append((current_city, current_count))

    if visits:
        return visits

    # Pattern 3: Look for city names with durations like "3 days in Paris"
    pattern3 = r"(\d+)\s*days?\s*(?:in|at)\s*([A-Za-z\s]+?)(?:\n|$|,|\.)"
    matches = re.findall(pattern3, solution_str, re.IGNORECASE)
    for match in matches:
        duration = int(match[0])
        city = match[1].strip()
        if city:
            visits.append((city, duration))

    return visits if visits else None


def compute_score_trip_planning(
    solution_str: str,
    ground_truth: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for Trip Planning task.

    Args:
        solution_str: Model's generated trip plan
        ground_truth: Dict containing:
            - cities: List of cities in order (e.g., ["Paris", "London", "Rome"])
            - durations: List of stay durations (e.g., [3, 2, 4])
            - or 'golden_plan': String with expected answer

    Returns:
        Dict with 'score' (0.0 or 1.0), 'acc', and 'pred'
    """
    # Parse ground truth
    if "cities" in ground_truth and "durations" in ground_truth:
        gt_cities = ground_truth["cities"]
        gt_durations = ground_truth["durations"]
    elif "golden_plan" in ground_truth:
        # Parse golden plan format: "city1**city2**city3" and "1**2**3"
        gt_plan = ground_truth["golden_plan"]
        if "**" in str(gt_plan):
            parts = str(gt_plan).split("**")
            gt_cities = parts
            gt_durations = ground_truth.get("durations", [1] * len(parts))
        else:
            return {"score": 0.0, "acc": False, "pred": None, "error": "invalid_ground_truth"}
    else:
        return {"score": 0.0, "acc": False, "pred": None, "error": "missing_ground_truth"}

    # Parse model output
    parsed = _parse_trip_plan(solution_str)

    if parsed is None:
        return {"score": 0.0, "acc": False, "pred": None, "error": "parse_failed"}

    pred_cities = [p[0] for p in parsed]
    pred_durations = [p[1] for p in parsed]

    # Normalize city names for comparison
    def normalize_city(city: str) -> str:
        return city.lower().strip()

    gt_cities_norm = [normalize_city(c) for c in gt_cities]
    pred_cities_norm = [normalize_city(c) for c in pred_cities]

    # Check exact match
    correct = (
        len(pred_cities_norm) == len(gt_cities_norm) and
        pred_cities_norm == gt_cities_norm and
        pred_durations == list(gt_durations)
    )

    # Compute partial match score (for analysis)
    num_match = 0
    for i in range(min(len(pred_cities_norm), len(gt_cities_norm))):
        if pred_cities_norm[i] == gt_cities_norm[i] and pred_durations[i] == gt_durations[i]:
            num_match += 1
        else:
            break

    partial_score = num_match / len(gt_cities) if gt_cities else 0.0

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": {"cities": pred_cities, "durations": pred_durations},
        "partial_score": partial_score,
        "num_match": num_match,
    }


# =============================================================================
# Meeting Planning Verification
# =============================================================================

def _parse_time(time_str: str) -> float | None:
    """Parse time string like '9:00' or '14:30' to hours (float)."""
    match = re.match(r"(\d{1,2}):(\d{2})", time_str.strip())
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours + minutes / 60.0
    return None


def _parse_meeting_plan(solution_str: str) -> list[dict[str, Any]] | None:
    """
    Parse a meeting planning response.

    Expected formats:
    - JSON: [{"location": "X", "person_name": "Y", "start_time": "HH:MM"}, ...]
    - Text: "You start at Location at Time. You travel to Location2..."
    - Text: "Meet Person at Location from Time to Time"

    Returns:
        List of meeting dicts with 'location', 'person_name', 'start_time', 'end_time'
    """
    if not solution_str:
        return None

    meetings = []

    # Try JSON format first
    import json
    try:
        # Look for JSON array in the response
        json_match = re.search(r'\[[\s\S]*\]', solution_str)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "location" in item and "person_name" in item:
                        meetings.append({
                            "location": item["location"],
                            "person_name": item["person_name"],
                            "start_time": _parse_time(str(item.get("start_time", "0:00"))),
                            "end_time": _parse_time(str(item.get("end_time", ""))) if "end_time" in item else None,
                        })
                if meetings:
                    return meetings
    except (json.JSONDecodeError, ValueError):
        pass

    # Pattern: "Meet [Person] at [Location] at [Time]" or "Meet [Person] at [Location] from [Time] to [Time]"
    pattern1 = r"[Mm]eet\s+([A-Za-z]+)\s+at\s+([A-Za-z\s]+?)\s+(?:at|from)\s+(\d{1,2}:\d{2})"
    matches = re.findall(pattern1, solution_str)
    for match in matches:
        person = match[0].strip()
        location = match[1].strip()
        start_time = _parse_time(match[2])
        meetings.append({
            "location": location,
            "person_name": person,
            "start_time": start_time,
            "end_time": None,
        })

    if meetings:
        return meetings

    # Pattern: "[Time] - [Time]: Meet [Person] at [Location]"
    pattern2 = r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})[:\s]+[Mm]eet\s+([A-Za-z]+)\s+at\s+([A-Za-z\s]+)"
    matches = re.findall(pattern2, solution_str)
    for match in matches:
        start_time = _parse_time(match[0])
        end_time = _parse_time(match[1])
        person = match[2].strip()
        location = match[3].strip()
        meetings.append({
            "location": location,
            "person_name": person,
            "start_time": start_time,
            "end_time": end_time,
        })

    return meetings if meetings else None


def compute_score_meeting_planning(
    solution_str: str,
    ground_truth: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for Meeting Planning task.

    The task is to meet as many people as possible given:
    - People's locations and availability windows
    - Travel times between locations
    - Meeting duration requirements

    Args:
        solution_str: Model's generated meeting plan
        ground_truth: Dict containing:
            - people: List of dicts with 'name', 'location', 'start_time', 'end_time'
            - distances: Dict mapping (loc1, loc2) to travel time in minutes
            - meeting_duration: Duration of each meeting in minutes
            - golden_score: Expected number of people that can be met
            - initial_location: Starting location
            - initial_time: Starting time

    Returns:
        Dict with 'score', 'acc', 'pred', 'meetings_count'
    """
    # Parse constraints from ground truth
    people = ground_truth.get("people", [])
    distances = ground_truth.get("distances", {})
    meeting_duration = ground_truth.get("meeting_duration", 30) / 60.0  # Convert to hours
    golden_score = ground_truth.get("golden_score", len(people))
    initial_location = ground_truth.get("initial_location", "")
    initial_time = ground_truth.get("initial_time", 9.0)

    if isinstance(initial_time, str):
        initial_time = _parse_time(initial_time) or 9.0

    # Parse model output
    parsed = _parse_meeting_plan(solution_str)

    if parsed is None:
        return {"score": 0.0, "acc": False, "pred": None, "meetings_count": 0, "error": "parse_failed"}

    # Build lookup for people availability
    people_lookup = {}
    for p in people:
        name = p["name"].lower()
        start = _parse_time(str(p.get("start_time", "0:00"))) or 0
        end = _parse_time(str(p.get("end_time", "24:00"))) or 24
        people_lookup[name] = {
            "location": p.get("location", "").lower(),
            "start_time": start,
            "end_time": end,
        }

    # Simulate the schedule
    current_time = initial_time
    current_location = initial_location.lower()
    met_people = set()
    valid_meetings = 0

    for meeting in parsed:
        person_name = meeting["person_name"].lower()
        location = meeting["location"].lower()
        meeting_start = meeting["start_time"]

        if person_name not in people_lookup:
            continue

        person_info = people_lookup[person_name]

        # Check if already met
        if person_name in met_people:
            continue

        # Calculate travel time
        if current_location != location:
            dist_key = (current_location, location)
            alt_key = (location, current_location)
            travel_time = distances.get(dist_key, distances.get(alt_key, 0))
            if isinstance(travel_time, (int, float)):
                travel_time = travel_time / 60.0  # Convert minutes to hours
            else:
                travel_time = 0
            arrival_time = current_time + travel_time
        else:
            arrival_time = current_time

        # Check constraints
        if meeting_start and meeting_start < arrival_time:
            # Can't be there in time
            continue

        actual_start = max(arrival_time, meeting_start or arrival_time)
        meeting_end = actual_start + meeting_duration

        # Check if person is at the right location and available
        if location != person_info["location"]:
            continue
        if actual_start < person_info["start_time"]:
            continue
        if meeting_end > person_info["end_time"]:
            continue

        # Meeting is valid
        met_people.add(person_name)
        valid_meetings += 1
        current_time = meeting_end
        current_location = location

    # Compute score based on whether we achieved the golden score
    correct = valid_meetings >= golden_score

    # Partial score: fraction of meetings achieved
    partial_score = valid_meetings / golden_score if golden_score > 0 else 0.0

    return {
        "score": 1.0 if correct else partial_score,
        "acc": correct,
        "pred": parsed,
        "meetings_count": valid_meetings,
        "golden_score": golden_score,
        "partial_score": partial_score,
    }


# =============================================================================
# Calendar Scheduling Verification
# =============================================================================

def _parse_calendar_slot(solution_str: str) -> dict[str, Any] | None:
    """
    Parse a calendar scheduling response to extract the proposed meeting time.

    Expected format: "Day, HH:MM - HH:MM" (e.g., "Monday, 9:00 - 10:30")

    Returns:
        Dict with 'day', 'start_time', 'end_time' or None if parsing fails
    """
    if not solution_str:
        return None

    # Pattern: "Day, HH:MM - HH:MM" or "Day HH:MM - HH:MM"
    days = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    pattern = rf"({days})[,\s]+(\d{{1,2}}:\d{{2}})\s*[-–to]+\s*(\d{{1,2}}:\d{{2}})"

    match = re.search(pattern, solution_str, re.IGNORECASE)
    if match:
        return {
            "day": match.group(1).lower().capitalize(),
            "start_time": _parse_time(match.group(2)),
            "end_time": _parse_time(match.group(3)),
        }

    # Alternative pattern: "HH:MM - HH:MM on Day"
    pattern2 = rf"(\d{{1,2}}:\d{{2}})\s*[-–to]+\s*(\d{{1,2}}:\d{{2}})\s+on\s+({days})"
    match = re.search(pattern2, solution_str, re.IGNORECASE)
    if match:
        return {
            "day": match.group(3).lower().capitalize(),
            "start_time": _parse_time(match.group(1)),
            "end_time": _parse_time(match.group(2)),
        }

    return None


def _check_slot_available(
    slot: dict[str, Any],
    attendees: list[dict[str, Any]],
    work_hours: tuple[float, float] = (9.0, 17.0),
) -> bool:
    """Check if a time slot is available for all attendees."""
    slot_day = slot["day"].lower()
    slot_start = slot["start_time"]
    slot_end = slot["end_time"]

    # Check work hours
    if slot_start < work_hours[0] or slot_end > work_hours[1]:
        return False

    # Check each attendee's busy times
    for attendee in attendees:
        busy_times = attendee.get("busy_times", [])
        for busy in busy_times:
            busy_day = busy.get("day", "").lower()
            if busy_day != slot_day and busy_day != "":
                continue

            busy_start = _parse_time(str(busy.get("start", "0:00"))) or 0
            busy_end = _parse_time(str(busy.get("end", "0:00"))) or 0

            # Check for overlap
            if not (slot_end <= busy_start or slot_start >= busy_end):
                return False

    return True


def compute_score_calendar_scheduling(
    solution_str: str,
    ground_truth: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for Calendar Scheduling task.

    Args:
        solution_str: Model's proposed meeting time
        ground_truth: Dict containing:
            - golden_plan: Expected answer (e.g., "Monday, 11:00 - 11:30")
            - attendees: List of attendee info with busy times
            - duration: Meeting duration in hours (0.5 or 1.0)
            - work_hours: Tuple of (start, end) work hours

    Returns:
        Dict with 'score' (0.0 or 1.0), 'acc', 'pred'
    """
    # Parse golden answer
    golden_plan = ground_truth.get("golden_plan", "")
    golden_slot = _parse_calendar_slot(golden_plan)

    if golden_slot is None:
        return {"score": 0.0, "acc": False, "pred": None, "error": "invalid_golden_plan"}

    # Parse model output
    pred_slot = _parse_calendar_slot(solution_str)

    if pred_slot is None:
        return {"score": 0.0, "acc": False, "pred": None, "error": "parse_failed"}

    # Check exact match with golden answer
    exact_match = (
        pred_slot["day"].lower() == golden_slot["day"].lower() and
        pred_slot["start_time"] == golden_slot["start_time"] and
        pred_slot["end_time"] == golden_slot["end_time"]
    )

    if exact_match:
        return {
            "score": 1.0,
            "acc": True,
            "pred": pred_slot,
            "match_type": "exact",
        }

    # If not exact match, check if the proposed slot is valid
    # (This allows for alternative correct answers)
    attendees = ground_truth.get("attendees", [])
    work_hours = ground_truth.get("work_hours", (9.0, 17.0))
    expected_duration = ground_truth.get("duration", 0.5)

    if isinstance(work_hours, str):
        wh_match = re.match(r"(\d+):(\d+)\s*to\s*(\d+):(\d+)", work_hours)
        if wh_match:
            work_hours = (
                int(wh_match.group(1)) + int(wh_match.group(2)) / 60,
                int(wh_match.group(3)) + int(wh_match.group(4)) / 60,
            )
        else:
            work_hours = (9.0, 17.0)

    # Check duration
    pred_duration = pred_slot["end_time"] - pred_slot["start_time"]
    duration_ok = abs(pred_duration - expected_duration) < 0.01

    # Check availability (if attendee info is provided)
    if attendees and duration_ok:
        is_valid = _check_slot_available(pred_slot, attendees, work_hours)
        if is_valid:
            return {
                "score": 1.0,
                "acc": True,
                "pred": pred_slot,
                "match_type": "alternative_valid",
            }

    return {
        "score": 0.0,
        "acc": False,
        "pred": pred_slot,
        "match_type": "mismatch",
        "golden": golden_slot,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    task_type: str = "auto",
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for NaturalPlan benchmark tasks.

    Args:
        solution_str: Model's generated solution
        ground_truth: Ground truth data (format depends on task type)
        task_type: One of 'trip_planning', 'meeting_planning', 'calendar_scheduling', or 'auto'

    Returns:
        Dict containing at minimum:
        - score: float (0.0 to 1.0)
        - acc: bool (whether the answer is correct)
        - pred: parsed prediction
    """
    # Auto-detect task type from ground_truth structure
    if task_type == "auto":
        if "cities" in ground_truth or ("golden_plan" in ground_truth and "**" in str(ground_truth.get("golden_plan", ""))):
            task_type = "trip_planning"
        elif "people" in ground_truth or "distances" in ground_truth:
            task_type = "meeting_planning"
        elif "attendees" in ground_truth or ("golden_plan" in ground_truth and any(day in str(ground_truth.get("golden_plan", "")).lower() for day in ["monday", "tuesday", "wednesday", "thursday", "friday"])):
            task_type = "calendar_scheduling"
        else:
            # Default to calendar scheduling as it has the simplest format
            task_type = "calendar_scheduling"

    if task_type == "trip_planning":
        return compute_score_trip_planning(solution_str, ground_truth, **kwargs)
    elif task_type == "meeting_planning":
        return compute_score_meeting_planning(solution_str, ground_truth, **kwargs)
    elif task_type == "calendar_scheduling":
        return compute_score_calendar_scheduling(solution_str, ground_truth, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
