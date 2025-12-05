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

Usage in RLVR:
    - solution_str: Model's generated response
    - ground_truth: Dict containing 'golden_plan' and task metadata
    - extra_info: Optional dict with 'prompt' for context
"""

import re
from typing import Any


# =============================================================================
# Calendar Scheduling Verification (Official NaturalPlan Logic)
# =============================================================================

def _hour_to_num(hr_str: str) -> float:
    """Convert time string like '14:30' to numeric hours (14.5)."""
    match = re.match(r"(\d{1,2}):(\d{2})", hr_str.strip())
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours + minutes / 60.0
    return 0.0


def _parse_calendar_response(response: str) -> tuple[str, float, float]:
    """
    Parse calendar scheduling response using official NaturalPlan regex.

    Pattern: 'Day, HH:MM - HH:MM' (e.g., 'Monday, 14:30 - 15:30')

    Returns:
        Tuple of (day, start_hour, end_hour) or ('', 0, 0) if not found
    """
    # Official NaturalPlan regex pattern
    pattern = r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+'
    match = re.search(pattern, response)

    if not match:
        return '', 0.0, 0.0

    matched_str = match.group()
    parts = matched_str.split(', ')
    if len(parts) != 2:
        return '', 0.0, 0.0

    day = parts[0].lower()
    time_range = parts[1].split(' - ')
    if len(time_range) != 2:
        return '', 0.0, 0.0

    start_hour = _hour_to_num(time_range[0])
    end_hour = _hour_to_num(time_range[1])

    return day, start_hour, end_hour


def compute_score_calendar_scheduling(
    solution_str: str,
    ground_truth: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for Calendar Scheduling task using official NaturalPlan logic.

    Args:
        solution_str: Model's response (e.g., "SOLUTION: Here is the proposed time: Monday, 14:30 - 15:30")
        ground_truth: Dict containing:
            - golden_plan: Expected answer (e.g., "Here is the proposed time: Monday, 14:30 - 15:30 ")

    Returns:
        Dict with 'score' (0.0 or 1.0), 'acc', 'pred'
    """
    golden_plan = ground_truth.get("golden_plan", "")

    # Parse both response and golden solution
    r_day, r_start, r_end = _parse_calendar_response(solution_str)
    s_day, s_start, s_end = _parse_calendar_response(golden_plan)

    # Check if golden plan is valid
    if not s_day:
        return {"score": 0.0, "acc": False, "pred": None, "error": "invalid_golden_plan"}

    # Check if response could be parsed
    if not r_day:
        return {"score": 0.0, "acc": False, "pred": None, "error": "parse_failed"}

    # Exact match comparison (official NaturalPlan logic)
    correct = (r_day == s_day and r_start == s_start and r_end == s_end)

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": {"day": r_day, "start": r_start, "end": r_end},
        "golden": {"day": s_day, "start": s_start, "end": s_end},
    }


# =============================================================================
# Trip Planning Verification (Official NaturalPlan Logic)
# =============================================================================

def _parse_trip_response(response: str) -> list[tuple[str, int]]:
    """
    Parse trip planning response to extract city-duration pairs.

    Looks for patterns like:
    - "Day 1-3: Paris" -> ("paris", 3)
    - "Day 4 from Paris to London" -> flight segment

    Returns:
        List of (city, duration) tuples
    """
    visits = []

    # Pattern: "Day X-Y" or "Days X-Y" followed by city or action
    # Look for stay patterns like "Day 1-3: City" or "Days 1-3 in City"
    pattern = r'[Dd]ays?\s*(\d+)\s*[-–]\s*(\d+)[:\s]+([A-Za-z][A-Za-z\s]*?)(?:\.|,|\n|$|[Dd]ay)'
    matches = re.findall(pattern, response)

    for match in matches:
        start_day = int(match[0])
        end_day = int(match[1])
        city = match[2].strip().lower()
        if city:
            duration = end_day - start_day + 1
            visits.append((city, duration))

    return visits


def compute_score_trip_planning(
    solution_str: str,
    ground_truth: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for Trip Planning task.

    Args:
        solution_str: Model's response with trip itinerary
        ground_truth: Dict containing:
            - cities: List of cities in order
            - durations: List of stay durations
            OR
            - golden_plan: String with expected city**duration format

    Returns:
        Dict with 'score' (0.0 or 1.0), 'acc', 'pred'
    """
    # Parse ground truth
    if "cities" in ground_truth and "durations" in ground_truth:
        gt_cities = [c.lower() for c in ground_truth["cities"]]
        gt_durations = list(ground_truth["durations"])
    elif "golden_plan" in ground_truth:
        # Try to parse golden_plan
        golden = ground_truth["golden_plan"]
        gt_parsed = _parse_trip_response(golden)
        if gt_parsed:
            gt_cities = [p[0] for p in gt_parsed]
            gt_durations = [p[1] for p in gt_parsed]
        else:
            return {"score": 0.0, "acc": False, "pred": None, "error": "invalid_golden_plan"}
    else:
        return {"score": 0.0, "acc": False, "pred": None, "error": "missing_ground_truth"}

    # Parse model response
    parsed = _parse_trip_response(solution_str)

    if not parsed:
        return {"score": 0.0, "acc": False, "pred": None, "error": "parse_failed"}

    pred_cities = [p[0] for p in parsed]
    pred_durations = [p[1] for p in parsed]

    # Sequential matching until first mismatch (official logic)
    num_match = 0
    for i in range(min(len(pred_cities), len(gt_cities))):
        if pred_cities[i] == gt_cities[i] and pred_durations[i] == gt_durations[i]:
            num_match += 1
        else:
            break

    # Exact match: all cities and durations match
    correct = (
        len(pred_cities) == len(gt_cities) and
        num_match == len(gt_cities)
    )

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": {"cities": pred_cities, "durations": pred_durations},
        "golden": {"cities": gt_cities, "durations": gt_durations},
        "num_match": num_match,
    }


# =============================================================================
# Meeting Planning Verification (Official NaturalPlan Logic)
# =============================================================================

def _parse_meeting_response(response: str) -> list[dict[str, Any]]:
    """
    Parse meeting planning response.

    Looks for patterns like:
    - JSON: [{"location": "X", "person_name": "Y", "start_time": "HH:MM"}]
    - Text: "Meet Person at Location at Time"

    Returns:
        List of meeting dicts
    """
    import json

    meetings = []

    # Try JSON format first
    try:
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "person_name" in item:
                        meetings.append({
                            "location": str(item.get("location", "")).lower(),
                            "person_name": str(item.get("person_name", "")).lower(),
                            "start_time": _hour_to_num(str(item.get("start_time", "0:00"))),
                        })
                if meetings:
                    return meetings
    except (json.JSONDecodeError, ValueError):
        pass

    # Text format: "Meet Person at Location at/from Time"
    pattern = r'[Mm]eet\s+(\w+)\s+at\s+([\w\s]+?)\s+(?:at|from)\s+(\d{1,2}:\d{2})'
    matches = re.findall(pattern, response)
    for match in matches:
        meetings.append({
            "person_name": match[0].lower(),
            "location": match[1].strip().lower(),
            "start_time": _hour_to_num(match[2]),
        })

    return meetings


def compute_score_meeting_planning(
    solution_str: str,
    ground_truth: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for Meeting Planning task.

    The goal is to meet as many people as possible. Score is based on
    whether the number of valid meetings matches the golden_score.

    Args:
        solution_str: Model's meeting schedule response
        ground_truth: Dict containing:
            - golden_score: Expected number of meetings
            - people: List of people with availability
            - distances: Travel times between locations
            - meeting_duration: Duration of each meeting

    Returns:
        Dict with 'score', 'acc', 'pred', 'meetings_count'
    """
    golden_score = ground_truth.get("golden_score", 0)
    people = ground_truth.get("people", [])
    distances = ground_truth.get("distances", {})
    meeting_duration = ground_truth.get("meeting_duration", 30) / 60.0
    initial_location = ground_truth.get("initial_location", "").lower()
    initial_time = ground_truth.get("initial_time", 9.0)

    if isinstance(initial_time, str):
        initial_time = _hour_to_num(initial_time)

    # Parse model response
    parsed = _parse_meeting_response(solution_str)

    if not parsed:
        return {"score": 0.0, "acc": False, "pred": None, "meetings_count": 0, "error": "parse_failed"}

    # Build people lookup
    people_lookup = {}
    for p in people:
        name = p["name"].lower()
        people_lookup[name] = {
            "location": p.get("location", "").lower(),
            "start_time": _hour_to_num(str(p.get("start_time", "0:00"))),
            "end_time": _hour_to_num(str(p.get("end_time", "24:00"))),
        }

    # Validate meetings
    current_time = initial_time
    current_location = initial_location
    met_people = set()
    valid_meetings = 0

    for meeting in parsed:
        person = meeting["person_name"]
        location = meeting["location"]
        start_time = meeting["start_time"]

        if person not in people_lookup or person in met_people:
            continue

        info = people_lookup[person]

        # Travel time
        if current_location != location:
            travel = distances.get((current_location, location), 0)
            if isinstance(travel, (int, float)):
                travel = travel / 60.0
            current_time += travel

        # Check constraints
        actual_start = max(current_time, start_time)
        meeting_end = actual_start + meeting_duration

        if (location == info["location"] and
            actual_start >= info["start_time"] and
            meeting_end <= info["end_time"]):
            valid_meetings += 1
            met_people.add(person)
            current_time = meeting_end
            current_location = location

    correct = valid_meetings >= golden_score

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": parsed,
        "meetings_count": valid_meetings,
        "golden_score": golden_score,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    task_type: str = "auto",
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Compute reward for NaturalPlan benchmark tasks.

    This is the main entry point for RLVR. The model generates a response
    to a planning prompt, and this function verifies if the response matches
    the expected answer.

    Args:
        solution_str: Model's generated response
        ground_truth: Dict containing 'golden_plan' or task-specific ground truth
        task_type: One of 'trip_planning', 'meeting_planning', 'calendar_scheduling', or 'auto'
        extra_info: Optional dict with additional context (e.g., 'prompt')

    Returns:
        Dict containing:
        - score: float (0.0 or 1.0)
        - acc: bool (whether the answer is correct)
        - pred: parsed prediction
    """
    # Auto-detect task type from ground_truth structure
    if task_type == "auto":
        golden_plan = str(ground_truth.get("golden_plan", ""))

        # Calendar: contains day patterns like "Monday, 14:30 - 15:30"
        if re.search(r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+', golden_plan):
            task_type = "calendar_scheduling"
        # Trip: contains cities/durations or "Day X-Y" patterns
        elif "cities" in ground_truth or re.search(r'[Dd]ays?\s*\d+\s*[-–]\s*\d+', golden_plan):
            task_type = "trip_planning"
        # Meeting: contains people/distances/golden_score
        elif "people" in ground_truth or "golden_score" in ground_truth:
            task_type = "meeting_planning"
        else:
            # Default to calendar scheduling
            task_type = "calendar_scheduling"

    if task_type == "trip_planning":
        return compute_score_trip_planning(solution_str, ground_truth, **kwargs)
    elif task_type == "meeting_planning":
        return compute_score_meeting_planning(solution_str, ground_truth, **kwargs)
    elif task_type == "calendar_scheduling":
        return compute_score_calendar_scheduling(solution_str, ground_truth, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
