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
Unified Reward Function for Natural Plan Environment

This module provides a single entry point for computing rewards across all
three Natural Plan tasks: Calendar Scheduling, Meeting Planning, and Trip Planning.

The reward function is deterministic and verifiable - it extracts structured
answers from model responses and compares them against ground truth.
"""

from typing import Optional

from recipe.natural_plan.tasks.calendar_scheduling import (
    compute_score as calendar_compute_score,
    parse_calendar_response,
    TimeSlot,
    Participant,
    CalendarSchedulingInstance,
    verify_calendar_solution,
)
from recipe.natural_plan.tasks.meeting_planning import (
    compute_score as meeting_compute_score,
    parse_meeting_response,
    Person,
    MeetingPlanningInstance,
    verify_meeting_plan,
)
from recipe.natural_plan.tasks.trip_planning import (
    compute_score as trip_compute_score,
    parse_trip_response,
    TripConstraint,
    TripPlanningInstance,
    verify_trip_solution,
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: Optional[dict] = None,
    **kwargs
) -> float:
    """
    Compute reward score for Natural Plan tasks.

    This is the main entry point for the reward function, compatible with
    verl's reward manager interface.

    Args:
        data_source: Task identifier, one of:
            - "natural_plan/calendar_scheduling"
            - "natural_plan/meeting_planning"
            - "natural_plan/trip_planning"
        solution_str: Model's response string
        ground_truth: Dict containing task-specific ground truth:
            - Calendar: {"day": str, "start": float, "end": float}
            - Meeting: {"solution_score": int}
            - Trip: {"solution_cities": list, "solution_durations": list}
        extra_info: Optional dict with full instance info for detailed validation

    Returns:
        Float score, typically 0.0 or 1.0 for exact match tasks
    """
    if "calendar" in data_source.lower():
        return calendar_compute_score(solution_str, ground_truth, extra_info)
    elif "meeting" in data_source.lower():
        return meeting_compute_score(solution_str, ground_truth, extra_info)
    elif "trip" in data_source.lower():
        return trip_compute_score(solution_str, ground_truth, extra_info)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: Optional[dict] = None,
    **kwargs
) -> dict:
    """
    Compute reward score with detailed breakdown.

    Returns a dict containing:
        - score: Float reward value
        - parsed: Whether response was successfully parsed
        - valid: Whether response is a valid solution
        - Additional task-specific metrics

    This is useful for debugging and analysis.
    """
    if "calendar" in data_source.lower():
        # Reconstruct instance if extra_info available
        if extra_info and "participants" in extra_info:
            participants = [
                Participant(
                    name=p["name"],
                    available_slots=[],
                    busy_slots=[
                        TimeSlot(s["day"], s["start"], s["end"])
                        for s in p.get("busy_slots", [])
                    ]
                )
                for p in extra_info["participants"]
            ]
            instance = CalendarSchedulingInstance(
                participants=participants,
                num_days=extra_info.get("num_days", 1),
                meeting_duration=ground_truth["end"] - ground_truth["start"],
                solution=TimeSlot(
                    ground_truth["day"],
                    ground_truth["start"],
                    ground_truth["end"]
                )
            )
            return verify_calendar_solution(solution_str, instance)

        # Basic parsing check
        parsed = parse_calendar_response(solution_str)
        if parsed is None:
            return {"score": 0.0, "parsed": False, "error": "Parse failed"}

        score = calendar_compute_score(solution_str, ground_truth, extra_info)
        return {
            "score": score,
            "parsed": True,
            "predicted_day": parsed.day,
            "predicted_start": parsed.start_hour,
            "predicted_end": parsed.end_hour,
            "expected_day": ground_truth["day"],
            "expected_start": ground_truth["start"],
            "expected_end": ground_truth["end"],
        }

    elif "meeting" in data_source.lower():
        if extra_info and "people" in extra_info:
            # Reconstruct instance
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
            distance_matrix = {}
            for key, value in extra_info.get("distance_matrix", {}).items():
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
            return verify_meeting_plan(solution_str, instance)

        # Basic parsing
        parsed = parse_meeting_response(solution_str)
        score = meeting_compute_score(solution_str, ground_truth, extra_info)
        return {
            "score": score,
            "parsed": len(parsed) > 0,
            "num_meetings_parsed": len(parsed),
            "optimal_meetings": ground_truth.get("solution_score", 0),
        }

    elif "trip" in data_source.lower():
        if extra_info and "constraints" in extra_info:
            constraints = [
                TripConstraint(
                    constraint_type=c["type"],
                    city1=c["city1"],
                    city2=c.get("city2"),
                    value=c.get("value")
                )
                for c in extra_info["constraints"]
            ]
            instance = TripPlanningInstance(
                cities=extra_info.get("cities", ground_truth["solution_cities"]),
                durations=extra_info.get("durations", ground_truth["solution_durations"]),
                total_days=sum(ground_truth["solution_durations"]),
                constraints=constraints,
                region=extra_info.get("region", "europe"),
                solution_cities=ground_truth["solution_cities"],
                solution_durations=ground_truth["solution_durations"]
            )
            return verify_trip_solution(solution_str, instance)

        # Basic parsing
        parsed = parse_trip_response(solution_str)
        score = trip_compute_score(solution_str, ground_truth, extra_info)
        result = {
            "score": score,
            "parsed": parsed is not None,
            "expected_cities": ground_truth["solution_cities"],
            "expected_durations": ground_truth["solution_durations"],
        }
        if parsed:
            result["predicted_cities"] = parsed[0]
            result["predicted_durations"] = parsed[1]
        return result

    else:
        raise ValueError(f"Unknown data source: {data_source}")


# Alias for backward compatibility
reward_func = compute_score


if __name__ == "__main__":
    # Test the reward function
    print("Testing Natural Plan Reward Function")
    print("=" * 50)

    # Test Calendar Scheduling
    print("\n1. Calendar Scheduling")
    calendar_gt = {"day": "Monday", "start": 10.0, "end": 11.0}

    test_responses = [
        "The proposed time is Monday, 10:00 - 11:00",  # Correct
        "The proposed time is Monday, 9:00 - 10:00",   # Wrong time
        "The proposed time is Tuesday, 10:00 - 11:00", # Wrong day
        "I think we should meet sometime",             # Invalid format
    ]

    for resp in test_responses:
        score = compute_score("natural_plan/calendar_scheduling", resp, calendar_gt)
        print(f"  Response: {resp[:50]}... -> Score: {score}")

    # Test Meeting Planning
    print("\n2. Meeting Planning")
    meeting_gt = {"solution_score": 2}

    test_responses = [
        "1. Meet Alice at Downtown from 9:00AM to 10:00AM\n2. Meet Bob at Marina from 11:00AM to 12:00PM",
        "Meet Alice at Downtown 9:00AM to 10:00AM",
        "I couldn't schedule any meetings",
    ]

    for resp in test_responses:
        score = compute_score("natural_plan/meeting_planning", resp, meeting_gt)
        print(f"  Response: {resp[:50]}... -> Score: {score}")

    # Test Trip Planning
    print("\n3. Trip Planning")
    trip_gt = {
        "solution_cities": ["Paris", "London", "Rome"],
        "solution_durations": [3, 2, 2]
    }

    test_responses = [
        "Days 1-3: Paris, Days 4-5: London, Days 6-7: Rome",  # Correct
        "Days 1-3: Paris, Days 4-5: Rome, Days 6-7: London",  # Wrong order
        "Paris (2 days), London (2 days), Rome (3 days)",     # Wrong durations
    ]

    for resp in test_responses:
        score = compute_score("natural_plan/trip_planning", resp, trip_gt)
        print(f"  Response: {resp[:50]}... -> Score: {score}")
