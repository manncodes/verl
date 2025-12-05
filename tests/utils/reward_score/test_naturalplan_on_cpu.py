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
Tests for the NaturalPlan verifiable reward functions.
"""

import pytest

from verl.utils.reward_score import default_compute_score, naturalplan


class TestTripPlanning:
    """Tests for Trip Planning verification."""

    def test_correct_trip_plan_day_range_format(self):
        """Test correct trip plan with Day X-Y format."""
        solution = """
        Here is my plan for your 9-day European trip:

        Day 1-3: Paris
        Day 4-5: London
        Day 6-9: Rome
        """
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],
        }
        result = naturalplan.compute_score_trip_planning(solution, ground_truth)
        assert result["score"] == 1.0
        assert result["acc"] is True

    def test_correct_trip_plan_normalized_case(self):
        """Test city name matching is case-insensitive."""
        solution = """
        Day 1-3: paris
        Day 4-5: LONDON
        Day 6-9: Rome
        """
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],
        }
        result = naturalplan.compute_score_trip_planning(solution, ground_truth)
        assert result["score"] == 1.0
        assert result["acc"] is True

    def test_incorrect_trip_plan_wrong_duration(self):
        """Test incorrect trip plan with wrong durations."""
        solution = """
        Day 1-2: Paris
        Day 3-5: London
        Day 6-9: Rome
        """
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],  # Paris should be 3 days, not 2
        }
        result = naturalplan.compute_score_trip_planning(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_incorrect_trip_plan_wrong_city_order(self):
        """Test incorrect trip plan with wrong city order."""
        solution = """
        Day 1-3: London
        Day 4-5: Paris
        Day 6-9: Rome
        """
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],
        }
        result = naturalplan.compute_score_trip_planning(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_partial_match_score(self):
        """Test partial match score calculation."""
        solution = """
        Day 1-3: Paris
        Day 4-5: London
        Day 6-8: Berlin
        """
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],
        }
        result = naturalplan.compute_score_trip_planning(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["partial_score"] == 2 / 3  # First 2 cities match

    def test_parse_failure(self):
        """Test handling of unparseable response."""
        solution = "I don't know how to plan a trip."
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],
        }
        result = naturalplan.compute_score_trip_planning(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["error"] == "parse_failed"

    def test_days_in_city_format(self):
        """Test parsing '3 days in Paris' format."""
        solution = """
        I recommend spending 3 days in Paris, then 2 days in London,
        and finally 4 days in Rome.
        """
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],
        }
        result = naturalplan.compute_score_trip_planning(solution, ground_truth)
        assert result["score"] == 1.0


class TestMeetingPlanning:
    """Tests for Meeting Planning verification."""

    def test_correct_meeting_plan_json_format(self):
        """Test correct meeting plan in JSON format."""
        solution = """
        Here is the optimal meeting schedule:
        [
            {"location": "Coffee Shop", "person_name": "Alice", "start_time": "9:00"},
            {"location": "Library", "person_name": "Bob", "start_time": "11:00"}
        ]
        """
        ground_truth = {
            "people": [
                {"name": "Alice", "location": "coffee shop", "start_time": "9:00", "end_time": "12:00"},
                {"name": "Bob", "location": "library", "start_time": "10:00", "end_time": "14:00"},
            ],
            "distances": {},
            "meeting_duration": 30,
            "golden_score": 2,
            "initial_location": "coffee shop",
            "initial_time": "9:00",
        }
        result = naturalplan.compute_score_meeting_planning(solution, ground_truth)
        assert result["meetings_count"] == 2
        assert result["score"] == 1.0

    def test_partial_meeting_plan(self):
        """Test meeting plan that achieves partial score."""
        solution = """
        [{"location": "Coffee Shop", "person_name": "Alice", "start_time": "9:00"}]
        """
        ground_truth = {
            "people": [
                {"name": "Alice", "location": "coffee shop", "start_time": "9:00", "end_time": "12:00"},
                {"name": "Bob", "location": "library", "start_time": "10:00", "end_time": "14:00"},
            ],
            "distances": {},
            "meeting_duration": 30,
            "golden_score": 2,
            "initial_location": "coffee shop",
            "initial_time": "9:00",
        }
        result = naturalplan.compute_score_meeting_planning(solution, ground_truth)
        assert result["meetings_count"] == 1
        assert result["partial_score"] == 0.5

    def test_meeting_plan_text_format(self):
        """Test meeting plan in text format."""
        solution = """
        Meet Alice at Coffee Shop at 9:00.
        Then meet Bob at Library at 11:00.
        """
        ground_truth = {
            "people": [
                {"name": "Alice", "location": "coffee shop", "start_time": "9:00", "end_time": "12:00"},
                {"name": "Bob", "location": "library", "start_time": "10:00", "end_time": "14:00"},
            ],
            "distances": {},
            "meeting_duration": 30,
            "golden_score": 2,
            "initial_location": "coffee shop",
            "initial_time": "9:00",
        }
        result = naturalplan.compute_score_meeting_planning(solution, ground_truth)
        assert result["meetings_count"] == 2

    def test_meeting_plan_parse_failure(self):
        """Test handling of unparseable meeting plan."""
        solution = "I cannot figure out a schedule."
        ground_truth = {
            "people": [{"name": "Alice", "location": "coffee shop", "start_time": "9:00", "end_time": "12:00"}],
            "distances": {},
            "meeting_duration": 30,
            "golden_score": 1,
            "initial_location": "coffee shop",
            "initial_time": "9:00",
        }
        result = naturalplan.compute_score_meeting_planning(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["error"] == "parse_failed"


class TestCalendarScheduling:
    """Tests for Calendar Scheduling verification."""

    def test_correct_calendar_exact_match(self):
        """Test exact match with golden plan."""
        solution = "Here is the proposed time: Monday, 11:00 - 11:30"
        ground_truth = {
            "golden_plan": "Monday, 11:00 - 11:30",
            "duration": 0.5,
        }
        result = naturalplan.compute_score_calendar_scheduling(solution, ground_truth)
        assert result["score"] == 1.0
        assert result["acc"] is True
        assert result["match_type"] == "exact"

    def test_correct_calendar_hour_meeting(self):
        """Test hour-long meeting scheduling."""
        solution = "The best time for everyone is Tuesday, 14:00 - 15:00"
        ground_truth = {
            "golden_plan": "Tuesday, 14:00 - 15:00",
            "duration": 1.0,
        }
        result = naturalplan.compute_score_calendar_scheduling(solution, ground_truth)
        assert result["score"] == 1.0
        assert result["acc"] is True

    def test_incorrect_calendar_wrong_day(self):
        """Test incorrect day."""
        solution = "Here is the proposed time: Tuesday, 11:00 - 11:30"
        ground_truth = {
            "golden_plan": "Monday, 11:00 - 11:30",
            "duration": 0.5,
        }
        result = naturalplan.compute_score_calendar_scheduling(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_incorrect_calendar_wrong_time(self):
        """Test incorrect time."""
        solution = "Here is the proposed time: Monday, 10:00 - 10:30"
        ground_truth = {
            "golden_plan": "Monday, 11:00 - 11:30",
            "duration": 0.5,
        }
        result = naturalplan.compute_score_calendar_scheduling(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_calendar_alternative_format(self):
        """Test alternative time format (time on day)."""
        solution = "Let's schedule the meeting for 11:00 - 11:30 on Monday"
        ground_truth = {
            "golden_plan": "Monday, 11:00 - 11:30",
            "duration": 0.5,
        }
        result = naturalplan.compute_score_calendar_scheduling(solution, ground_truth)
        assert result["score"] == 1.0

    def test_calendar_parse_failure(self):
        """Test handling of unparseable calendar response."""
        solution = "I'm not sure when everyone is available."
        ground_truth = {
            "golden_plan": "Monday, 11:00 - 11:30",
            "duration": 0.5,
        }
        result = naturalplan.compute_score_calendar_scheduling(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["error"] == "parse_failed"

    def test_calendar_all_weekdays(self):
        """Test all weekdays are recognized."""
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day in weekdays:
            solution = f"Here is the proposed time: {day}, 9:00 - 9:30"
            ground_truth = {"golden_plan": f"{day}, 9:00 - 9:30", "duration": 0.5}
            result = naturalplan.compute_score_calendar_scheduling(solution, ground_truth)
            assert result["score"] == 1.0, f"Failed for {day}"


class TestAutoTaskDetection:
    """Tests for automatic task type detection."""

    def test_auto_detect_trip_planning(self):
        """Test auto-detection of trip planning task."""
        solution = "Day 1-3: Paris"
        ground_truth = {"cities": ["Paris"], "durations": [3]}
        result = naturalplan.compute_score(solution, ground_truth, task_type="auto")
        assert result["score"] == 1.0

    def test_auto_detect_calendar_scheduling(self):
        """Test auto-detection of calendar scheduling task."""
        solution = "Monday, 11:00 - 11:30"
        ground_truth = {"golden_plan": "Monday, 11:00 - 11:30"}
        result = naturalplan.compute_score(solution, ground_truth, task_type="auto")
        assert result["score"] == 1.0


class TestDefaultComputeScore:
    """Tests for integration with default_compute_score dispatcher."""

    def test_naturalplan_trip_planning_data_source(self):
        """Test naturalplan_trip_planning data source."""
        solution = "Day 1-3: Paris\nDay 4-5: London"
        ground_truth = {"cities": ["Paris", "London"], "durations": [3, 2]}
        result = default_compute_score("naturalplan_trip_planning", solution, ground_truth)
        assert result["score"] == 1.0

    def test_naturalplan_calendar_scheduling_data_source(self):
        """Test naturalplan_calendar_scheduling data source."""
        solution = "Wednesday, 10:00 - 10:30"
        ground_truth = {"golden_plan": "Wednesday, 10:00 - 10:30"}
        result = default_compute_score("naturalplan_calendar_scheduling", solution, ground_truth)
        assert result["score"] == 1.0

    def test_naturalplan_generic_data_source(self):
        """Test generic naturalplan data source with auto-detection."""
        solution = "Friday, 15:00 - 16:00"
        ground_truth = {"golden_plan": "Friday, 15:00 - 16:00"}
        result = default_compute_score("naturalplan", solution, ground_truth)
        assert result["score"] == 1.0
