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
Uses the official NaturalPlan evaluation logic.
"""

import pytest

from verl.utils.reward_score import default_compute_score, naturalplan


class TestCalendarScheduling:
    """Tests for Calendar Scheduling verification using official NaturalPlan format."""

    def test_exact_match_hour_meeting(self):
        """Test exact match for 1-hour meeting."""
        # Actual format from NaturalPlan benchmark
        ground_truth = {
            "num_people": "3",
            "num_days": "1",
            "duration": "1",
            "golden_plan": "Here is the proposed time: Monday, 14:30 - 15:30 ",
        }
        solution_str = "SOLUTION: Here is the proposed time: Monday, 14:30 - 15:30"
        result = naturalplan.compute_score_calendar_scheduling(solution_str, ground_truth)
        assert result["score"] == 1.0
        assert result["acc"] is True
        assert result["pred"]["day"] == "monday"
        assert result["pred"]["start"] == 14.5
        assert result["pred"]["end"] == 15.5

    def test_exact_match_half_hour_meeting(self):
        """Test exact match for 30-min meeting."""
        ground_truth = {"golden_plan": "Here is the proposed time: Monday, 11:00 - 11:30"}
        solution_str = "SOLUTION: Here is the proposed time: Monday, 11:00 - 11:30"
        result = naturalplan.compute_score_calendar_scheduling(solution_str, ground_truth)
        assert result["score"] == 1.0
        assert result["acc"] is True

    def test_wrong_day(self):
        """Test mismatch when day is wrong."""
        ground_truth = {"golden_plan": "Monday, 14:30 - 15:30"}
        solution_str = "Tuesday, 14:30 - 15:30"
        result = naturalplan.compute_score_calendar_scheduling(solution_str, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_wrong_start_time(self):
        """Test mismatch when start time is wrong."""
        ground_truth = {"golden_plan": "Monday, 14:30 - 15:30"}
        solution_str = "Monday, 10:00 - 11:00"
        result = naturalplan.compute_score_calendar_scheduling(solution_str, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_wrong_end_time(self):
        """Test mismatch when end time is wrong."""
        ground_truth = {"golden_plan": "Monday, 14:30 - 15:30"}
        solution_str = "Monday, 14:30 - 16:00"
        result = naturalplan.compute_score_calendar_scheduling(solution_str, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_parse_failure(self):
        """Test handling when response cannot be parsed."""
        ground_truth = {"golden_plan": "Monday, 14:30 - 15:30"}
        solution_str = "I cannot find a suitable time for everyone."
        result = naturalplan.compute_score_calendar_scheduling(solution_str, ground_truth)
        assert result["score"] == 0.0
        assert result["error"] == "parse_failed"

    def test_all_weekdays(self):
        """Test all weekdays are recognized."""
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day in weekdays:
            ground_truth = {"golden_plan": f"{day}, 9:00 - 10:00"}
            solution_str = f"SOLUTION: {day}, 9:00 - 10:00"
            result = naturalplan.compute_score_calendar_scheduling(solution_str, ground_truth)
            assert result["score"] == 1.0, f"Failed for {day}"

    def test_time_parsing(self):
        """Test various time formats."""
        test_cases = [
            ("Monday, 9:00 - 9:30", 9.0, 9.5),
            ("Monday, 12:30 - 13:00", 12.5, 13.0),
            ("Monday, 16:00 - 17:00", 16.0, 17.0),
        ]
        for time_str, expected_start, expected_end in test_cases:
            ground_truth = {"golden_plan": time_str}
            result = naturalplan.compute_score_calendar_scheduling(time_str, ground_truth)
            assert result["score"] == 1.0
            assert result["pred"]["start"] == expected_start
            assert result["pred"]["end"] == expected_end


class TestTripPlanning:
    """Tests for Trip Planning verification."""

    def test_correct_trip_plan(self):
        """Test correct trip plan parsing."""
        ground_truth = {
            "cities": ["Paris", "London", "Rome"],
            "durations": [3, 2, 4],
        }
        solution_str = "Day 1-3: Paris\nDay 4-5: London\nDay 6-9: Rome"
        result = naturalplan.compute_score_trip_planning(solution_str, ground_truth)
        assert result["score"] == 1.0
        assert result["acc"] is True

    def test_incorrect_duration(self):
        """Test wrong duration detection."""
        ground_truth = {
            "cities": ["Paris", "London"],
            "durations": [3, 2],
        }
        solution_str = "Day 1-2: Paris\nDay 3-5: London"  # Paris should be 3 days
        result = naturalplan.compute_score_trip_planning(solution_str, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_incorrect_city_order(self):
        """Test wrong city order detection."""
        ground_truth = {
            "cities": ["Paris", "London"],
            "durations": [3, 2],
        }
        solution_str = "Day 1-3: London\nDay 4-5: Paris"  # Wrong order
        result = naturalplan.compute_score_trip_planning(solution_str, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_case_insensitive_cities(self):
        """Test city matching is case insensitive."""
        ground_truth = {
            "cities": ["Paris", "London"],
            "durations": [3, 2],
        }
        solution_str = "Day 1-3: PARIS\nDay 4-5: london"
        result = naturalplan.compute_score_trip_planning(solution_str, ground_truth)
        assert result["score"] == 1.0

    def test_parse_failure(self):
        """Test handling unparseable response."""
        ground_truth = {
            "cities": ["Paris"],
            "durations": [3],
        }
        solution_str = "I don't know how to plan this trip."
        result = naturalplan.compute_score_trip_planning(solution_str, ground_truth)
        assert result["score"] == 0.0
        assert result["error"] == "parse_failed"


class TestMeetingPlanning:
    """Tests for Meeting Planning verification."""

    def test_json_format_meetings(self):
        """Test JSON format meeting parsing."""
        ground_truth = {
            "golden_score": 2,
            "people": [
                {"name": "Alice", "location": "Coffee Shop", "start_time": "9:00", "end_time": "12:00"},
                {"name": "Bob", "location": "Library", "start_time": "10:00", "end_time": "14:00"},
            ],
            "distances": {},
            "meeting_duration": 30,
            "initial_location": "Coffee Shop",
            "initial_time": "9:00",
        }
        solution_str = '[{"location": "Coffee Shop", "person_name": "Alice", "start_time": "9:00"},' \
                       '{"location": "Library", "person_name": "Bob", "start_time": "11:00"}]'
        result = naturalplan.compute_score_meeting_planning(solution_str, ground_truth)
        assert result["meetings_count"] == 2
        assert result["score"] == 1.0

    def test_text_format_meetings(self):
        """Test text format meeting parsing."""
        ground_truth = {
            "golden_score": 1,
            "people": [
                {"name": "Alice", "location": "Coffee Shop", "start_time": "9:00", "end_time": "12:00"},
            ],
            "distances": {},
            "meeting_duration": 30,
            "initial_location": "Coffee Shop",
            "initial_time": "9:00",
        }
        solution_str = "Meet Alice at Coffee Shop at 9:00"
        result = naturalplan.compute_score_meeting_planning(solution_str, ground_truth)
        assert result["meetings_count"] == 1
        assert result["score"] == 1.0

    def test_insufficient_meetings(self):
        """Test when fewer meetings than golden_score."""
        ground_truth = {
            "golden_score": 2,
            "people": [
                {"name": "Alice", "location": "Coffee Shop", "start_time": "9:00", "end_time": "12:00"},
                {"name": "Bob", "location": "Library", "start_time": "10:00", "end_time": "14:00"},
            ],
            "distances": {},
            "meeting_duration": 30,
            "initial_location": "Coffee Shop",
            "initial_time": "9:00",
        }
        solution_str = '[{"location": "Coffee Shop", "person_name": "Alice", "start_time": "9:00"}]'
        result = naturalplan.compute_score_meeting_planning(solution_str, ground_truth)
        assert result["meetings_count"] == 1
        assert result["score"] == 0.0  # Didn't meet golden_score


class TestAutoDetection:
    """Tests for automatic task type detection."""

    def test_auto_detect_calendar(self):
        """Test auto-detection of calendar scheduling task."""
        ground_truth = {"golden_plan": "Monday, 14:30 - 15:30"}
        solution_str = "Monday, 14:30 - 15:30"
        result = naturalplan.compute_score(solution_str, ground_truth, task_type="auto")
        assert result["score"] == 1.0

    def test_auto_detect_trip(self):
        """Test auto-detection of trip planning task."""
        ground_truth = {"cities": ["Paris"], "durations": [3]}
        solution_str = "Day 1-3: Paris"
        result = naturalplan.compute_score(solution_str, ground_truth, task_type="auto")
        assert result["score"] == 1.0

    def test_auto_detect_meeting(self):
        """Test auto-detection of meeting planning task."""
        ground_truth = {
            "golden_score": 1,
            "people": [{"name": "Alice", "location": "Cafe", "start_time": "9:00", "end_time": "12:00"}],
            "meeting_duration": 30,
            "initial_location": "Cafe",
            "initial_time": "9:00",
        }
        solution_str = '[{"location": "Cafe", "person_name": "Alice", "start_time": "9:00"}]'
        result = naturalplan.compute_score(solution_str, ground_truth, task_type="auto")
        assert result["score"] == 1.0


class TestDefaultComputeScore:
    """Tests for integration with default_compute_score dispatcher."""

    def test_naturalplan_calendar_data_source(self):
        """Test naturalplan_calendar_scheduling data source."""
        ground_truth = {"golden_plan": "Wednesday, 10:00 - 10:30"}
        solution_str = "Wednesday, 10:00 - 10:30"
        result = default_compute_score("naturalplan_calendar_scheduling", solution_str, ground_truth)
        assert result["score"] == 1.0

    def test_naturalplan_trip_data_source(self):
        """Test naturalplan_trip_planning data source."""
        ground_truth = {"cities": ["Paris", "London"], "durations": [3, 2]}
        solution_str = "Day 1-3: Paris\nDay 4-5: London"
        result = default_compute_score("naturalplan_trip_planning", solution_str, ground_truth)
        assert result["score"] == 1.0

    def test_naturalplan_generic_data_source(self):
        """Test generic naturalplan data source with auto-detection."""
        ground_truth = {"golden_plan": "Friday, 15:00 - 16:00"}
        solution_str = "Friday, 15:00 - 16:00"
        result = default_compute_score("naturalplan", solution_str, ground_truth)
        assert result["score"] == 1.0
