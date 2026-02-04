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
"""Tests for the structured output (JSON schema) reward function."""

import json

import pytest

from verl.utils.reward_score.structured_output import (
    compute_score,
    extract_json_from_response,
    strictify_schema,
    validate_json_schema,
)


# --- Test schemas ---

SIMPLE_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
)

NESTED_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                },
            },
            "active": {"type": "boolean"},
        },
    }
)

ARRAY_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                },
            },
        },
    }
)

ENUM_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    }
)


# --- Tests for strictify_schema ---


class TestStrictifySchema:
    def test_adds_required_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        strictify_schema(schema)
        assert set(schema["required"]) == {"name", "age"}
        assert schema["additionalProperties"] is False

    def test_nested_strictification(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }
        strictify_schema(schema)
        assert "required" in schema
        assert "required" in schema["properties"]["user"]
        assert schema["properties"]["user"]["additionalProperties"] is False

    def test_handles_non_dict(self):
        # Should not raise
        strictify_schema("not a dict")
        strictify_schema(42)
        strictify_schema(None)

    def test_handles_list(self):
        schema_list = [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "integer"}}},
        ]
        strictify_schema(schema_list)
        assert "required" in schema_list[0]
        assert "required" in schema_list[1]


# --- Tests for extract_json_from_response ---


class TestExtractJsonFromResponse:
    def test_plain_json(self):
        text = '{"name": "Alice", "age": 30}'
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_markdown_code_block(self):
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_markdown_code_block_no_language(self):
        text = '```\n{"name": "Alice", "age": 30}\n```'
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"name": "Alice", "age": 30}\nDone!'
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_json_array(self):
        text = '[{"id": 1}, {"id": 2}]'
        result = extract_json_from_response(text)
        assert json.loads(result) == [{"id": 1}, {"id": 2}]

    def test_whitespace_handling(self):
        text = '  \n  {"name": "Alice"}  \n  '
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Alice"}


# --- Tests for validate_json_schema ---


class TestValidateJsonSchema:
    def test_valid_simple(self):
        response = '{"name": "Alice", "age": 30}'
        assert validate_json_schema(SIMPLE_SCHEMA, response) is True

    def test_missing_required_field_after_strictify(self):
        response = '{"name": "Alice"}'  # missing "age"
        with pytest.raises(Exception):
            validate_json_schema(SIMPLE_SCHEMA, response)

    def test_additional_properties_rejected(self):
        response = '{"name": "Alice", "age": 30, "extra": "field"}'
        with pytest.raises(Exception):
            validate_json_schema(SIMPLE_SCHEMA, response)

    def test_wrong_type(self):
        response = '{"name": "Alice", "age": "thirty"}'  # age should be integer
        with pytest.raises(Exception):
            validate_json_schema(SIMPLE_SCHEMA, response)

    def test_invalid_json(self):
        response = "not json at all"
        with pytest.raises(Exception):
            validate_json_schema(SIMPLE_SCHEMA, response)

    def test_valid_nested(self):
        response = json.dumps(
            {
                "user": {"name": "Alice", "email": "alice@example.com"},
                "active": True,
            }
        )
        assert validate_json_schema(NESTED_SCHEMA, response) is True

    def test_valid_array(self):
        response = json.dumps(
            {
                "items": [
                    {"id": 1, "label": "first"},
                    {"id": 2, "label": "second"},
                ],
            }
        )
        assert validate_json_schema(ARRAY_SCHEMA, response) is True

    def test_valid_enum(self):
        response = '{"status": "active", "priority": 3}'
        assert validate_json_schema(ENUM_SCHEMA, response) is True

    def test_invalid_enum_value(self):
        response = '{"status": "unknown", "priority": 3}'
        with pytest.raises(Exception):
            validate_json_schema(ENUM_SCHEMA, response)


# --- Tests for compute_score ---


class TestComputeScore:
    def test_valid_json_returns_1(self):
        response = '{"name": "Alice", "age": 30}'
        score = compute_score(response, SIMPLE_SCHEMA)
        assert score == 1.0

    def test_invalid_json_returns_0(self):
        response = "not json"
        score = compute_score(response, SIMPLE_SCHEMA)
        assert score == 0.0

    def test_missing_field_returns_0(self):
        response = '{"name": "Alice"}'
        score = compute_score(response, SIMPLE_SCHEMA)
        assert score == 0.0

    def test_extra_field_returns_0(self):
        response = '{"name": "Alice", "age": 30, "extra": true}'
        score = compute_score(response, SIMPLE_SCHEMA)
        assert score == 0.0

    def test_wrong_type_returns_0(self):
        response = '{"name": "Alice", "age": "thirty"}'
        score = compute_score(response, SIMPLE_SCHEMA)
        assert score == 0.0

    def test_markdown_wrapped_json_returns_1(self):
        response = '```json\n{"name": "Alice", "age": 30}\n```'
        score = compute_score(response, SIMPLE_SCHEMA)
        assert score == 1.0

    def test_json_with_preamble_returns_1(self):
        response = 'Here is the output:\n{"name": "Alice", "age": 30}'
        score = compute_score(response, SIMPLE_SCHEMA)
        assert score == 1.0

    def test_empty_response_returns_0(self):
        score = compute_score("", SIMPLE_SCHEMA)
        assert score == 0.0

    def test_extra_info_schema_type(self):
        response = '{"name": "Alice", "age": 30}'
        # json schema_type should work
        score = compute_score(response, SIMPLE_SCHEMA, extra_info={"schema_type": "json"})
        assert score == 1.0

    def test_unsupported_schema_type_returns_0(self):
        response = '{"name": "Alice", "age": 30}'
        score = compute_score(response, SIMPLE_SCHEMA, extra_info={"schema_type": "xml"})
        assert score == 0.0

    def test_complex_schema_valid(self):
        """Test with a more complex schema similar to the Nemotron dataset."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "activityName": {"type": "string", "minLength": 1},
                    "date": {"type": "string", "format": "date"},
                    "participants": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer", "minimum": 0},
                                "role": {"type": "string", "enum": ["parent", "child", "grandparent"]},
                            },
                        },
                    },
                    "durationHours": {"type": "number", "minimum": 0.5, "maximum": 12},
                    "completed": {"type": "boolean"},
                },
            }
        )
        response = json.dumps(
            {
                "activityName": "Park Cleanup Day",
                "date": "2023-09-16",
                "participants": [
                    {"name": "Maria", "age": 38, "role": "parent"},
                    {"name": "James", "age": 14, "role": "child"},
                ],
                "durationHours": 3,
                "completed": False,
            }
        )
        score = compute_score(response, schema)
        assert score == 1.0

    def test_complex_schema_invalid_enum(self):
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["active", "inactive"]},
                },
            }
        )
        response = '{"status": "unknown"}'
        score = compute_score(response, schema)
        assert score == 0.0


# --- Tests for default_compute_score integration ---


class TestDefaultComputeScoreIntegration:
    def test_nemotron_data_source(self):
        from verl.utils.reward_score import default_compute_score

        response = '{"name": "Alice", "age": 30}'
        score = default_compute_score(
            data_source="nvidia/Nemotron-RL-instruction_following-structured_outputs",
            solution_str=response,
            ground_truth=SIMPLE_SCHEMA,
            extra_info={"schema_type": "json"},
        )
        assert score == 1.0

    def test_structured_outputs_data_source(self):
        from verl.utils.reward_score import default_compute_score

        response = '{"name": "Alice", "age": 30}'
        score = default_compute_score(
            data_source="structured_outputs",
            solution_str=response,
            ground_truth=SIMPLE_SCHEMA,
            extra_info={"schema_type": "json"},
        )
        assert score == 1.0

    def test_invalid_returns_0(self):
        from verl.utils.reward_score import default_compute_score

        score = default_compute_score(
            data_source="structured_outputs",
            solution_str="not json",
            ground_truth=SIMPLE_SCHEMA,
        )
        assert score == 0.0
