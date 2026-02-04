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
"""Tests for the structured output reward scoring module."""

import json

import pytest

from verl.utils.reward_score.structured_output import (
    _compute_field_scores,
    _try_extract_json,
    _validate_type,
    compute_score,
    compute_score_binary,
)


class TestTryExtractJson:
    """Tests for JSON extraction from model outputs."""

    def test_plain_json_object(self):
        text = '{"name": "Alice", "age": 30}'
        result = _try_extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'

    def test_json_in_markdown_code_block(self):
        text = 'Here is the output:\n```json\n{"name": "Alice"}\n```'
        result = _try_extract_json(text)
        assert result == '{"name": "Alice"}'

    def test_json_in_generic_code_block(self):
        text = 'Here is the output:\n```\n{"name": "Alice"}\n```'
        result = _try_extract_json(text)
        assert result == '{"name": "Alice"}'

    def test_json_with_surrounding_text(self):
        text = 'The answer is {"name": "Alice", "age": 30} and that is the result.'
        result = _try_extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["name"] == "Alice"

    def test_json_array(self):
        text = '[1, 2, 3]'
        result = _try_extract_json(text)
        assert result == '[1, 2, 3]'

    def test_nested_json(self):
        text = '{"user": {"name": "Alice", "address": {"city": "NYC"}}}'
        result = _try_extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["user"]["address"]["city"] == "NYC"

    def test_no_json(self):
        text = "This is just plain text without any JSON."
        result = _try_extract_json(text)
        assert result is None

    def test_empty_string(self):
        result = _try_extract_json("")
        assert result is None

    def test_reasoning_then_json(self):
        text = """Let me think about this step by step.
The user wants a person object with name and age.
I'll create the following JSON:
{"name": "Bob", "age": 25, "email": "bob@example.com"}"""
        result = _try_extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["name"] == "Bob"


class TestValidateType:
    """Tests for JSON type validation."""

    def test_string(self):
        assert _validate_type("hello", "string") is True
        assert _validate_type(123, "string") is False

    def test_integer(self):
        assert _validate_type(42, "integer") is True
        assert _validate_type(3.14, "integer") is False
        assert _validate_type(True, "integer") is False  # bool is not int for our purposes

    def test_number(self):
        assert _validate_type(42, "number") is True
        assert _validate_type(3.14, "number") is True
        assert _validate_type(True, "number") is False

    def test_boolean(self):
        assert _validate_type(True, "boolean") is True
        assert _validate_type(False, "boolean") is True
        assert _validate_type(1, "boolean") is False

    def test_array(self):
        assert _validate_type([1, 2, 3], "array") is True
        assert _validate_type("not an array", "array") is False

    def test_object(self):
        assert _validate_type({"key": "value"}, "object") is True
        assert _validate_type([1, 2], "object") is False

    def test_null(self):
        assert _validate_type(None, "null") is True
        assert _validate_type("", "null") is False

    def test_unknown_type(self):
        assert _validate_type("anything", "custom") is True


class TestComputeFieldScores:
    """Tests for per-field schema validation scoring."""

    def test_all_fields_present_correct(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = {"name": "Alice", "age": 30}
        scores = _compute_field_scores(data, schema)
        assert scores["name"] == 1.0
        assert scores["age"] == 1.0

    def test_missing_required_field(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = {"name": "Alice"}
        scores = _compute_field_scores(data, schema)
        assert scores["name"] == 1.0
        assert scores["age"] == 0.0

    def test_wrong_type(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
            "required": ["age"],
        }
        data = {"age": "thirty"}
        scores = _compute_field_scores(data, schema)
        assert scores["age"] == 0.0

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                },
            },
            "required": ["user"],
        }
        data = {"user": {"name": "Alice", "email": "alice@example.com"}}
        scores = _compute_field_scores(data, schema)
        assert scores["user.name"] == 1.0
        assert scores["user.email"] == 1.0
        assert scores["user"] == 1.0

    def test_array_with_typed_items(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["tags"],
        }
        data = {"tags": ["python", "ml", "ai"]}
        scores = _compute_field_scores(data, schema)
        assert scores["tags"] == 1.0

    def test_enum_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            },
            "required": ["status"],
        }
        data_valid = {"status": "active"}
        scores = _compute_field_scores(data_valid, schema)
        assert scores["status"] == 1.0

        data_invalid = {"status": "unknown"}
        scores = _compute_field_scores(data_invalid, schema)
        assert scores["status"] == 0.0


class TestComputeScore:
    """Tests for the main compute_score function."""

    def test_valid_json_matching_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        solution = '{"name": "Alice", "age": 30}'
        result = compute_score(solution, schema)

        assert isinstance(result, dict)
        assert result["json_parse_score"] == 1.0
        assert result["field_coverage_score"] == 1.0
        assert result["score"] > 0.0

    def test_invalid_json(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        solution = "this is not json at all"
        result = compute_score(solution, schema)

        assert result["json_parse_score"] == 0.0
        assert result["score"] == 0.0

    def test_valid_json_wrong_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        solution = '{"name": "Alice", "age": "thirty"}'  # age should be integer
        result = compute_score(solution, schema)

        assert result["json_parse_score"] == 1.0
        # Field coverage should be partial since age has wrong type
        assert result["field_coverage_score"] < 1.0

    def test_schema_as_string(self):
        schema_str = json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })
        solution = '{"name": "Alice"}'
        result = compute_score(solution, schema_str)
        assert result["json_parse_score"] == 1.0

    def test_schema_in_ground_truth_dict(self):
        ground_truth = {
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            "answer": '{"name": "Alice"}',
        }
        solution = '{"name": "Alice"}'
        result = compute_score(solution, ground_truth)
        assert result["json_parse_score"] == 1.0
        assert result["content_score"] == 1.0

    def test_schema_in_extra_info(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        extra_info = {"schema_str": json.dumps(schema)}
        solution = '{"name": "Alice"}'
        result = compute_score(solution, {}, extra_info=extra_info)
        assert result["json_parse_score"] == 1.0

    def test_json_in_code_block(self):
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }
        solution = 'Here is the JSON:\n```json\n{"result": "success"}\n```'
        result = compute_score(solution, schema)
        assert result["json_parse_score"] == 1.0

    def test_custom_reward_weights(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        solution = '{"name": "Alice"}'

        # All weight on json_parse
        weights = {"json_parse": 1.0, "schema_valid": 0.0, "field_coverage": 0.0, "content": 0.0}
        result = compute_score(solution, schema, reward_weights=weights)
        assert result["score"] == 1.0

        # All weight on content (no expected answer, so content = schema_valid)
        weights = {"json_parse": 0.0, "schema_valid": 0.0, "field_coverage": 0.0, "content": 1.0}
        result = compute_score(solution, schema, reward_weights=weights)
        assert result["score"] > 0.0

    def test_no_schema_just_json_check(self):
        solution = '{"anything": true}'
        result = compute_score(solution, "not a schema")
        # Should still get credit for valid JSON
        assert result["json_parse_score"] == 1.0


class TestComputeScoreBinary:
    """Tests for binary scoring mode."""

    def test_valid_output(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert compute_score_binary('{"name": "Alice"}', schema) == 1.0

    def test_invalid_json(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        assert compute_score_binary("not json", schema) == 0.0

    def test_json_but_invalid_schema(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        # Missing required field
        assert compute_score_binary('{"age": 30}', schema) < 1.0


class TestDefaultComputeScoreIntegration:
    """Tests that structured_output is properly registered in default_compute_score."""

    def test_structured_output_data_source(self):
        from verl.utils.reward_score import default_compute_score

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = default_compute_score(
            data_source="structured_output",
            solution_str='{"name": "Alice"}',
            ground_truth=schema,
        )
        assert isinstance(result, dict)
        assert result["score"] > 0.0

    def test_json_schema_data_source(self):
        from verl.utils.reward_score import default_compute_score

        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        result = default_compute_score(
            data_source="json_schema",
            solution_str='{"value": 42}',
            ground_truth=schema,
        )
        assert isinstance(result, dict)
        assert result["score"] > 0.0

    def test_nemotron_data_source(self):
        from verl.utils.reward_score import default_compute_score

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = default_compute_score(
            data_source="nvidia/Nemotron-RL-instruction_following-structured_outputs",
            solution_str='{"x": "hello"}',
            ground_truth=schema,
        )
        assert isinstance(result, dict)
