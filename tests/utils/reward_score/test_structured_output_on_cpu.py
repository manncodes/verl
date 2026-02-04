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
"""Tests for the structured output (JSON schema) reward function.

These tests simulate real-world LLM responses during RLVR training, including:
- Models that produce clean JSON (guided decoding or well-trained)
- Models that wrap JSON in markdown code blocks
- Models that include <think> reasoning blocks (DeepSeek-R1 style)
- Models that produce plain text refusals or explanations
- Models that produce truncated JSON (hit max_tokens)
- Models that produce wrong schema structure
- Models that produce partial or malformed JSON
"""

import json

import pytest

from verl.utils.reward_score.structured_output import (
    compute_score,
    extract_json_from_response,
    strictify_schema,
    validate_json_schema,
)


# ---------------------------------------------------------------------------
# Schemas taken from the actual Nemotron-RL-structured_outputs dataset
# ---------------------------------------------------------------------------

# Simple 2-field schema (easy)
SIMPLE_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
)

# Nemotron-style family volunteer activity schema (complex, deeply nested)
NEMOTRON_ACTIVITY_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "activityName": {"type": "string", "minLength": 1, "description": "Name of the volunteer activity."},
            "organization": {
                "type": "object",
                "required": ["name", "contact", "address"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "contact": {
                        "type": "object",
                        "required": ["person", "phone", "email"],
                        "additionalProperties": False,
                        "properties": {
                            "person": {"type": "string", "minLength": 1},
                            "phone": {"type": "string", "pattern": r"^\+?[1-9]\d{1,14}$"},
                            "email": {"type": "string", "format": "email"},
                        },
                    },
                    "address": {
                        "type": "object",
                        "required": ["street", "city", "state", "zipCode", "country"],
                        "additionalProperties": False,
                        "properties": {
                            "street": {"type": "string", "minLength": 1},
                            "city": {"type": "string", "minLength": 1},
                            "state": {"type": "string", "minLength": 1},
                            "zipCode": {"type": "string", "pattern": r"^\d{5}(-\d{4})?$|^\w\d\w ?\d\w\d$"},
                            "country": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
            "date": {"type": "string", "format": "date", "description": "Planned date of the volunteer activity."},
            "location": {
                "type": "object",
                "required": ["venue", "coordinates"],
                "additionalProperties": False,
                "properties": {
                    "venue": {"type": "string", "minLength": 1},
                    "coordinates": {
                        "type": "object",
                        "required": ["latitude", "longitude"],
                        "additionalProperties": False,
                        "properties": {
                            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                            "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                        },
                    },
                },
            },
            "familyParticipants": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name", "age", "role", "availability"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "age": {"type": "integer", "minimum": 0},
                        "role": {
                            "type": "string",
                            "enum": ["parent", "child", "grandparent", "guardian", "teenager"],
                        },
                        "availability": {"type": "boolean"},
                        "skills": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                    },
                },
            },
            "activityType": {
                "type": "string",
                "enum": [
                    "communityCleanup",
                    "foodDistribution",
                    "elderlyCompanionship",
                    "tutoring",
                    "animalCare",
                    "eventSupport",
                ],
            },
            "durationHours": {"type": "number", "minimum": 0.5, "maximum": 12},
            "preparationTasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["task", "assignedTo", "dueDate", "completed"],
                    "additionalProperties": False,
                    "properties": {
                        "task": {"type": "string", "minLength": 1},
                        "assignedTo": {"type": "string", "minLength": 1},
                        "dueDate": {"type": "string", "format": "date"},
                        "completed": {"type": "boolean"},
                    },
                },
            },
            "suppliesNeeded": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["item", "quantity", "providedBy"],
                    "additionalProperties": False,
                    "properties": {
                        "item": {"type": "string", "minLength": 1},
                        "quantity": {"type": "integer", "minimum": 1},
                        "providedBy": {"type": "string", "enum": ["family", "organization", "shared", "purchased"]},
                    },
                },
            },
            "specialConsiderations": {
                "type": "object",
                "required": ["allergies", "mobilityIssues", "childFriendly", "weatherDependent"],
                "additionalProperties": False,
                "properties": {
                    "allergies": {"type": "array", "items": {"type": "string"}},
                    "mobilityIssues": {"type": "boolean"},
                    "childFriendly": {"type": "boolean"},
                    "weatherDependent": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
            },
            "completed": {"type": "boolean", "description": "Indicates if the activity has been completed."},
            "feedback": {
                "type": "object",
                "required": ["familyRating", "organizationFeedback", "lessonsLearned"],
                "additionalProperties": False,
                "properties": {
                    "familyRating": {"type": "integer", "minimum": 1, "maximum": 5},
                    "organizationFeedback": {"type": "string"},
                    "lessonsLearned": {"type": "array", "items": {"type": "string"}},
                    "photosTaken": {"type": "boolean"},
                },
            },
        },
    }
)

# The correct response for the Nemotron activity schema (from dataset client.py)
NEMOTRON_VALID_RESPONSE = json.dumps(
    {
        "activityName": "Park Cleanup Day",
        "organization": {
            "name": "Green Earth Organization",
            "contact": {
                "person": "Jane Miller",
                "phone": "+15551234567",
                "email": "jane.miller@greenearth.org",
            },
            "address": {
                "street": "123 Eco Street",
                "city": "Springfield",
                "state": "IL",
                "zipCode": "62701",
                "country": "United States",
            },
        },
        "date": "2023-09-16",
        "location": {
            "venue": "Riverside Park",
            "coordinates": {"latitude": 39.7817, "longitude": -89.6501},
        },
        "familyParticipants": [
            {
                "name": "Maria Lopez",
                "age": 38,
                "role": "parent",
                "availability": True,
                "skills": ["gardening", "first aid"],
            },
            {
                "name": "James Lopez",
                "age": 14,
                "role": "teenager",
                "availability": True,
                "skills": ["basic cleanup"],
            },
            {
                "name": "Elena Lopez",
                "age": 68,
                "role": "grandparent",
                "availability": True,
                "skills": ["community organizing"],
            },
        ],
        "activityType": "communityCleanup",
        "durationHours": 3,
        "preparationTasks": [
            {
                "task": "Confirm family attendance",
                "assignedTo": "Maria Lopez",
                "dueDate": "2023-09-10",
                "completed": False,
            },
            {
                "task": "Pack necessary supplies",
                "assignedTo": "James Lopez",
                "dueDate": "2023-09-15",
                "completed": False,
            },
            {
                "task": "Review safety guidelines",
                "assignedTo": "Maria Lopez",
                "dueDate": "2023-09-10",
                "completed": True,
            },
        ],
        "suppliesNeeded": [
            {"item": "trash grabber", "quantity": 5, "providedBy": "organization"},
            {"item": "garbage bag", "quantity": 10, "providedBy": "family"},
            {"item": "first aid kit", "quantity": 2, "providedBy": "shared"},
        ],
        "specialConsiderations": {
            "allergies": [],
            "mobilityIssues": False,
            "childFriendly": True,
            "weatherDependent": True,
            "notes": "Monitor weather forecasts. Event may be canceled due to heavy rain.",
        },
        "completed": False,
        "feedback": {
            "familyRating": 1,
            "organizationFeedback": "",
            "lessonsLearned": [],
            "photosTaken": False,
        },
    },
    ensure_ascii=False,
)

# Nemotron-style craft project schema (from dataset client.py schema_str field)
NEMOTRON_CRAFT_SCHEMA = json.dumps(
    {
        "type": "object",
        "required": [
            "projectId",
            "projectName",
            "category",
            "materials",
            "steps",
            "estimatedDuration",
            "difficultyLevel",
            "isCompleted",
            "creator",
            "lastModified",
        ],
        "properties": {
            "projectId": {"type": "string", "description": "Unique identifier for the craft project"},
            "projectName": {"type": "string", "description": "Name of the craft project"},
            "category": {
                "type": "string",
                "enum": ["knitting", "painting", "scrapbooking", "woodworking", "origami", "jewelry"],
            },
            "materials": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "quantity", "unit", "isPurchased"],
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": {"type": "number", "minimum": 0},
                        "unit": {"type": "string", "enum": ["piece", "meter", "gram", "ounce", "liter", "unit"]},
                        "isPurchased": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["stepNumber", "instruction", "estimatedTimeMinutes"],
                    "properties": {
                        "stepNumber": {"type": "integer", "minimum": 1},
                        "instruction": {"type": "string"},
                        "estimatedTimeMinutes": {"type": "integer", "minimum": 1},
                    },
                    "additionalProperties": False,
                },
            },
            "estimatedDuration": {"type": "integer"},
            "difficultyLevel": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
            "isCompleted": {"type": "boolean"},
            "creator": {
                "type": "object",
                "required": ["name", "contactEmail"],
                "properties": {
                    "name": {"type": "string"},
                    "contactEmail": {"type": "string", "format": "email"},
                },
                "additionalProperties": False,
            },
            "lastModified": {"type": "string", "format": "date-time"},
        },
        "additionalProperties": False,
    }
)


# ============================================================================
# Unit tests: strictify_schema
# ============================================================================


class TestStrictifySchema:
    def test_adds_required_and_disallows_extras(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        strictify_schema(schema)
        assert set(schema["required"]) == {"name", "age"}
        assert schema["additionalProperties"] is False

    def test_nested_objects_strictified_recursively(self):
        schema = {
            "type": "object",
            "properties": {"user": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
        strictify_schema(schema)
        assert schema["properties"]["user"]["additionalProperties"] is False
        assert schema["properties"]["user"]["required"] == ["name"]

    def test_array_items_strictified(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
                }
            },
        }
        strictify_schema(schema)
        assert schema["properties"]["items"]["items"]["required"] == ["id"]

    def test_safe_on_non_dict_types(self):
        strictify_schema("hello")
        strictify_schema(42)
        strictify_schema(None)
        strictify_schema([1, 2, 3])


# ============================================================================
# Unit tests: extract_json_from_response
# ============================================================================


class TestExtractJsonFromResponse:
    def test_plain_json_object(self):
        result = extract_json_from_response('{"name": "Alice", "age": 30}')
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_plain_json_array(self):
        result = extract_json_from_response('[{"id": 1}, {"id": 2}]')
        assert json.loads(result) == [{"id": 1}, {"id": 2}]

    def test_markdown_json_code_block(self):
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        assert json.loads(extract_json_from_response(text)) == {"name": "Alice", "age": 30}

    def test_markdown_code_block_no_language_tag(self):
        text = '```\n{"name": "Alice", "age": 30}\n```'
        assert json.loads(extract_json_from_response(text)) == {"name": "Alice", "age": 30}

    def test_think_tags_stripped(self):
        text = (
            "<think>\nLet me analyze the document and extract the relevant information "
            "to fill the JSON schema...\nThe activity is a park cleanup.\n</think>\n"
            '{"name": "Park Cleanup", "age": 5}'
        )
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Park Cleanup", "age": 5}

    def test_think_tags_with_json_inside_think_block(self):
        """The model might mention JSON inside its reasoning. We should ignore that."""
        text = (
            '<think>\nThe schema wants {"name": "string"}. Let me construct it.\n</think>\n'
            '{"name": "Alice", "age": 30}'
        )
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_preamble_text_before_json(self):
        text = "Here is the extracted information formatted as JSON:\n\n" '{"name": "Alice", "age": 30}'
        assert json.loads(extract_json_from_response(text)) == {"name": "Alice", "age": 30}

    def test_postamble_text_after_json(self):
        text = '{"name": "Alice", "age": 30}\n\nI hope this helps! Let me know if you need changes.'
        assert json.loads(extract_json_from_response(text)) == {"name": "Alice", "age": 30}

    def test_json_surrounded_by_explanation(self):
        text = (
            "Based on the document provided, I've extracted the following structured data:\n\n"
            '{"name": "Alice", "age": 30}\n\n'
            "Note: The age was inferred from the text mentioning she is 30 years old."
        )
        assert json.loads(extract_json_from_response(text)) == {"name": "Alice", "age": 30}

    def test_multiple_code_blocks_picks_valid_json(self):
        """If model shows schema then answer, pick the last valid JSON block."""
        text = (
            "Here's the schema I'll follow:\n"
            '```json\n{"type": "object"}\n```\n\n'
            "And here's my answer:\n"
            '```json\n{"name": "Alice", "age": 30}\n```'
        )
        result = extract_json_from_response(text)
        assert json.loads(result) == {"name": "Alice", "age": 30}


# ============================================================================
# Real-world LLM response patterns: compute_score
# ============================================================================


class TestRealWorldResponses:
    """Test compute_score with responses that models actually produce during RLVR rollouts."""

    # --- Responses that SHOULD score 1.0 ---

    def test_clean_json_from_guided_decoding(self):
        """vLLM/SGLang with guided_json produces perfectly formatted JSON."""
        response = '{"name":"Alice","age":30}'
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_pretty_printed_json(self):
        """Some models produce indented JSON."""
        response = '{\n  "name": "Alice",\n  "age": 30\n}'
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_json_in_markdown_block(self):
        """Common pattern: model wraps answer in ```json blocks."""
        response = '```json\n{"name": "Alice", "age": 30}\n```'
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_r1_style_think_then_json(self):
        """DeepSeek-R1 style: <think> reasoning followed by JSON answer."""
        response = (
            "<think>\nI need to extract the name and age from the document.\n"
            "The person mentioned is Alice who is 30 years old.\n"
            "Let me format this as JSON.\n</think>\n"
            '{"name": "Alice", "age": 30}'
        )
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_r1_style_think_then_markdown_json(self):
        """DeepSeek-R1 reasoning followed by markdown-wrapped JSON."""
        response = (
            "<think>\nAnalyzing the document...\nFound name=Alice, age=30\n</think>\n\n"
            "```json\n"
            '{"name": "Alice", "age": 30}\n'
            "```"
        )
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_preamble_then_json(self):
        """Model explains what it's doing then gives JSON."""
        response = (
            "Based on the information provided in the document, "
            "here is the structured JSON output:\n\n"
            '{"name": "Alice", "age": 30}'
        )
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_nemotron_activity_full_valid_response(self):
        """Full Nemotron dataset example: complex nested schema with valid response."""
        assert compute_score(NEMOTRON_VALID_RESPONSE, NEMOTRON_ACTIVITY_SCHEMA) == 1.0

    def test_nemotron_activity_in_markdown_block(self):
        """Nemotron response wrapped in markdown code block."""
        response = f"```json\n{NEMOTRON_VALID_RESPONSE}\n```"
        assert compute_score(response, NEMOTRON_ACTIVITY_SCHEMA) == 1.0

    def test_nemotron_craft_project_valid(self):
        """Valid response for the Nemotron craft project schema."""
        response = json.dumps(
            {
                "projectId": "PROJ-001",
                "projectName": "Knitted Winter Scarf",
                "category": "knitting",
                "materials": [
                    {"name": "Wool yarn", "quantity": 3, "unit": "unit", "isPurchased": True},
                    {"name": "Knitting needles", "quantity": 2, "unit": "piece", "isPurchased": True},
                ],
                "steps": [
                    {"stepNumber": 1, "instruction": "Cast on 40 stitches", "estimatedTimeMinutes": 10},
                    {"stepNumber": 2, "instruction": "Knit in ribbing pattern for 150 rows", "estimatedTimeMinutes": 180},
                    {"stepNumber": 3, "instruction": "Cast off and weave in ends", "estimatedTimeMinutes": 15},
                ],
                "estimatedDuration": 205,
                "difficultyLevel": "beginner",
                "isCompleted": False,
                "creator": {"name": "Sarah Chen", "contactEmail": "sarah@crafts.com"},
                "lastModified": "2024-01-15T10:30:00Z",
            }
        )
        assert compute_score(response, NEMOTRON_CRAFT_SCHEMA) == 1.0

    def test_unicode_content_in_json(self):
        """JSON with non-ASCII characters (common in multilingual tasks)."""
        response = '{"name": "\u5c0f\u660e (Xiao Ming)", "age": 25}'
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    # --- Responses that SHOULD score 0.0 ---

    def test_plain_text_refusal(self):
        """Model refuses the task with a plain string response."""
        response = "I'm sorry, I cannot generate structured data from the given document."
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_plain_text_explanation(self):
        """Model explains the schema instead of producing JSON."""
        response = (
            "The schema requires an object with two fields: 'name' (string) and 'age' (integer). "
            "Based on the document, the person's name is Alice and she is 30 years old."
        )
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_conversational_response(self):
        """Model responds conversationally instead of producing JSON."""
        response = (
            "Sure! I'd be happy to help extract the information.\n\n"
            "From the document, I can see that:\n"
            "- Name: Alice\n"
            "- Age: 30\n\n"
            "Would you like me to format this differently?"
        )
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_empty_response(self):
        """Model produces empty output (e.g., hit EOS immediately)."""
        assert compute_score("", SIMPLE_SCHEMA) == 0.0

    def test_whitespace_only_response(self):
        """Model produces only whitespace."""
        assert compute_score("   \n\n\t  ", SIMPLE_SCHEMA) == 0.0

    def test_truncated_json_max_tokens(self):
        """Model hit max_tokens mid-JSON (very common during training)."""
        response = '{"name": "Alice", "age": 30, "address": {"street": "123 Main St", "ci'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_truncated_json_mid_string(self):
        """JSON cut off in the middle of a string value."""
        response = '{"name": "Ali'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_truncated_json_mid_array(self):
        """JSON array cut off mid-element."""
        response = (
            '{"activityName": "Park Cleanup Day", "familyParticipants": ['
            '{"name": "Maria Lopez", "age": 38, "role": "parent"'
        )
        assert compute_score(response, NEMOTRON_ACTIVITY_SCHEMA) == 0.0

    def test_wrong_type_string_for_integer(self):
        """Model puts a string where an integer is expected."""
        response = '{"name": "Alice", "age": "thirty"}'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_wrong_type_string_for_boolean(self):
        """Model puts string 'true' instead of boolean true."""
        schema = json.dumps({"type": "object", "properties": {"active": {"type": "boolean"}}})
        response = '{"active": "true"}'
        assert compute_score(response, schema) == 0.0

    def test_wrong_type_float_for_integer(self):
        """Model puts a float where integer is required."""
        response = '{"name": "Alice", "age": 30.5}'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_missing_required_field(self):
        """Schema strictification makes all fields required."""
        response = '{"name": "Alice"}'  # missing 'age'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_extra_field_not_in_schema(self):
        """strictify sets additionalProperties=false, so extra fields are rejected."""
        response = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_invalid_enum_value(self):
        """Model halluccinates an enum value not in the schema."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "activityType": {
                        "type": "string",
                        "enum": ["communityCleanup", "foodDistribution", "tutoring"],
                    }
                },
            }
        )
        response = '{"activityType": "parkCleanup"}'  # not in enum
        assert compute_score(response, schema) == 0.0

    def test_null_where_object_expected(self):
        """Model puts null for a required object field."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "user": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "active": {"type": "boolean"},
                },
            }
        )
        response = '{"user": null, "active": true}'
        assert compute_score(response, schema) == 0.0

    def test_array_where_object_expected(self):
        """Model confuses array and object types."""
        response = '[{"name": "Alice", "age": 30}]'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_json_with_trailing_comma(self):
        """Invalid JSON: trailing comma (common model mistake)."""
        response = '{"name": "Alice", "age": 30,}'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_json_with_single_quotes(self):
        """Invalid JSON: single quotes instead of double quotes."""
        response = "{'name': 'Alice', 'age': 30}"
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_json_with_comments(self):
        """Invalid JSON: JavaScript-style comments."""
        response = '{"name": "Alice", // this is the name\n"age": 30}'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_yaml_instead_of_json(self):
        """Model produces YAML instead of JSON."""
        response = "name: Alice\nage: 30"
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_xml_instead_of_json(self):
        """Model produces XML instead of JSON."""
        response = "<person><name>Alice</name><age>30</age></person>"
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_python_dict_instead_of_json(self):
        """Model produces Python dict literal (True/False/None instead of true/false/null)."""
        response = "{'name': 'Alice', 'age': 30, 'active': True}"
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_think_tags_only_no_answer(self):
        """Model produces reasoning but never outputs an answer."""
        response = (
            "<think>\nLet me analyze the document...\n"
            "The person is Alice, age 30.\n"
            "I should format this as JSON.\n"
            "The schema requires name and age fields.\n</think>"
        )
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_think_tags_with_truncated_json_answer(self):
        """Model reasons then starts JSON but gets truncated."""
        response = (
            "<think>\nExtracting information...\n</think>\n"
            '{"name": "Alice", "age": 3'
        )
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_nemotron_activity_missing_nested_field(self):
        """Nemotron schema: missing a required nested field (contact.email)."""
        response = json.dumps(
            {
                "activityName": "Park Cleanup Day",
                "organization": {
                    "name": "Green Earth Organization",
                    "contact": {
                        "person": "Jane Miller",
                        "phone": "+15551234567",
                        # missing 'email'
                    },
                    "address": {
                        "street": "123 Eco Street",
                        "city": "Springfield",
                        "state": "IL",
                        "zipCode": "62701",
                        "country": "United States",
                    },
                },
                "date": "2023-09-16",
                "location": {"venue": "Riverside Park", "coordinates": {"latitude": 39.7817, "longitude": -89.6501}},
                "familyParticipants": [
                    {"name": "Maria", "age": 38, "role": "parent", "availability": True}
                ],
                "activityType": "communityCleanup",
                "durationHours": 3,
                "preparationTasks": [],
                "suppliesNeeded": [],
                "specialConsiderations": {
                    "allergies": [],
                    "mobilityIssues": False,
                    "childFriendly": True,
                    "weatherDependent": True,
                },
                "completed": False,
                "feedback": {"familyRating": 3, "organizationFeedback": "", "lessonsLearned": []},
            }
        )
        assert compute_score(response, NEMOTRON_ACTIVITY_SCHEMA) == 0.0

    def test_nemotron_activity_wrong_enum(self):
        """Nemotron schema: model uses an invalid role enum value."""
        response = json.dumps(
            {
                "activityName": "Park Cleanup",
                "organization": {
                    "name": "Green Earth",
                    "contact": {"person": "Jane", "phone": "+15551234567", "email": "j@g.org"},
                    "address": {
                        "street": "123 Eco St",
                        "city": "Springfield",
                        "state": "IL",
                        "zipCode": "62701",
                        "country": "US",
                    },
                },
                "date": "2023-09-16",
                "location": {"venue": "Park", "coordinates": {"latitude": 39.78, "longitude": -89.65}},
                "familyParticipants": [
                    {"name": "Maria", "age": 38, "role": "mother", "availability": True}  # "mother" not in enum
                ],
                "activityType": "communityCleanup",
                "durationHours": 3,
                "preparationTasks": [],
                "suppliesNeeded": [],
                "specialConsiderations": {
                    "allergies": [],
                    "mobilityIssues": False,
                    "childFriendly": True,
                    "weatherDependent": True,
                },
                "completed": False,
                "feedback": {"familyRating": 3, "organizationFeedback": "", "lessonsLearned": []},
            }
        )
        assert compute_score(response, NEMOTRON_ACTIVITY_SCHEMA) == 0.0

    def test_nemotron_craft_missing_required_top_level(self):
        """Nemotron craft schema: missing a required top-level field."""
        response = json.dumps(
            {
                "projectId": "PROJ-001",
                "projectName": "Scarf",
                "category": "knitting",
                "materials": [],
                "steps": [],
                # missing: estimatedDuration, difficultyLevel, isCompleted, creator, lastModified
            }
        )
        assert compute_score(response, NEMOTRON_CRAFT_SCHEMA) == 0.0

    def test_nemotron_craft_invalid_category_enum(self):
        """Nemotron craft schema: invalid category not in enum."""
        response = json.dumps(
            {
                "projectId": "PROJ-001",
                "projectName": "Ceramic Vase",
                "category": "pottery",  # not in enum
                "materials": [],
                "steps": [],
                "estimatedDuration": 120,
                "difficultyLevel": "intermediate",
                "isCompleted": False,
                "creator": {"name": "Bob", "contactEmail": "bob@crafts.com"},
                "lastModified": "2024-01-15T10:30:00Z",
            }
        )
        assert compute_score(response, NEMOTRON_CRAFT_SCHEMA) == 0.0

    def test_number_out_of_range(self):
        """Schema has min/max constraints on a number field."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "durationHours": {"type": "number", "minimum": 0.5, "maximum": 12},
                },
            }
        )
        response = '{"durationHours": 24}'  # exceeds maximum
        assert compute_score(response, schema) == 0.0

    def test_empty_string_violates_minlength(self):
        """Schema requires minLength: 1 but model gives empty string."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string", "minLength": 1}},
            }
        )
        response = '{"name": ""}'
        assert compute_score(response, schema) == 0.0

    def test_unsupported_schema_type(self):
        """extra_info specifies a non-JSON schema type."""
        response = '{"name": "Alice", "age": 30}'
        assert compute_score(response, SIMPLE_SCHEMA, extra_info={"schema_type": "xml"}) == 0.0
        assert compute_score(response, SIMPLE_SCHEMA, extra_info={"schema_type": "yaml"}) == 0.0

    def test_json_schema_type_works(self):
        """Explicitly specifying schema_type='json' works."""
        response = '{"name": "Alice", "age": 30}'
        assert compute_score(response, SIMPLE_SCHEMA, extra_info={"schema_type": "json"}) == 1.0


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_empty_object_matches_empty_schema(self):
        """Empty object matches a schema with no properties."""
        schema = json.dumps({"type": "object", "properties": {}})
        assert compute_score("{}", schema) == 1.0

    def test_empty_object_fails_schema_with_required(self):
        """Empty object fails if schema has properties (strictified to required)."""
        assert compute_score("{}", SIMPLE_SCHEMA) == 0.0

    def test_deeply_nested_valid_json(self):
        """5 levels of nesting (max supported by OpenAI structured outputs)."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "l1": {
                        "type": "object",
                        "properties": {
                            "l2": {
                                "type": "object",
                                "properties": {
                                    "l3": {
                                        "type": "object",
                                        "properties": {
                                            "l4": {
                                                "type": "object",
                                                "properties": {"value": {"type": "string"}},
                                            }
                                        },
                                    }
                                },
                            }
                        },
                    }
                },
            }
        )
        response = '{"l1": {"l2": {"l3": {"l4": {"value": "deep"}}}}}'
        assert compute_score(response, schema) == 1.0

    def test_large_array_valid(self):
        """Schema with array containing many items."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    }
                },
            }
        )
        items = [{"id": i} for i in range(100)]
        response = json.dumps({"items": items})
        assert compute_score(response, schema) == 1.0

    def test_special_characters_in_strings(self):
        """JSON with special characters (newlines, tabs, quotes, backslashes)."""
        response = '{"name": "Alice \\"Bob\\" Smith\\nJr.", "age": 30}'
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_very_long_response_valid(self):
        """Response with very long string values (simulating verbose model output)."""
        long_name = "A" * 10000
        response = json.dumps({"name": long_name, "age": 30})
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_json_with_null_values(self):
        """JSON with null values where string/integer expected."""
        response = '{"name": null, "age": null}'
        assert compute_score(response, SIMPLE_SCHEMA) == 0.0

    def test_negative_integer(self):
        """Negative integer should be valid if no minimum constraint."""
        response = '{"name": "Alice", "age": -1}'
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_zero_integer(self):
        """Zero is a valid integer."""
        response = '{"name": "Baby", "age": 0}'
        assert compute_score(response, SIMPLE_SCHEMA) == 1.0

    def test_response_is_just_the_word_json(self):
        """Model literally outputs the word 'json'."""
        assert compute_score("json", SIMPLE_SCHEMA) == 0.0

    def test_response_is_schema_itself(self):
        """Model echoes back the schema instead of a valid instance."""
        assert compute_score(SIMPLE_SCHEMA, SIMPLE_SCHEMA) == 0.0
