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
Reward function for structured output (JSON schema) validation.

This module provides a binary reward signal for RLVR training on structured
output tasks. The reward is 1.0 if the model's response is valid JSON that
conforms to the given JSON schema, and 0.0 otherwise.

Compatible with the nvidia/Nemotron-RL-instruction_following-structured_outputs
dataset format.

Requires: pip install openapi-schema-validator
"""

import json
import random
from typing import Any, Dict


def strictify_schema(schema: Any) -> None:
    """Make a JSON schema strict by requiring all properties and disallowing extras.

    Recursively traverses the schema and for every object that has "properties",
    sets "required" to include all property names and "additionalProperties" to False.
    This mirrors the OpenAI strict mode for structured outputs.

    Args:
        schema: A parsed JSON schema (dict or nested structure).
    """
    if isinstance(schema, dict):
        if "properties" in schema:
            schema["required"] = list(schema["properties"])
            schema["additionalProperties"] = False
        for v in schema.values():
            strictify_schema(v)
    elif isinstance(schema, list):
        for item in schema:
            strictify_schema(item)


def validate_json_schema(schema_str: str, response_text: str) -> bool:
    """Validate a response string against a JSON schema.

    Args:
        schema_str: JSON string representation of the schema.
        response_text: The model's response text to validate.

    Returns:
        True if the response is valid JSON conforming to the schema, False otherwise.
    """
    from openapi_schema_validator import validate as validate_against_schema_openapi

    schema = json.loads(schema_str)
    strictify_schema(schema)
    response_obj = json.loads(response_text)
    validate_against_schema_openapi(response_obj, schema)
    return True


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON content from a model response.

    Handles cases where the model wraps JSON in markdown code blocks
    or includes extra text before/after the JSON.

    Args:
        response_text: The raw model response text.

    Returns:
        The extracted JSON string.
    """
    text = response_text.strip()

    # Handle markdown code blocks: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    # Try to find JSON object or array boundaries
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return text


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] = None,
    **kwargs,
) -> float:
    """Compute a binary reward for structured output (JSON schema) compliance.

    The reward is 1.0 if the model's response is valid JSON that conforms to the
    provided JSON schema, and 0.0 otherwise. No semantic content evaluation is
    performed -- only schema adherence is checked.

    Args:
        solution_str: The model's generated response text.
        ground_truth: The JSON schema string to validate against.
        extra_info: Optional dict with additional info. May contain:
            - "schema_type": The type of schema (currently only "json" supported).
            - "schema_fields_count": Number of fields in the schema.

    Returns:
        float: 1.0 if the response conforms to the schema, 0.0 otherwise.
    """
    do_print = random.randint(1, 64) == 1

    if extra_info is None:
        extra_info = {}

    schema_type = extra_info.get("schema_type", "json")

    if schema_type != "json":
        if do_print:
            print(f"[structured_output] Unsupported schema_type: {schema_type}")
        return 0.0

    try:
        # Try to extract JSON from the response (handles markdown blocks, etc.)
        response_text = extract_json_from_response(solution_str)
        is_valid = validate_json_schema(ground_truth, response_text)
        reward = 1.0 if is_valid else 0.0
    except Exception as e:
        reward = 0.0
        if do_print:
            print("--------------------------------")
            print(f"[structured_output] Validation failed: {type(e).__name__}: {e}")
            print(f"Schema (first 200 chars): {ground_truth[:200]}...")
            print(f"Response (first 200 chars): {solution_str[:200]}...")

    if do_print:
        print("--------------------------------")
        print(f"[structured_output] Schema type: {schema_type}")
        print(f"[structured_output] Reward: {reward}")
        print(f"[structured_output] Response (first 200 chars): {solution_str[:200]}...")

    return reward
