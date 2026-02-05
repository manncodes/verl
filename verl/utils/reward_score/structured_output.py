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
Structured output reward scoring for JSON schema validation.

Implements fine-grained reward computation for structured output generation,
following the Schema RL (SRL) approach from arxiv:2502.18878.

The reward function validates LLM outputs against JSON schemas with
hierarchical scoring:
  - JSON parsability (can the output be parsed as valid JSON?)
  - Schema structural validity (does the JSON conform to the schema?)
  - Field coverage (what fraction of required fields are present and valid?)
  - Type correctness (are field types correct?)

This enables RL training (e.g., GRPO) to provide fine-grained rewards
rather than binary valid/invalid signals.
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _try_extract_json(text: str) -> Optional[str]:
    """Try to extract a JSON object or array from text.

    Handles cases where the model wraps JSON in markdown code blocks
    or includes reasoning text before/after the JSON.

    Args:
        text: The raw model output string.

    Returns:
        The extracted JSON string, or None if no JSON found.
    """
    # Try the full text first
    text = text.strip()

    # Remove markdown code blocks if present
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    if code_blocks:
        # Use the last code block (most likely to be the final answer)
        return code_blocks[-1].strip()

    # Try to find JSON object by matching outermost braces
    brace_depth = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                return text[start_idx : i + 1]

    # Try to find JSON array
    bracket_depth = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == "[":
            if bracket_depth == 0:
                start_idx = i
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start_idx is not None:
                return text[start_idx : i + 1]

    return None


def _validate_type(value: Any, type_spec) -> bool:
    """Check if a value matches a JSON Schema type specification.

    Args:
        value: The value to check.
        type_spec: A JSON Schema type string (e.g., "string") or a list of
            type strings for union types (e.g., ["string", "null"]).
    """
    # Handle union types (e.g., ["string", "null"])
    if isinstance(type_spec, list):
        return any(_validate_type(value, t) for t in type_spec)

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = type_map.get(type_spec)
    if expected is None:
        return True  # Unknown type, don't penalize
    # JSON has no distinction between int and float for "number"
    if type_spec == "integer" and isinstance(value, bool):
        return False  # bool is subclass of int in Python
    if type_spec == "number" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _validate_enum(value: Any, enum_values: list) -> bool:
    """Check if a value is one of the allowed enum values."""
    return value in enum_values


def _compute_field_scores(
    data: dict,
    schema: dict,
    prefix: str = "",
) -> dict[str, float]:
    """Compute per-field validation scores for an object against a schema.

    Returns a dict mapping field paths to scores (0.0 or 1.0).
    """
    scores = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for field_name, field_schema in properties.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name
        field_type = field_schema.get("type", None)

        if field_name not in data:
            # Missing field
            if field_name in required:
                scores[field_path] = 0.0
            # Optional fields that are missing don't count against
            continue

        value = data[field_name]

        # Type check
        if field_type and not _validate_type(value, field_type):
            scores[field_path] = 0.0
            continue

        # Enum check
        if "enum" in field_schema and not _validate_enum(value, field_schema["enum"]):
            scores[field_path] = 0.0
            continue

        # Normalize field_type for checks below (handles union types like ["object", "null"])
        _type_set = set(field_type) if isinstance(field_type, list) else {field_type} if field_type else set()

        # Nested object validation
        if "object" in _type_set and isinstance(value, dict):
            nested_scores = _compute_field_scores(value, field_schema, prefix=field_path)
            scores.update(nested_scores)
            # The parent field is valid if all nested fields are valid
            if nested_scores:
                scores[field_path] = sum(nested_scores.values()) / len(nested_scores)
            else:
                scores[field_path] = 1.0
            continue

        # Array validation
        if "array" in _type_set and isinstance(value, list):
            items_schema = field_schema.get("items", {})
            if value and items_schema:
                item_scores = []
                for idx, item in enumerate(value):
                    item_path = f"{field_path}[{idx}]"
                    if items_schema.get("type") == "object" and isinstance(item, dict):
                        nested = _compute_field_scores(item, items_schema, prefix=item_path)
                        if nested:
                            item_scores.append(sum(nested.values()) / len(nested))
                            scores.update(nested)
                        else:
                            item_scores.append(1.0)
                    elif "type" in items_schema:
                        item_valid = _validate_type(item, items_schema["type"])
                        item_scores.append(1.0 if item_valid else 0.0)
                    else:
                        item_scores.append(1.0)
                scores[field_path] = sum(item_scores) / len(item_scores) if item_scores else 1.0
            else:
                scores[field_path] = 1.0
            continue

        # Simple field that passed type and enum checks
        scores[field_path] = 1.0

    return scores


def _try_jsonschema_validate(data: Any, schema: dict) -> tuple[bool, list[str]]:
    """Validate data against a JSON schema using jsonschema library if available.

    Returns (is_valid, list_of_error_messages).
    """
    try:
        import jsonschema

        validator_cls = jsonschema.validators.validator_for(schema)
        validator = validator_cls(schema)
        errors = list(validator.iter_errors(data))
        error_msgs = [e.message for e in errors]
        return len(errors) == 0, error_msgs
    except ImportError:
        # jsonschema not installed, fall back to custom validation
        return None, []
    except Exception as e:
        logger.debug(f"jsonschema validation error: {e}")
        return None, []


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    reward_weights: Optional[dict] = None,
) -> dict:
    """Compute structured output reward score with fine-grained components.

    The reward is a weighted combination of:
    - json_parse_score: Whether the output is valid JSON (0 or 1)
    - schema_valid_score: Whether the JSON validates against the schema (0 or 1)
    - field_coverage_score: Fraction of required fields present and type-correct [0, 1]
    - content_score: Bonus for content matching ground truth (if available)

    Args:
        solution_str: The model's output string.
        ground_truth: Either a JSON schema string, a dict with {"schema": ..., "answer": ...},
                     or just a schema dict.
        extra_info: Optional dict with additional info. May contain:
            - "schema_str": JSON schema as string
            - "reward_weights": Custom weights for score components
        reward_weights: Custom weights dict. Defaults:
            {"json_parse": 0.2, "schema_valid": 0.3, "field_coverage": 0.3, "content": 0.2}

    Returns:
        dict with "score" (float) and component scores for logging.
    """
    # Default weights
    if reward_weights is None:
        reward_weights = extra_info.get("reward_weights", {}) if extra_info else {}
    weights = {
        "json_parse": reward_weights.get("json_parse", 0.2),
        "schema_valid": reward_weights.get("schema_valid", 0.3),
        "field_coverage": reward_weights.get("field_coverage", 0.3),
        "content": reward_weights.get("content", 0.2),
    }

    result = {
        "score": 0.0,
        "json_parse_score": 0.0,
        "schema_valid_score": 0.0,
        "field_coverage_score": 0.0,
        "content_score": 0.0,
    }

    # Parse the schema
    schema = None
    expected_answer = None

    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(ground_truth, dict):
        if "schema" in ground_truth and "answer" in ground_truth:
            schema = ground_truth["schema"]
            expected_answer = ground_truth["answer"]
        elif "type" in ground_truth or "properties" in ground_truth:
            # It's a schema directly
            schema = ground_truth
        elif "schema_str" in ground_truth:
            try:
                schema = json.loads(ground_truth["schema_str"])
            except (json.JSONDecodeError, TypeError):
                pass

    # Also check extra_info for schema
    if schema is None and extra_info:
        schema_str = extra_info.get("schema_str")
        if schema_str:
            try:
                schema = json.loads(schema_str)
            except (json.JSONDecodeError, TypeError):
                pass

    if schema is None:
        # Can't validate without a schema - just check JSON parsability
        json_str = _try_extract_json(solution_str)
        if json_str:
            try:
                json.loads(json_str)
                result["json_parse_score"] = 1.0
                result["score"] = weights["json_parse"]
            except json.JSONDecodeError:
                pass
        return result

    # Step 1: Extract and parse JSON
    json_str = _try_extract_json(solution_str)
    if json_str is None:
        # No JSON found at all
        return result

    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError:
        return result

    # JSON is parseable
    result["json_parse_score"] = 1.0

    # Step 2: Schema validation
    schema_valid, errors = _try_jsonschema_validate(parsed_data, schema)
    if schema_valid is None:
        # jsonschema not available, use custom validation
        if isinstance(parsed_data, dict) and schema.get("type") == "object":
            field_scores = _compute_field_scores(parsed_data, schema)
            if field_scores:
                schema_valid = all(s >= 1.0 for s in field_scores.values())
            else:
                schema_valid = True
        elif isinstance(parsed_data, list) and schema.get("type") == "array":
            schema_valid = True  # Basic type match
        else:
            expected_type = schema.get("type")
            schema_valid = _validate_type(parsed_data, expected_type) if expected_type else True

    result["schema_valid_score"] = 1.0 if schema_valid else 0.0

    # Step 3: Field coverage scoring (fine-grained)
    if isinstance(parsed_data, dict) and schema.get("type") == "object":
        field_scores = _compute_field_scores(parsed_data, schema)
        if field_scores:
            result["field_coverage_score"] = sum(field_scores.values()) / len(field_scores)
        else:
            result["field_coverage_score"] = 1.0 if schema_valid else 0.0

        # Check required field coverage specifically
        required_fields = set(schema.get("required", []))
        if required_fields:
            present_required = required_fields.intersection(set(parsed_data.keys()))
            result["required_field_ratio"] = len(present_required) / len(required_fields)
        else:
            result["required_field_ratio"] = 1.0
    else:
        result["field_coverage_score"] = 1.0 if schema_valid else 0.0

    # Step 4: Content score (if expected answer is available)
    if expected_answer is not None:
        if isinstance(expected_answer, str):
            try:
                expected_answer = json.loads(expected_answer)
            except (json.JSONDecodeError, TypeError):
                pass

        if isinstance(expected_answer, dict) and isinstance(parsed_data, dict):
            # Compare key-by-key
            all_keys = set(expected_answer.keys()) | set(parsed_data.keys())
            matching_keys = 0
            for key in all_keys:
                if key in expected_answer and key in parsed_data:
                    if str(expected_answer[key]).strip().lower() == str(parsed_data[key]).strip().lower():
                        matching_keys += 1
            result["content_score"] = matching_keys / len(all_keys) if all_keys else 1.0
        elif parsed_data == expected_answer:
            result["content_score"] = 1.0
        else:
            result["content_score"] = 0.0
    else:
        # No expected answer, content score based on whether schema is valid
        result["content_score"] = result["schema_valid_score"]

    # Compute weighted total
    result["score"] = (
        weights["json_parse"] * result["json_parse_score"]
        + weights["schema_valid"] * result["schema_valid_score"]
        + weights["field_coverage"] * result["field_coverage_score"]
        + weights["content"] * result["content_score"]
    )

    return result


def compute_score_binary(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> float:
    """Compute a binary structured output reward (0 or 1).

    Returns 1.0 only if the output is valid JSON that passes schema validation.
    Useful for strict enforcement during RL training.

    Args:
        solution_str: The model's output string.
        ground_truth: JSON schema (as string or dict).
        extra_info: Optional additional info.

    Returns:
        1.0 if valid, 0.0 otherwise.
    """
    result = compute_score(solution_str, ground_truth, extra_info)
    return 1.0 if result["schema_valid_score"] >= 1.0 and result["json_parse_score"] >= 1.0 else 0.0
