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
IFEval (Instruction Following Evaluation) reward function.

Migrated from allenai/open-instruct codebase.
This module contains all 25 constraint verification functions from the IFEval taxonomy.

Reference: Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models."
arXiv preprint arXiv:2311.07911 (2023).
"""

import json
import re
from typing import Any, List


# ============================================================================
# All 25 Constraint Verification Functions (from open-instruct)
# ============================================================================


def verify_keywords(text, keyword_list):
    """Verify if the response contains all the specified keywords."""
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text, word, N):
    """Verifies if a keyword appears exactly N times in the given text."""
    text = text.lower()
    keyword = word.lower()
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for w in words if w == keyword)
    return actual_count == N


def validate_forbidden_words(text, forbidden_words):
    """Validates that the text does not contain any of the specified forbidden words."""
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """Verifies if a given letter appears exactly the specified number of times in the text."""
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")
    actual_count = text.count(letter)
    return actual_count == N


def validate_response_language(text, language):
    """Validates that the entire response is in the specified language."""
    try:
        from langdetect import detect

        detected_language = detect(text)
        return detected_language == language
    except:
        # If langdetect fails, return True to avoid penalizing
        return True


def verify_paragraph_count(text: str, N: int) -> bool:
    """Verifies that a text contains the expected number of paragraphs."""

    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    text = clean_text(text)
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False
    return actual_count == N


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """Validates if a text meets specified word count constraints."""
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """Verifies if a text contains the expected number of sentences."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\.)\s", text)
    actual_count = len(sentences)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


def validate_paragraphs(text, N, first_word, i):
    """Validates that a text contains the expected number of paragraphs and that the i-th paragraph starts with a specific word."""
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    if paragraphs[i - 1].strip().startswith(first_word):
        return True
    return False


def verify_postscript(text, postscript_marker):
    """Verifies if a text contains a postscript starting with the given marker."""
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text: str, N: int):
    """Validates if a text contains at least the specified number of placeholders in square brackets."""
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    return len(placeholders) >= N


def verify_bullet_points(text: str, N: int):
    """Verifies if a text contains exactly N bullet points in markdown format."""
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    return len(bullet_points) == N


def validate_title(text: str) -> bool:
    """Validates if text contains a title wrapped in double angular brackets."""
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    return len(matches) > 0


def validate_choice(text: str, options: list) -> bool:
    """Validates if text matches one of the given options."""
    for option in options:
        if text in option:
            return True
    return False


def validate_highlighted_sections(text: str, N: int) -> bool:
    """Validates if text has at least N highlighted sections (markdown emphasis)."""
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)
    return len(matches) >= N


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    """Validates if text has exactly N sections."""
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


def validate_json_format(text: str) -> bool:
    """Validates if entire output is wrapped in JSON format."""
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """Validates if text starts with the original prompt."""
    return text.startswith(original_prompt)


def validate_two_responses(text: str) -> bool:
    """Validates if text contains two different responses separated by 6 asterisks."""
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        return first_response != second_response
    return False


def validate_uppercase(text: str) -> bool:
    """Validates if entire response is in uppercase."""
    return text == text.upper()


def validate_lowercase(text: str) -> bool:
    """Validates if entire response is in lowercase."""
    return text == text.lower()


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    """Validates frequency of all-capital words."""
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return len(words) == N
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


def validate_end(text: str, end_phrase: str) -> bool:
    """Validates if response ends with the exact phrase."""
    return text.endswith(end_phrase)


def validate_quotation(text: str) -> bool:
    """Validates if entire response is wrapped in double quotation marks."""
    return text.startswith('"') and text.endswith('"')


def validate_no_commas(text: str) -> bool:
    """Validates if response contains no commas."""
    return "," not in text


# Map of all IF constraint verification functions
IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}


# ============================================================================
# VERL Integration - compute_score function
# ============================================================================


def compute_score(
    solution_str: str,
    ground_truth: dict | list,
    extra_info: dict[str, Any] | None = None,
    strict: bool = True,
    use_ifeval_library: bool = True,
) -> dict[str, Any]:
    """Compute IFEval reward score for instruction following.

    Args:
        solution_str: The model's generated response to evaluate
        ground_truth: Dictionary or list containing instruction details:
            - If dict: should contain 'instruction_id_list' and 'kwargs' for each instruction
            - If list: list of instruction verification functions/rules
        extra_info: Optional dict containing additional context like 'prompt', 'instruction_id_list', etc.
        strict: If True, use strict verification. If False, apply preprocessing before verification
        use_ifeval_library: If True, attempt to use the official ifeval library if installed

    Returns:
        Dictionary containing:
            - score: Overall instruction following rate (0.0 to 1.0)
            - prompt_strict_acc: Prompt-level strict accuracy (all instructions followed)
            - inst_strict_acc: Instruction-level strict accuracy (average across instructions)
            - num_instructions: Total number of instructions
            - num_followed: Number of instructions successfully followed
    """
    # Try to use the official ifeval library if available
    if use_ifeval_library:
        try:
            from instruction_following_eval import instructions_registry

            return _compute_score_with_library(solution_str, ground_truth, extra_info, strict)
        except ImportError:
            # Fall back to custom verification functions
            pass

    # Use custom verification functions from open-instruct
    return _compute_score_with_custom_functions(solution_str, ground_truth, extra_info, strict)


def _compute_score_with_library(
    solution_str: str, ground_truth: dict | list, extra_info: dict[str, Any] | None, strict: bool
) -> dict[str, Any]:
    """Compute score using the official ifeval library."""
    from instruction_following_eval import instructions_registry

    # Preprocess response for loose accuracy
    response = solution_str
    if not strict:
        response = _preprocess_response_loose(response)

    # Extract instruction information
    instruction_list, kwargs_list = _extract_instruction_data(ground_truth, extra_info)

    if not instruction_list:
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0}

    # Verify each instruction
    num_instructions = len(instruction_list)
    num_followed = 0

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
            if instruction_cls is None:
                continue

            instruction = instruction_cls(instruction_id)
            is_followed = instruction.check_following(response, **kwargs)
            if is_followed:
                num_followed += 1
        except Exception:
            continue

    # Calculate metrics
    inst_level_acc = num_followed / num_instructions if num_instructions > 0 else 0.0
    prompt_level_acc = 1.0 if num_followed == num_instructions else 0.0

    return {
        "score": inst_level_acc,
        "prompt_strict_acc": prompt_level_acc,
        "inst_strict_acc": inst_level_acc,
        "num_instructions": num_instructions,
        "num_followed": num_followed,
    }


def _compute_score_with_custom_functions(
    solution_str: str, ground_truth: dict | list, extra_info: dict[str, Any] | None, strict: bool
) -> dict[str, Any]:
    """Compute score using custom verification functions from open-instruct."""
    # Preprocess response for loose accuracy
    response = solution_str
    if not strict:
        response = _preprocess_response_loose(response)

    # Extract instruction information
    instruction_list, kwargs_list = _extract_instruction_data(ground_truth, extra_info)

    if not instruction_list:
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0}

    # Verify each instruction using custom functions
    num_instructions = len(instruction_list)
    num_followed = 0

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            # Map instruction_id to function name
            func_name = _map_instruction_to_function(instruction_id, kwargs)
            if func_name and func_name in IF_FUNCTIONS_MAP:
                func = IF_FUNCTIONS_MAP[func_name]
                # Remove None values and func_name from kwargs
                clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and k != "func_name"}
                # Call the verification function
                if clean_kwargs:
                    is_followed = func(response, **clean_kwargs)
                else:
                    is_followed = func(response)

                if is_followed:
                    num_followed += 1
        except Exception:
            # If verification fails, count as not followed
            continue

    # Calculate metrics
    inst_level_acc = num_followed / num_instructions if num_instructions > 0 else 0.0
    prompt_level_acc = 1.0 if num_followed == num_instructions else 0.0

    return {
        "score": inst_level_acc,
        "prompt_strict_acc": prompt_level_acc,
        "inst_strict_acc": inst_level_acc,
        "num_instructions": num_instructions,
        "num_followed": num_followed,
    }


def _extract_instruction_data(ground_truth: dict | list, extra_info: dict[str, Any] | None):
    """Extract instruction_id_list and kwargs from ground_truth or extra_info."""
    instruction_list = []
    kwargs_list = []

    # Try ground_truth first
    if isinstance(ground_truth, dict):
        instruction_list = ground_truth.get("instruction_id_list", [])
        kwargs_list = ground_truth.get("kwargs", [])

    # Fall back to extra_info if needed
    if not instruction_list and extra_info is not None:
        instruction_list = extra_info.get("instruction_id_list", [])
        kwargs_list = extra_info.get("kwargs", [])

    # Ensure kwargs_list matches instruction_list length
    if len(kwargs_list) < len(instruction_list):
        kwargs_list.extend([{}] * (len(instruction_list) - len(kwargs_list)))

    return instruction_list, kwargs_list


def _map_instruction_to_function(instruction_id: str, kwargs: dict) -> str | None:
    """Map an instruction_id to a verification function name.

    This mapping is based on the open-instruct IFEvalVerifierOld implementation,
    which uses explicit function names in the constraint dict.
    """
    # If kwargs has func_name, use it directly (old format)
    if "func_name" in kwargs:
        return kwargs["func_name"]

    # Otherwise, infer from instruction_id (new format)
    # Map common instruction patterns to function names
    if "keywords:existence" in instruction_id:
        return "verify_keywords"
    elif "keywords:frequency" in instruction_id:
        return "verify_keyword_frequency"
    elif "keywords:forbidden_words" in instruction_id:
        return "validate_forbidden_words"
    elif "keywords:letter_frequency" in instruction_id:
        return "verify_letter_frequency"
    elif "language:response_language" in instruction_id:
        return "validate_response_language"
    elif "length_constraints:number_paragraphs" in instruction_id:
        return "verify_paragraph_count"
    elif "length_constraints:number_words" in instruction_id:
        return "validate_word_constraint"
    elif "length_constraints:number_sentences" in instruction_id:
        return "verify_sentence_constraint"
    elif "length_constraints:nth_paragraph_first_word" in instruction_id:
        return "validate_paragraphs"
    elif "detectable_content:postscript" in instruction_id:
        return "verify_postscript"
    elif "detectable_content:number_placeholders" in instruction_id:
        return "validate_placeholders"
    elif "detectable_format:number_bullet_lists" in instruction_id:
        return "verify_bullet_points"
    elif "detectable_format:title" in instruction_id:
        return "validate_title"
    elif "detectable_format:constrained_response" in instruction_id:
        return "validate_choice"
    elif "detectable_format:number_highlighted_sections" in instruction_id:
        return "validate_highlighted_sections"
    elif "detectable_format:multiple_sections" in instruction_id:
        return "validate_sections"
    elif "detectable_format:json_format" in instruction_id:
        return "validate_json_format"
    elif "combination:repeat_prompt" in instruction_id:
        return "validate_repeat_prompt"
    elif "combination:two_responses" in instruction_id:
        return "validate_two_responses"
    elif "change_case:english_capital" in instruction_id:
        return "validate_uppercase"
    elif "change_case:english_lowercase" in instruction_id:
        return "validate_lowercase"
    elif "detectable_format:number_highlighted_sections" in instruction_id:
        return "validate_highlighted_sections"
    elif "detectable_content:number_placeholders" in instruction_id:
        return "validate_placeholders"
    elif "startswith:quotation" in instruction_id:
        return "validate_quotation"
    elif "punctuation:no_comma" in instruction_id:
        return "validate_no_commas"
    else:
        # Return None if no mapping found
        return None


def _preprocess_response_loose(response: str) -> str:
    """Preprocess response for loose accuracy evaluation."""
    lines = response.split("\n")

    # Remove first and last lines if response has more than 2 lines
    if len(lines) > 2:
        lines = lines[1:-1]

    response = "\n".join(lines)

    # Remove common markdown formatting
    response = re.sub(r"\*\*(.+?)\*\*", r"\1", response)  # Bold
    response = re.sub(r"\*(.+?)\*", r"\1", response)  # Italic
    response = re.sub(r"__(.+?)__", r"\1", response)  # Bold
    response = re.sub(r"_(.+?)_", r"\1", response)  # Italic

    return response
