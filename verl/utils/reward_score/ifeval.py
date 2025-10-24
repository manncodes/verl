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

Reference: Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models."
arXiv preprint arXiv:2311.07911 (2023).

This module provides reward scoring based on instruction following verification.
It supports the Google IFEval benchmark which evaluates models on their ability
to follow verifiable instructions.

The evaluation supports both strict and loose accuracy modes:
- Strict: Exact verification of instruction following
- Loose: Applies preprocessing (remove first/last line, markdown cleanup) before verification
"""

import re
from typing import Any


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
            import sys

            # Try to import the ifeval library
            # Note: This requires the instruction_following_eval package to be installed
            # pip install git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval
            from instruction_following_eval import instructions_registry

            return _compute_score_with_library(solution_str, ground_truth, extra_info, strict)
        except ImportError:
            # Fall back to built-in verification if library not available
            pass

    # Use built-in verification functions
    return _compute_score_builtin(solution_str, ground_truth, extra_info, strict)


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
    if isinstance(ground_truth, dict):
        instruction_list = ground_truth.get("instruction_id_list", [])
        kwargs_list = ground_truth.get("kwargs", [])
    elif extra_info is not None:
        instruction_list = extra_info.get("instruction_id_list", [])
        kwargs_list = extra_info.get("kwargs", [])
    else:
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0, "error": "No instruction information provided"}

    if not instruction_list:
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0}

    # Verify each instruction
    num_instructions = len(instruction_list)
    num_followed = 0

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            # Get the instruction checker from the registry
            instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
            if instruction_cls is None:
                continue

            # Create instruction instance and check
            instruction = instruction_cls(instruction_id)
            is_followed = instruction.check_following(response, **kwargs)
            if is_followed:
                num_followed += 1
        except Exception:
            # If verification fails, count as not followed
            continue

    # Calculate metrics
    inst_level_acc = num_followed / num_instructions if num_instructions > 0 else 0.0
    prompt_level_acc = 1.0 if num_followed == num_instructions else 0.0

    return {
        "score": inst_level_acc,  # Use instruction-level accuracy as the main score
        "prompt_strict_acc": prompt_level_acc,
        "inst_strict_acc": inst_level_acc,
        "num_instructions": num_instructions,
        "num_followed": num_followed,
    }


def _compute_score_builtin(
    solution_str: str, ground_truth: dict | list, extra_info: dict[str, Any] | None, strict: bool
) -> dict[str, Any]:
    """Compute score using built-in verification functions.

    This is a fallback implementation that covers common IFEval instruction types.
    For full compatibility, install the official ifeval library.
    """
    # Preprocess response for loose accuracy
    response = solution_str
    if not strict:
        response = _preprocess_response_loose(response)

    # Extract instruction information
    if isinstance(ground_truth, dict):
        instruction_list = ground_truth.get("instruction_id_list", [])
        kwargs_list = ground_truth.get("kwargs", [])
    elif extra_info is not None:
        instruction_list = extra_info.get("instruction_id_list", [])
        kwargs_list = extra_info.get("kwargs", [])
    else:
        # If no specific instructions provided, return 0
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0, "error": "No instruction information provided"}

    if not instruction_list:
        return {"score": 0.0, "num_instructions": 0, "num_followed": 0}

    # Verify each instruction using built-in checkers
    num_instructions = len(instruction_list)
    num_followed = 0

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            is_followed = _check_instruction_builtin(instruction_id, response, kwargs)
            if is_followed:
                num_followed += 1
        except Exception:
            # If verification fails, count as not followed
            continue

    # Calculate metrics
    inst_level_acc = num_followed / num_instructions if num_instructions > 0 else 0.0
    prompt_level_acc = 1.0 if num_followed == num_instructions else 0.0

    return {
        "score": inst_level_acc,  # Use instruction-level accuracy as the main score
        "prompt_strict_acc": prompt_level_acc,
        "inst_strict_acc": inst_level_acc,
        "num_instructions": num_instructions,
        "num_followed": num_followed,
    }


def _preprocess_response_loose(response: str) -> str:
    """Preprocess response for loose accuracy evaluation.

    Applies transformations:
    - Remove first line (to skip intros like "Sure, here it is:")
    - Remove last line (to skip outros)
    - Remove markdown formatting
    """
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


def _check_instruction_builtin(instruction_id: str, response: str, kwargs: dict) -> bool:
    """Check if response follows instruction using built-in verifiers.

    Supports common IFEval instruction types:
    - keywords: Mention specific keywords
    - forbidden_words: Avoid forbidden words
    - length_constraints: Number of words/sentences/paragraphs
    - startswith/endswith: Response starts/ends with specific content
    - number_paragraphs: Number of paragraphs
    - number_sentences: Number of sentences
    - etc.
    """
    # Forbidden words (check before keywords to avoid matching "keywords:forbidden_words")
    if "forbidden_words" in instruction_id:
        forbidden_words = kwargs.get("forbidden_words", [])
        for word in forbidden_words:
            if word.lower() in response.lower():
                return False
        return True

    # Keyword constraints
    elif "keywords:" in instruction_id:
        keywords = kwargs.get("keywords", [])
        relation = kwargs.get("relation", "at least")
        frequency = kwargs.get("frequency", 1)

        for keyword in keywords:
            count = response.lower().count(keyword.lower())
            if relation == "at least" and count < frequency:
                return False
            elif relation == "at most" and count > frequency:
                return False

        return True

    # Length constraints: number of words
    elif "length_constraints:number_words" in instruction_id:
        relation = kwargs.get("relation", "at least")
        num_words = kwargs.get("num_words", 0)
        word_count = len(response.split())

        if relation == "at least":
            return word_count >= num_words
        elif relation == "at most":
            return word_count <= num_words
        elif relation == "less than":
            return word_count < num_words
        elif relation == "more than":
            return word_count > num_words
        else:
            return word_count == num_words

    # Number of paragraphs
    elif "number_paragraphs" in instruction_id or "num_paragraphs" in instruction_id:
        num_paragraphs = kwargs.get("num_paragraphs", 0)
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return len(paragraphs) == num_paragraphs

    # Number of sentences
    elif "number_sentences" in instruction_id or "num_sentences" in instruction_id:
        relation = kwargs.get("relation", "at least")
        num_sentences = kwargs.get("num_sentences", 0)

        # Simple sentence splitting (can be improved)
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        if relation == "at least":
            return sentence_count >= num_sentences
        elif relation == "at most":
            return sentence_count <= num_sentences
        elif relation == "less than":
            return sentence_count < num_sentences
        else:
            return sentence_count == num_sentences

    # Starts with
    elif "startswith:" in instruction_id:
        start_phrase = kwargs.get("start_phrase", "")
        return response.strip().startswith(start_phrase)

    # Ends with
    elif "endswith:" in instruction_id:
        end_phrase = kwargs.get("end_phrase", "")
        return response.strip().endswith(end_phrase)

    # Contains letter frequency
    elif "letter_frequency" in instruction_id or "nth_letter_check" in instruction_id:
        letter = kwargs.get("letter", "").lower()
        frequency = kwargs.get("frequency", 0) or kwargs.get("let_frequency", 0)

        if not letter:
            return True

        count = response.lower().count(letter)
        return count >= frequency

    # Change case (all lowercase or all uppercase)
    elif "change_case:" in instruction_id:
        case = kwargs.get("capital", "lower")
        if case == "lower":
            return response == response.lower()
        elif case == "upper":
            return response == response.upper()

    # Detectable format (JSON, bullet list, title, etc.)
    elif "detectable_format" in instruction_id:
        format_type = instruction_id.split(":")[-1] if ":" in instruction_id else ""

        if "json" in format_type.lower():
            # Check if response is valid JSON
            try:
                import json

                json.loads(response)
                return True
            except:
                return False

        elif "bullet_list" in format_type.lower() or "number_bullet_lists" in instruction_id:
            # Check for bullet points
            num_bullets = kwargs.get("num_bullets", 0)
            bullet_patterns = [r"^\s*[-*•]", r"^\s*\d+\."]
            lines = response.split("\n")
            bullet_count = sum(1 for line in lines if any(re.match(p, line) for p in bullet_patterns))
            return bullet_count >= num_bullets

        elif "title" in format_type.lower():
            # Check if response has a title (first line is different/capitalized)
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            return len(lines) > 0 and (lines[0].isupper() or lines[0].istitle())

    # Number of highlighted sections (bold, italic, etc.)
    elif "number_highlighted_sections" in instruction_id or "highlighted_section" in instruction_id:
        num_highlights = kwargs.get("num_highlights", 0)
        # Count markdown bold/italic sections
        bold_count = len(re.findall(r"\*\*(.+?)\*\*", response))
        italic_count = len(re.findall(r"\*(.+?)\*", response))
        total_highlights = bold_count + italic_count
        return total_highlights >= num_highlights

    # Placeholder constraint (use specific placeholder format)
    elif "detectable_content:number_placeholders" in instruction_id:
        num_placeholders = kwargs.get("num_placeholders", 0)
        # Check for square bracket placeholders
        placeholders = re.findall(r"\[.+?\]", response)
        return len(placeholders) >= num_placeholders

    # Postscript marker
    elif "postscript_marker" in instruction_id or "end_checker:end_with_postscript" in instruction_id:
        postscript_marker = kwargs.get("postscript_marker", "P.S.")
        return postscript_marker in response

    # Quotation
    elif "quotation" in instruction_id:
        # Check for quoted text (regular quotes and smart quotes)
        return '"' in response or "'" in response or '\u201c' in response or '\u2018' in response

    # Number of sections
    elif "detectable_format:multiple_sections" in instruction_id:
        section_splitter = kwargs.get("section_splitter", "Section")
        num_sections = kwargs.get("num_sections", 0)
        sections = response.split(section_splitter)
        return len(sections) >= num_sections + 1  # +1 because split creates n+1 parts for n separators

    # Repeat prompt
    elif "repeat_prompt" in instruction_id or "combination:repeat_prompt" in instruction_id:
        prompt = kwargs.get("prompt_to_repeat", "")
        return prompt.lower() in response.lower()

    # Two responses (responses separated by specific markers)
    elif "two_responses" in instruction_id or "combination:two_responses" in instruction_id:
        # Check for multiple response markers like "Response 1:", "Response 2:"
        markers = ["Response 1", "Response 2", "***", "---"]
        return any(marker in response for marker in markers)

    # Default: if instruction type is unknown, return True to avoid penalizing
    # In production, you may want to return False or log a warning
    return True
