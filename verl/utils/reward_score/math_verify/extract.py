# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""
Answer extraction strategies for mathematical solutions.

This module provides multiple strategies for extracting answers from
model outputs, with proper handling of LaTeX syntax.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from verl.utils.reward_score.math_verify.core import ExtractionResult, ExtractionMethod

logger = logging.getLogger(__name__)


def extract_boxed(text: str, find_last: bool = True) -> ExtractionResult:
    """
    Extract content from \\boxed{...} with proper brace matching.

    Handles nested braces correctly, unlike naive regex approaches.

    Args:
        text: The text to search
        find_last: If True, find the last \\boxed occurrence (typical for CoT)

    Returns:
        ExtractionResult with the boxed content or None

    Example:
        >>> extract_boxed(r"So \\boxed{\\frac{1}{2}} is the answer")
        ExtractionResult(answer='\\frac{1}{2}', method=ExtractionMethod.BOXED, ...)
    """
    # Find the starting position
    if find_last:
        idx = text.rfind("\\boxed{")
    else:
        idx = text.find("\\boxed{")

    if idx < 0:
        # Try alternative boxed formats
        for alt in ["\\boxed {", "\\fbox{", "\\fbox {"]:
            if find_last:
                idx = text.rfind(alt)
            else:
                idx = text.find(alt)
            if idx >= 0:
                break

    if idx < 0:
        return ExtractionResult(
            answer=None,
            method=ExtractionMethod.FAILED,
        )

    # Find the opening brace
    brace_start = text.find("{", idx)
    if brace_start < 0:
        return ExtractionResult(
            answer=None,
            method=ExtractionMethod.FAILED,
        )

    # Match braces properly
    content_start = brace_start + 1
    depth = 1
    i = content_start

    while i < len(text) and depth > 0:
        char = text[i]

        # Handle escaped braces
        if i > 0 and text[i - 1] == "\\":
            i += 1
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        i += 1

    if depth != 0:
        # Unbalanced braces - try to recover by taking content up to last }
        logger.warning(f"Unbalanced braces in boxed expression at position {idx}")
        last_brace = text.rfind("}", content_start)
        if last_brace > content_start:
            content = text[content_start:last_brace]
            return ExtractionResult(
                answer=content.strip(),
                method=ExtractionMethod.BOXED,
                raw_match=text[idx:last_brace + 1],
                confidence=0.7,  # Lower confidence due to recovery
            )
        return ExtractionResult(
            answer=None,
            method=ExtractionMethod.FAILED,
        )

    content = text[content_start:i - 1]
    return ExtractionResult(
        answer=content.strip(),
        method=ExtractionMethod.BOXED,
        raw_match=text[idx:i],
        confidence=1.0,
    )


def extract_by_pattern(
    text: str,
    patterns: list[str],
    find_last: bool = True,
) -> ExtractionResult:
    """
    Extract answer using regex patterns.

    Tries patterns in order, returns first match.

    Args:
        text: The text to search
        patterns: List of regex patterns with a capture group for the answer
        find_last: If True, find the last match for each pattern

    Returns:
        ExtractionResult with the matched answer or None
    """
    for pattern in patterns:
        try:
            if find_last:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
                match = matches[-1] if matches else None
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                # Get the captured group if present, otherwise full match
                answer = match.group(1) if match.groups() else match.group(0)
                answer = answer.strip()

                # Clean up common artifacts
                answer = re.sub(r"[\.\,\s]+$", "", answer)  # Trailing punctuation

                if answer:  # Don't return empty strings
                    return ExtractionResult(
                        answer=answer,
                        method=ExtractionMethod.PATTERN,
                        raw_match=match.group(0),
                        confidence=0.8,
                    )
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            continue

    return ExtractionResult(
        answer=None,
        method=ExtractionMethod.FAILED,
    )


def extract_last_number(text: str) -> ExtractionResult:
    """
    Extract the last number from text as a fallback strategy.

    Handles integers, decimals, fractions, and scientific notation.

    Args:
        text: The text to search

    Returns:
        ExtractionResult with the last number found or None
    """
    # Pattern for various number formats
    number_patterns = [
        # LaTeX fractions: \frac{a}{b}
        r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}",
        # Decimal/integer with optional negative: -123.456
        r"-?\d+\.?\d*",
        # Scientific notation: 1.23e-4
        r"-?\d+\.?\d*[eE][+-]?\d+",
        # Simple fractions: 1/2
        r"-?\d+/\d+",
    ]

    last_match = None
    last_pos = -1

    for pattern in number_patterns:
        for match in re.finditer(pattern, text):
            if match.end() > last_pos:
                last_pos = match.end()
                last_match = match.group(0)

    if last_match:
        return ExtractionResult(
            answer=last_match,
            method=ExtractionMethod.LAST_NUMBER,
            raw_match=last_match,
            confidence=0.5,  # Lower confidence for this fallback
        )

    return ExtractionResult(
        answer=None,
        method=ExtractionMethod.FAILED,
    )


def extract_answer(
    text: str,
    patterns: Optional[list[str]] = None,
    hint: Optional[ExtractionMethod] = None,
    try_all: bool = True,
) -> ExtractionResult:
    """
    Extract answer using multiple strategies with fallback.

    Default order:
    1. \\boxed{} extraction (most reliable for math)
    2. Pattern matching (for "Answer: X" formats)
    3. Last number extraction (fallback)

    Args:
        text: The model output text
        patterns: Custom regex patterns for pattern extraction
        hint: Preferred extraction method to try first
        try_all: If True, try all methods until one succeeds

    Returns:
        ExtractionResult from the first successful method
    """
    if patterns is None:
        patterns = [
            r"(?i)\*{0,2}(?:final\s+)?answer\*{0,2}\s*[:=]\s*(.+?)(?:\n|$)",
            r"(?i)(?:the\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|,|\n|$)",
            r"(?i)therefore[,\s]+(.+?)(?:\.|$)",
            r"(?i)thus[,\s]+(.+?)(?:\.|$)",
        ]

    # Define extraction methods in priority order
    extractors = [
        (ExtractionMethod.BOXED, lambda: extract_boxed(text)),
        (ExtractionMethod.PATTERN, lambda: extract_by_pattern(text, patterns)),
        (ExtractionMethod.LAST_NUMBER, lambda: extract_last_number(text)),
    ]

    # Reorder if hint provided
    if hint is not None:
        extractors = sorted(
            extractors,
            key=lambda x: 0 if x[0] == hint else 1
        )

    # Try each extractor
    for method, extractor in extractors:
        result = extractor()
        if result.success:
            logger.debug(f"Extraction succeeded with method {method.name}: {result.answer}")
            return result
        if not try_all:
            break

    logger.debug("All extraction methods failed")
    return ExtractionResult(
        answer=None,
        method=ExtractionMethod.FAILED,
    )


def extract_all_boxed(text: str) -> list[str]:
    """
    Extract all boxed expressions from text.

    Useful for multi-part answers or when intermediate boxed
    expressions need to be considered.

    Args:
        text: The text to search

    Returns:
        List of all boxed contents in order of appearance
    """
    results = []
    remaining = text

    while "\\boxed{" in remaining:
        result = extract_boxed(remaining, find_last=False)
        if not result.success:
            break
        results.append(result.answer)
        # Move past this match
        idx = remaining.find("\\boxed{")
        remaining = remaining[idx + len("\\boxed{") + len(result.answer) + 1:]

    return results
