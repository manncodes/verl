# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""
LaTeX and mathematical expression normalization.

This module provides configurable normalization for mathematical
expressions to enable fair string comparison.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for answer normalization."""

    # Whitespace handling
    strip_whitespace: bool = True
    collapse_whitespace: bool = True

    # LaTeX normalization
    normalize_fractions: bool = True  # \dfrac -> \frac, \tfrac -> \frac
    normalize_sqrt: bool = True  # Expand shorthand \sqrta -> \sqrt{a}
    expand_frac_shorthand: bool = True  # \frac12 -> \frac{1}{2}
    remove_display_style: bool = True  # Remove \displaystyle, \textstyle
    remove_sizing: bool = True  # Remove \left, \right, \big, etc.

    # Text normalization
    remove_text_wrappers: bool = True  # \text{foo} -> foo
    remove_mathrm: bool = True  # \mathrm{foo} -> foo

    # Unit handling (be careful with this)
    remove_units: bool = False
    unit_patterns: list[str] = field(default_factory=lambda: [
        r"\\text\{[a-zA-Z]+\}$",  # Trailing \text{units}
        r"\s*(cm|mm|m|km|ft|in|mi|mph|kg|g|lb|oz|s|min|hr|degrees?)$",
    ])

    # Number normalization
    remove_thousands_separator: bool = True  # 1,000 -> 1000
    normalize_decimals: bool = False  # Don't change 0.5 to .5 by default


def normalize_latex(
    expr: str,
    config: Optional[NormalizationConfig] = None,
) -> str:
    """
    Normalize a LaTeX mathematical expression for comparison.

    Applies a series of transformations to put expressions in a
    canonical form. Only applies transformations that preserve
    mathematical meaning.

    Args:
        expr: The LaTeX expression to normalize
        config: Normalization configuration

    Returns:
        Normalized expression string

    Example:
        >>> normalize_latex(r"\\dfrac{1}{2}")
        '\\frac{1}{2}'
        >>> normalize_latex(r"\\frac12")
        '\\frac{1}{2}'
    """
    if config is None:
        config = NormalizationConfig()

    if not expr:
        return ""

    result = expr

    # Strip outer whitespace first
    if config.strip_whitespace:
        result = result.strip()

    # Remove display/text style commands
    if config.remove_display_style:
        result = re.sub(r"\\displaystyle\s*", "", result)
        result = re.sub(r"\\textstyle\s*", "", result)

    # Normalize fraction commands
    if config.normalize_fractions:
        result = result.replace("\\dfrac", "\\frac")
        result = result.replace("\\tfrac", "\\frac")
        result = result.replace("\\cfrac", "\\frac")

    # Expand fraction shorthand: \frac12 -> \frac{1}{2}
    # This is tricky - we need to handle:
    #   \frac12 -> \frac{1}{2}
    #   \frac1{23} -> \frac{1}{23}
    #   \frac{12}3 -> \frac{12}{3}
    #   \frac{12}{34} -> unchanged
    if config.expand_frac_shorthand:
        result = _expand_frac_shorthand(result)

    # Expand sqrt shorthand: \sqrta -> \sqrt{a}
    if config.normalize_sqrt:
        result = _expand_sqrt_shorthand(result)

    # Remove sizing commands
    if config.remove_sizing:
        for cmd in ["\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg"]:
            result = result.replace(cmd, "")

    # Remove text wrappers
    if config.remove_text_wrappers:
        result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\textbf\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\textit\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mbox\{([^}]*)\}", r"\1", result)

    if config.remove_mathrm:
        result = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathit\{([^}]*)\}", r"\1", result)

    # Remove units if configured
    if config.remove_units:
        for pattern in config.unit_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    # Normalize thousands separators
    if config.remove_thousands_separator:
        # Match numbers with commas as thousand separators
        result = re.sub(r"(\d),(\d{3})", r"\1\2", result)
        # Repeat for numbers with multiple separators
        while re.search(r"(\d),(\d{3})", result):
            result = re.sub(r"(\d),(\d{3})", r"\1\2", result)

    # Collapse whitespace
    if config.collapse_whitespace:
        result = re.sub(r"\s+", " ", result)
        # Remove spaces around operators and braces for consistency
        result = re.sub(r"\s*([{}()\[\]+\-*/^_=])\s*", r"\1", result)

    # Final strip
    if config.strip_whitespace:
        result = result.strip()

    return result


def _expand_frac_shorthand(expr: str) -> str:
    """
    Expand LaTeX fraction shorthand notations.

    Handles the various ways fractions can be written:
    - \\frac12 -> \\frac{1}{2}
    - \\frac1{23} -> \\frac{1}{23}
    - \\frac{12}3 -> \\frac{12}{3}
    """
    result = expr

    # Pattern 1: \frac followed by single char then single char (no braces)
    # \frac12 -> \frac{1}{2}
    # But NOT \fracab where a,b are letters (ambiguous)
    result = re.sub(
        r"\\frac([0-9])([0-9])",
        r"\\frac{\1}{\2}",
        result
    )

    # Pattern 2: \frac followed by single char then {stuff}
    # \frac1{23} -> \frac{1}{23}
    result = re.sub(
        r"\\frac([0-9])\{",
        r"\\frac{\1}{",
        result
    )

    # Pattern 3: \frac{stuff} followed by single char
    # \frac{12}3 -> \frac{12}{3}
    # This is trickier - need to find closing brace first

    # Use a loop to handle this properly
    i = 0
    while i < len(result):
        if result[i:i+6] == "\\frac{":
            # Find matching close brace
            depth = 0
            j = i + 5  # Start at the {
            while j < len(result):
                if result[j] == "{":
                    depth += 1
                elif result[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1

            # j now points to the closing brace of numerator
            if j + 1 < len(result):
                next_char = result[j + 1]
                # If next char is a digit (not a brace), wrap it
                if next_char.isdigit():
                    result = result[:j+1] + "{" + next_char + "}" + result[j+2:]
            i = j + 1
        else:
            i += 1

    return result


def _expand_sqrt_shorthand(expr: str) -> str:
    """
    Expand LaTeX sqrt shorthand: \\sqrta -> \\sqrt{a}

    Only expands when followed by a single non-brace character.
    """
    result = expr

    # \sqrt followed by single digit -> \sqrt{digit}
    result = re.sub(
        r"\\sqrt([0-9])",
        r"\\sqrt{\1}",
        result
    )

    # \sqrt followed by single letter (but not followed by more letters)
    # \sqrtx -> \sqrt{x}, but not \sqrtxy
    result = re.sub(
        r"\\sqrt([a-zA-Z])(?![a-zA-Z{])",
        r"\\sqrt{\1}",
        result
    )

    return result


def normalize_for_numeric_comparison(expr: str) -> Optional[float]:
    """
    Attempt to convert an expression to a float for numeric comparison.

    Handles:
    - Plain numbers: 42, 3.14, -17
    - Fractions: 1/2, \\frac{1}{2}
    - Scientific notation: 1.5e-3
    - Percentages: 50% -> 0.5

    Args:
        expr: The expression to convert

    Returns:
        Float value or None if conversion fails
    """
    if not expr:
        return None

    expr = expr.strip()

    # Remove LaTeX math delimiters
    expr = re.sub(r"^\$+|\$+$", "", expr)
    expr = re.sub(r"^\\+\(|\\+\)$", "", expr)

    # Handle percentage
    if expr.endswith("%"):
        try:
            return float(expr[:-1]) / 100
        except ValueError:
            pass

    # Try direct float conversion
    try:
        return float(expr)
    except ValueError:
        pass

    # Handle simple fractions: a/b
    frac_match = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", expr)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            denom = float(frac_match.group(2))
            if denom != 0:
                return num / denom
        except ValueError:
            pass

    # Handle LaTeX fractions: \frac{a}{b}
    latex_frac = re.match(r"^\\frac\{([^}]+)\}\{([^}]+)\}$", expr)
    if latex_frac:
        try:
            num = float(latex_frac.group(1))
            denom = float(latex_frac.group(2))
            if denom != 0:
                return num / denom
        except ValueError:
            pass

    # Handle negative with parentheses: -(3)
    neg_match = re.match(r"^-\((\d+(?:\.\d+)?)\)$", expr)
    if neg_match:
        try:
            return -float(neg_match.group(1))
        except ValueError:
            pass

    return None
