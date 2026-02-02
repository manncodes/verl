# Copyright 2024
# Licensed under the Apache License, Version 2.0
# Incorporates normalization logic from prime_math (PRIME team, OpenAI prm800k, Hendrycks MATH)

"""
LaTeX and mathematical expression normalization.

This module provides configurable normalization for mathematical
expressions to enable fair string comparison.

Features incorporated from prime_math:
- Unicode symbol conversion (√, π, ∞, etc.)
- Unit stripping (degree, cm, meter, etc.)
- Large number words (million, billion, trillion)
- Mixed number handling (7 3/4 -> 7+3/4)
- Pi normalization
"""

from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Unicode to ASCII/LaTeX mappings (from prime_math)
UNICODE_MAPPINGS = {
    "√": "\\sqrt",  # Map to LaTeX command
    "π": "\\pi",    # Map to LaTeX command
    "∞": "\\infty",
    "∪": "\\cup",
    "·": "\\cdot",
    "×": "\\times",
    "÷": "\\div",
    "−": "-",  # Unicode minus
    "–": "-",  # En dash
    "—": "-",  # Em dash
    "'": "'",
    "'": "'",
    """: '"',
    """: '"',
    "≠": "\\neq",
    "≤": "\\leq",
    "≥": "\\geq",
    "±": "\\pm",
    "∓": "\\mp",
    "°": "^\\circ",
}

# Units that can be stripped (from prime_math)
UNITS_TO_STRIP = [
    "degree", "degrees",
    "cm", "centimeter", "centimeters",
    "mm", "millimeter", "millimeters",
    "m", "meter", "meters",
    "km", "kilometer", "kilometers",
    "mile", "miles",
    "second", "seconds",
    "minute", "minutes",
    "hour", "hours",
    "day", "days",
    "week", "weeks",
    "month", "months",
    "year", "years",
    "foot", "feet",
    "inch", "inches",
    "yard", "yards",
    "liter", "liters",
    "gallon", "gallons",
    "pound", "pounds",
    "ounce", "ounces",
    "gram", "grams",
    "kilogram", "kilograms",
]

# Large number words (from prime_math)
LARGE_NUMBER_WORDS = {
    "million": "*10^6",
    "billion": "*10^9",
    "trillion": "*10^12",
}

# Roman numeral mappings
ROMAN_NUMERALS = {
    'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
    'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
    'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1
}

def roman_to_int(s: str) -> Optional[int]:
    """
    Convert Roman numeral string to integer.

    Args:
        s: Roman numeral string (e.g., "XIV")

    Returns:
        Integer value or None if invalid
    """
    if not s:
        return None
    s = s.upper().strip()

    # Quick validation: only valid Roman numeral chars
    if not all(c in 'MDCLXVI' for c in s):
        return None

    result = 0
    i = 0
    while i < len(s):
        # Check for two-character numerals first
        if i + 1 < len(s) and s[i:i+2] in ROMAN_NUMERALS:
            result += ROMAN_NUMERALS[s[i:i+2]]
            i += 2
        elif s[i] in ROMAN_NUMERALS:
            result += ROMAN_NUMERALS[s[i]]
            i += 1
        else:
            return None
    return result


def int_to_roman(num: int) -> Optional[str]:
    """
    Convert integer to Roman numeral string.

    Args:
        num: Integer (1-3999)

    Returns:
        Roman numeral string or None if out of range
    """
    if num <= 0 or num >= 4000:
        return None

    result = ""
    for numeral, value in [('M', 1000), ('CM', 900), ('D', 500), ('CD', 400),
                           ('C', 100), ('XC', 90), ('L', 50), ('XL', 40),
                           ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1)]:
        while num >= value:
            result += numeral
            num -= value
    return result


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

    # Unit handling
    remove_units: bool = False
    unit_patterns: list[str] = field(default_factory=lambda: [
        r"\\text\{[a-zA-Z]+\}$",  # Trailing \text{units}
        r"\s*(cm|mm|m|km|ft|in|mi|mph|kg|g|lb|oz|s|min|hr|degrees?)$",
    ])

    # Number normalization
    remove_thousands_separator: bool = True  # 1,000 -> 1000
    normalize_decimals: bool = True  # .5 -> 0.5

    # Advanced features (from prime_math)
    convert_unicode: bool = True  # √ -> sqrt, π -> pi, etc.
    handle_large_numbers: bool = True  # million -> *10^6
    handle_mixed_numbers: bool = True  # 7 3/4 -> 7+3/4
    remove_degree_symbol: bool = True  # Remove ^\circ
    remove_percentage: bool = True  # Remove \% and %
    remove_dollar_sign: bool = True  # Remove $ from currency
    extract_from_equals: bool = True  # k = 5 -> 5 (short var prefix)
    normalize_inverse_space: bool = True  # Remove \!
    normalize_double_backslash: bool = True  # \\\\ -> \\


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

    # Convert Unicode symbols to ASCII equivalents (from prime_math)
    if config.convert_unicode:
        for unicode_char, replacement in UNICODE_MAPPINGS.items():
            result = result.replace(unicode_char, replacement)

    # Normalize double backslashes (from prime_math)
    if config.normalize_double_backslash:
        result = result.replace("\\\\", "\\")

    # Remove inverse spaces \! (from prime_math)
    if config.normalize_inverse_space:
        result = result.replace("\\!", "")

    # Remove dollar signs (currency, not math delimiters)
    if config.remove_dollar_sign:
        result = result.replace("\\$", "")
        # Only remove standalone $ not used as math delimiters
        if not (result.startswith("$") and result.endswith("$")):
            result = result.replace("$", "")

    # Remove percentage symbols
    if config.remove_percentage:
        result = result.replace("\\%", "")
        result = result.replace("%", "")

    # Handle large number words (from prime_math)
    if config.handle_large_numbers:
        for word, replacement in LARGE_NUMBER_WORDS.items():
            result = result.replace(word, replacement)

    # Remove degree symbol (from prime_math)
    if config.remove_degree_symbol:
        result = result.replace("^{\\circ}", "")
        result = result.replace("^\\circ", "")
        result = re.sub(r"\^ *\\circ", "", result)

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
    if config.expand_frac_shorthand:
        result = _expand_frac_shorthand(result)

    # Expand sqrt shorthand: \sqrta -> \sqrt{a}
    if config.normalize_sqrt:
        result = _expand_sqrt_shorthand(result)

    # Remove sizing commands
    if config.remove_sizing:
        for cmd in ["\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg"]:
            result = result.replace(cmd, "")

    # Remove enclosing \text{} wrapper (from prime_math)
    if config.remove_text_wrappers:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", result)
        if m is not None:
            result = m.group("text")
        # Also handle inline text wrappers
        result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\textbf\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\textit\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mbox\{([^}]*)\}", r"\1", result)

    if config.remove_mathrm:
        result = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathit\{([^}]*)\}", r"\1", result)

    # Strip enclosing braces {} if present
    if len(result) > 1 and result[0] == "{" and result[-1] == "}":
        result = result[1:-1]

    # Remove units if configured
    if config.remove_units:
        for pattern in config.unit_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        # Also strip common units (from prime_math)
        for unit in UNITS_TO_STRIP:
            result = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", result, flags=re.IGNORECASE)

    # Handle "or" and "and" in answers (from prime_math)
    result = result.replace(" or ", " , ")
    result = result.replace(" and ", " , ")

    # Normalize thousands separators
    if config.remove_thousands_separator:
        result = _strip_thousands_separators(result)

    # Normalize decimals: .5 -> 0.5 (from prime_math)
    if config.normalize_decimals:
        result = result.replace(" .", " 0.")
        result = result.replace("{.", "{0.")
        if result.startswith("."):
            result = "0" + result

    # Extract value from short variable assignments: k = 5 -> 5 (from prime_math)
    if config.extract_from_equals:
        if len(result.split("=")) == 2 and len(result.split("=")[0].strip()) <= 2:
            result = result.split("=")[1].strip()

    # Handle mixed numbers: 7 3/4 -> 7+3/4 (from prime_math)
    if config.handle_mixed_numbers:
        result = _inject_implicit_mixed_number(result)

    # Collapse whitespace
    if config.collapse_whitespace:
        result = re.sub(r"\s+", " ", result)
        # Remove spaces around operators and braces for consistency
        result = re.sub(r"\s*([{}()\[\]+\-*/^_=])\s*", r"\1", result)

    # Final strip
    if config.strip_whitespace:
        result = result.strip()

    return result


def _strip_thousands_separators(expr: str) -> str:
    """
    Strip properly formatted thousand separators from numbers.

    Handles commas that are thousand separators (1,000,000) while
    preserving commas in tuples and other contexts.

    From prime_math.
    """
    # Pattern matches: digit, comma, exactly 3 digits, then end or non-digit
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return expr


def _inject_implicit_mixed_number(expr: str) -> str:
    """
    Convert mixed numbers to addition form.

    E.g., "7 3/4" -> "7+3/4"

    From prime_math.
    """
    p1 = re.compile(r"([0-9]) +([0-9])")
    return p1.sub(r"\1+\2", expr)


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


def normalize_for_numeric_comparison(
    expr: str,
    pi_value: float = math.pi,
) -> Optional[float]:
    """
    Attempt to convert an expression to a float for numeric comparison.

    Handles:
    - Plain numbers: 42, 3.14, -17
    - Fractions: 1/2, \\frac{1}{2}
    - Scientific notation: 1.5e-3
    - Percentages: 50% -> 0.5
    - Pi expressions: 2\\pi -> 2*pi
    - Thousands separators: 1,000 -> 1000

    Args:
        expr: The expression to convert
        pi_value: Value to use for pi (default: math.pi, can use 3.14 for approx)

    Returns:
        Float value or None if conversion fails
    """
    if not expr:
        return None

    expr = str(expr).strip()

    # Remove LaTeX math delimiters
    expr = re.sub(r"^\$+|\$+$", "", expr)
    expr = re.sub(r"^\\+\(|\\+\)$", "", expr)

    # Remove LaTeX percentage
    expr = expr.replace("\\%", "")

    # Handle percentage (do this before removing %)
    if expr.endswith("%"):
        try:
            return float(expr[:-1].replace(",", "")) / 100
        except ValueError:
            pass

    # Strip thousands separators
    expr = _strip_thousands_separators(expr)

    # Handle pi expressions (from prime_math)
    expr = _handle_pi_for_numeric(expr, pi_value)

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
            num_str = latex_frac.group(1)
            denom_str = latex_frac.group(2)
            # Recursively handle pi in numerator/denominator
            num = normalize_for_numeric_comparison(num_str, pi_value)
            denom = normalize_for_numeric_comparison(denom_str, pi_value)
            if num is not None and denom is not None and denom != 0:
                return num / denom
        except ValueError:
            pass

    # Handle LaTeX sqrt: \sqrt{a} or \sqrt[n]{a}
    latex_sqrt = re.match(r"^\\sqrt\{([^}]+)\}$", expr)
    if latex_sqrt:
        try:
            inner = latex_sqrt.group(1)
            inner_val = normalize_for_numeric_comparison(inner, pi_value)
            if inner_val is not None and inner_val >= 0:
                return math.sqrt(inner_val)
        except ValueError:
            pass

    # Handle nth root: \sqrt[n]{a}
    latex_nthroot = re.match(r"^\\sqrt\[([^]]+)\]\{([^}]+)\}$", expr)
    if latex_nthroot:
        try:
            n_str = latex_nthroot.group(1)
            inner = latex_nthroot.group(2)
            n_val = normalize_for_numeric_comparison(n_str, pi_value)
            inner_val = normalize_for_numeric_comparison(inner, pi_value)
            if n_val is not None and inner_val is not None and n_val != 0:
                return inner_val ** (1 / n_val)
        except ValueError:
            pass

    # Handle negative with parentheses: -(3)
    neg_match = re.match(r"^-\((\d+(?:\.\d+)?)\)$", expr)
    if neg_match:
        try:
            return -float(neg_match.group(1))
        except ValueError:
            pass

    # Handle base notation: 123_8 means 123 in base 8, but we just take the number
    if "_" in expr:
        try:
            base_part = expr.split("_")[0]
            return float(base_part)
        except ValueError:
            pass

    return None


def _handle_pi_for_numeric(expr: str, pi_value: float = math.pi) -> str:
    """
    Replace \\pi with numeric value for evaluation.

    Handles cases like:
    - "2\\pi" -> "2*3.14159..."
    - "\\pi" -> "3.14159..."
    - "\\pi/2" -> "3.14159.../2"

    From prime_math.
    """
    if "\\pi" not in expr and "pi" not in expr.lower():
        return expr

    result = expr

    # Replace \pi with pi_value
    idx = result.find("\\pi")
    while idx != -1:
        if idx > 0 and (result[idx - 1].isdigit() or result[idx - 1] == ")"):
            # Previous char is digit or ), insert multiplication
            result = result[:idx] + f"*{pi_value}" + result[idx + 3:]
        else:
            # Just replace \pi
            result = result[:idx] + f"{pi_value}" + result[idx + 3:]
        idx = result.find("\\pi", idx + 1)

    # Also handle plain "pi" (case insensitive)
    result = re.sub(r"\bpi\b", str(pi_value), result, flags=re.IGNORECASE)

    # Try to evaluate the expression
    try:
        # Safe evaluation of simple math expressions
        result = str(eval(result, {"__builtins__": {}}, {"pi": pi_value}))
    except Exception:
        pass

    return result


def try_numeric_with_pi_variants(
    pred: str,
    gt: str,
    tolerance: float = 1e-6,
) -> Optional[bool]:
    """
    Try numeric comparison with different pi values.

    Some problems expect exact pi, others expect 3.14 approximation.
    Try both to be lenient.

    From prime_math.

    Args:
        pred: Predicted value
        gt: Ground truth value
        tolerance: Numeric tolerance

    Returns:
        True if match found, False if no match, None if can't compare
    """
    pi_values = [math.pi, 3.14, 3.14159]

    for pi_val in pi_values:
        pred_num = normalize_for_numeric_comparison(pred, pi_val)
        gt_num = normalize_for_numeric_comparison(gt, pi_val)

        if pred_num is not None and gt_num is not None:
            if abs(pred_num - gt_num) <= tolerance:
                return True
            # Also try relative tolerance
            if gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) <= tolerance:
                return True

    return None


def normalize_answer_string(answer: str) -> str:
    """
    Normalize an answer string for string comparison.

    This applies all normalizations and is suitable for final
    string-based comparison.

    Args:
        answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    config = NormalizationConfig(
        remove_units=True,
        normalize_decimals=True,
    )
    result = normalize_latex(answer, config)

    # Additional normalization: lowercase for text answers
    # Only if it contains letters and isn't LaTeX
    if not "\\" in result and any(c.isalpha() for c in result):
        result = result.lower()

    return result


def normalize_set(expr: str) -> Optional[set]:
    """
    Parse a set expression into a Python set.

    Handles formats like:
    - {1, 2, 3}
    - {2,3,5}
    - \\{2, 3, 5\\}

    Args:
        expr: The expression to parse

    Returns:
        Set of elements or None if not a set
    """
    expr = str(expr).strip()

    # Remove LaTeX braces
    expr = expr.replace("\\{", "{").replace("\\}", "}")

    # Check if it looks like a set
    if not (expr.startswith("{") and expr.endswith("}")):
        return None

    # Extract elements
    inner = expr[1:-1].strip()
    if not inner:
        return set()

    elements = [e.strip() for e in inner.split(",")]
    return set(elements)


def normalize_ratio(expr: str) -> Optional[tuple]:
    """
    Parse a ratio expression into a tuple.

    Handles formats like:
    - 1:3
    - 1 : 3
    - 1:2:3 (multi-part ratios)

    Args:
        expr: The expression to parse

    Returns:
        Tuple of ratio parts or None if not a ratio
    """
    expr = str(expr).strip()

    # Check if it contains colon
    if ":" not in expr:
        return None

    parts = [p.strip() for p in expr.split(":")]

    # Try to convert to numbers for comparison
    try:
        numeric_parts = []
        for p in parts:
            val = normalize_for_numeric_comparison(p)
            if val is not None:
                numeric_parts.append(val)
            else:
                numeric_parts.append(p)
        return tuple(numeric_parts)
    except Exception:
        return tuple(parts)


def simplify_ratio(ratio: tuple) -> tuple:
    """
    Simplify a ratio to its lowest terms.

    Args:
        ratio: Tuple of numeric ratio parts

    Returns:
        Simplified ratio tuple
    """
    import math

    # Check all parts are numeric
    if not all(isinstance(p, (int, float)) for p in ratio):
        return ratio

    # Convert to integers if possible
    int_parts = []
    for p in ratio:
        if float(p).is_integer():
            int_parts.append(int(p))
        else:
            # If any part is not integer, return original
            return ratio

    if not int_parts:
        return ratio

    # Find GCD of all parts
    gcd = int_parts[0]
    for p in int_parts[1:]:
        gcd = math.gcd(gcd, p)

    if gcd > 0:
        return tuple(p // gcd for p in int_parts)
    return tuple(int_parts)


def is_text_answer(answer: str) -> bool:
    """
    Check if an answer is primarily text (not numeric or LaTeX).

    Args:
        answer: The answer string

    Returns:
        True if the answer appears to be text
    """
    answer = str(answer).strip()

    # Empty string
    if not answer:
        return False

    # Contains LaTeX commands
    if "\\" in answer:
        return False

    # Mostly letters
    letters = sum(1 for c in answer if c.isalpha())
    digits = sum(1 for c in answer if c.isdigit())

    return letters > digits


# Common text answer variations (for fuzzy matching)
TEXT_ANSWER_ALIASES = {
    "median": ["median", "the median", "middle value"],
    "mean": ["mean", "average", "the mean", "the average"],
    "mode": ["mode", "the mode", "most frequent"],
    "infinite": ["infinite", "infinity", "inf", "∞", "\\infty"],
    "undefined": ["undefined", "undef", "no solution", "does not exist", "dne"],
    "none": ["none", "no solution", "empty", "∅", "\\emptyset", "{}"],
    "true": ["true", "yes", "correct"],
    "false": ["false", "no", "incorrect"],
}
