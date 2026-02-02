# Copyright 2024
# Licensed under the Apache License, Version 2.0
# Incorporates comparison logic from prime_math (PRIME team, OpenAI prm800k, Microsoft ToRA)

"""
Answer comparison strategies for mathematical verification.

This module provides multiple comparison strategies:
- String comparison (exact and normalized)
- Numeric comparison (with tolerance, pi-aware)
- Symbolic comparison (using sympy)
- Tuple/list comparison
- Interval comparison

Features incorporated from prime_math:
- Pi-aware numeric comparison (tries math.pi and 3.14)
- Tuple/list element-wise comparison
- Interval format handling
- Percentage tolerance (x, x/100, x*100)
"""

from __future__ import annotations

import re
import logging
from typing import Optional, List
import math
from difflib import SequenceMatcher

from verl.utils.reward_score.math_verify.core import ComparisonMethod
from verl.utils.reward_score.math_verify.normalize import (
    normalize_for_numeric_comparison,
    try_numeric_with_pi_variants,
    _strip_thousands_separators,
    normalize_set,
    normalize_ratio,
    simplify_ratio,
    roman_to_int,
    is_text_answer,
    TEXT_ANSWER_ALIASES,
)

logger = logging.getLogger(__name__)

# Characters that indicate tuple/interval expressions
TUPLE_CHARS = "()[]"

# Default fuzzy matching threshold (85% similarity)
DEFAULT_FUZZY_THRESHOLD = 0.85


def fuzzy_string_match(
    s1: str,
    s2: str,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ignore_case: bool = True,
    ignore_whitespace: bool = True,
    ignore_punctuation: bool = True,
) -> dict:
    """
    Perform fuzzy string matching between two strings.

    Uses SequenceMatcher to compute similarity ratio.

    Args:
        s1: First string
        s2: Second string
        threshold: Minimum similarity ratio to consider a match (0.0-1.0)
        ignore_case: Whether to ignore case differences
        ignore_whitespace: Whether to normalize whitespace
        ignore_punctuation: Whether to ignore punctuation

    Returns:
        Dict with 'match' bool, 'similarity' ratio, and processing details
    """
    if not s1 or not s2:
        return {"match": False, "similarity": 0.0}

    # Preprocessing
    a, b = str(s1), str(s2)

    if ignore_case:
        a, b = a.lower(), b.lower()

    if ignore_whitespace:
        a = " ".join(a.split())
        b = " ".join(b.split())

    if ignore_punctuation:
        a = re.sub(r'[^\w\s]', '', a)
        b = re.sub(r'[^\w\s]', '', b)

    # Exact match after preprocessing
    if a == b:
        return {
            "match": True,
            "similarity": 1.0,
            "exact_after_preprocessing": True,
        }

    # Compute similarity
    ratio = SequenceMatcher(None, a, b).ratio()

    return {
        "match": ratio >= threshold,
        "similarity": ratio,
        "threshold": threshold,
        "preprocessed": {
            "s1": a,
            "s2": b,
        },
    }


def compare_strings(
    pred: str,
    gt: str,
    pred_normalized: Optional[str] = None,
    gt_normalized: Optional[str] = None,
) -> dict:
    """
    Compare answers using string equality.

    Tries exact match first, then normalized match.

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        pred_normalized: Pre-normalized prediction (optional)
        gt_normalized: Pre-normalized ground truth (optional)

    Returns:
        Dict with 'match' bool and 'method' ComparisonMethod
    """
    # Exact match
    if pred == gt:
        return {
            "match": True,
            "method": ComparisonMethod.STRING_EXACT,
        }

    # Normalized match
    if pred_normalized is not None and gt_normalized is not None:
        if pred_normalized == gt_normalized:
            return {
                "match": True,
                "method": ComparisonMethod.STRING_NORMALIZED,
            }

    return {
        "match": False,
        "method": ComparisonMethod.STRING_NORMALIZED,
    }


def compare_numeric(
    pred: str,
    gt: str,
    tolerance: float = 1e-6,
    relative_tolerance: float = 1e-2,
    include_percentage: bool = True,
    try_pi_variants: bool = True,
) -> dict:
    """
    Compare answers as numeric values with tolerance.

    Useful for floating point answers where string comparison
    would fail due to representation differences.

    Features from prime_math:
    - Pi-aware comparison (tries math.pi and 3.14)
    - Percentage tolerance (compares x with x/100 and x*100)

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        tolerance: Absolute tolerance for comparison
        relative_tolerance: Relative tolerance for comparison
        include_percentage: Whether to try percentage variants (x, x/100, x*100)
        try_pi_variants: Whether to try different pi values

    Returns:
        Dict with 'match' bool, 'method', and optional numeric values
    """
    # First try with standard pi
    pred_val = normalize_for_numeric_comparison(pred)
    gt_val = normalize_for_numeric_comparison(gt)

    # If pi expressions detected, try multiple pi values
    if try_pi_variants and ("\\pi" in str(pred) or "\\pi" in str(gt) or "pi" in str(pred).lower() or "pi" in str(gt).lower()):
        pi_result = try_numeric_with_pi_variants(pred, gt, tolerance, relative_tolerance)
        if pi_result is True:
            return {
                "match": True,
                "method": ComparisonMethod.NUMERIC,
                "metadata": {"pi_variant": True},
            }

    if pred_val is None or gt_val is None:
        return {
            "match": False,
            "method": ComparisonMethod.NUMERIC,
            "metadata": {
                "pred_numeric": pred_val,
                "gt_numeric": gt_val,
                "error": "conversion_failed",
            }
        }

    # Check for exact equality first (handles special cases like inf)
    if pred_val == gt_val:
        return {
            "match": True,
            "method": ComparisonMethod.NUMERIC,
            "metadata": {
                "pred_numeric": pred_val,
                "gt_numeric": gt_val,
            }
        }

    # Handle special float values
    if math.isnan(pred_val) or math.isnan(gt_val):
        return {
            "match": False,
            "method": ComparisonMethod.NUMERIC,
            "metadata": {
                "pred_numeric": pred_val,
                "gt_numeric": gt_val,
                "error": "nan_values",
            }
        }

    if math.isinf(pred_val) or math.isinf(gt_val):
        # Infinities must match exactly (already checked above)
        return {
            "match": False,
            "method": ComparisonMethod.NUMERIC,
            "metadata": {
                "pred_numeric": pred_val,
                "gt_numeric": gt_val,
            }
        }

    # Try percentage variants if enabled (from prime_math)
    if include_percentage:
        gt_variants = [gt_val / 100, gt_val, gt_val * 100]
    else:
        gt_variants = [gt_val]

    for gt_variant in gt_variants:
        try:
            if math.isclose(pred_val, gt_variant, rel_tol=relative_tolerance, abs_tol=tolerance):
                return {
                    "match": True,
                    "method": ComparisonMethod.NUMERIC,
                    "metadata": {
                        "pred_numeric": pred_val,
                        "gt_numeric": gt_variant,
                        "percentage_variant": gt_variant != gt_val,
                    }
                }
        except Exception:
            continue

    # Calculate diff for metadata
    abs_diff = abs(pred_val - gt_val)
    rel_diff = abs_diff / max(abs(gt_val), 1e-10)

    return {
        "match": False,
        "method": ComparisonMethod.NUMERIC,
        "metadata": {
            "pred_numeric": pred_val,
            "gt_numeric": gt_val,
            "absolute_diff": abs_diff,
            "relative_diff": rel_diff,
        }
    }


def compare_symbolic(pred: str, gt: str) -> dict:
    """
    Compare answers using symbolic mathematics (sympy).

    Attempts to parse both expressions as LaTeX and check
    mathematical equivalence. Handles algebraic simplification.

    Args:
        pred: Predicted answer (LaTeX)
        gt: Ground truth answer (LaTeX)

    Returns:
        Dict with 'match' bool and 'method'
    """
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, N
    except ImportError:
        logger.warning("sympy not available for symbolic comparison")
        return {
            "match": False,
            "method": ComparisonMethod.SYMBOLIC,
            "metadata": {"error": "sympy_not_available"},
        }

    try:
        # Clean up expressions for parsing
        pred_clean = _prepare_for_sympy(pred)
        gt_clean = _prepare_for_sympy(gt)

        # Parse expressions
        pred_expr = parse_latex(pred_clean)
        gt_expr = parse_latex(gt_clean)

        # Try direct equality
        if pred_expr == gt_expr:
            return {
                "match": True,
                "method": ComparisonMethod.SYMBOLIC,
            }

        # Try simplification
        diff = simplify(pred_expr - gt_expr)
        if diff == 0:
            return {
                "match": True,
                "method": ComparisonMethod.SYMBOLIC,
                "metadata": {"simplified": True},
            }

        # Try numerical evaluation for expressions with no free symbols
        if not pred_expr.free_symbols and not gt_expr.free_symbols:
            pred_num = complex(N(pred_expr))
            gt_num = complex(N(gt_expr))

            # Compare with tolerance
            if abs(pred_num - gt_num) < 1e-9:
                return {
                    "match": True,
                    "method": ComparisonMethod.SYMBOLIC,
                    "metadata": {
                        "numeric_evaluation": True,
                        "pred_value": pred_num,
                        "gt_value": gt_num,
                    },
                }

        return {
            "match": False,
            "method": ComparisonMethod.SYMBOLIC,
            "metadata": {
                "pred_parsed": str(pred_expr),
                "gt_parsed": str(gt_expr),
            },
        }

    except Exception as e:
        logger.debug(f"Symbolic comparison failed: {e}")
        return {
            "match": False,
            "method": ComparisonMethod.SYMBOLIC,
            "metadata": {"error": str(e)},
        }


def _prepare_for_sympy(expr: str) -> str:
    """
    Prepare a LaTeX expression for sympy parsing.

    Handles common issues that cause parse failures.
    """
    result = expr.strip()

    # Remove dollar signs
    result = re.sub(r"^\$+|\$+$", "", result)

    # Remove \left and \right (sympy doesn't need them)
    result = result.replace("\\left", "").replace("\\right", "")

    # Normalize fraction types
    result = result.replace("\\dfrac", "\\frac")
    result = result.replace("\\tfrac", "\\frac")

    # Handle \cdot -> *
    result = result.replace("\\cdot", "*")

    # Handle \times -> *
    result = result.replace("\\times", "*")

    # Handle \div -> /
    result = result.replace("\\div", "/")

    return result


def compare_tuple_or_list(
    pred: str,
    gt: str,
    tolerance: float = 1e-6,
) -> dict:
    """
    Compare tuple or list expressions element-wise.

    Handles formats like:
    - (1, 2, 3) vs (1, 2, 3)
    - [1, 2] vs [1, 2]
    - Point(1, 2) vs (1, 2)

    From prime_math.

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        tolerance: Numeric tolerance for element comparison

    Returns:
        Dict with 'match' bool and 'method'
    """
    pred = str(pred).strip()
    gt = str(gt).strip()

    # Handle sympy Point format
    if pred.startswith("Point") and gt.startswith("(") and gt.endswith(")"):
        pred = pred[pred.find("("):]

    # Check if both are tuple/list format
    if not (pred and gt):
        return {"match": False, "method": ComparisonMethod.FAILED}

    # Check bracket matching
    if pred[0] in "([" and pred[-1] in ")]" and gt[0] in "([" and gt[-1] in ")]":
        # Same bracket type or compatible
        if pred[0] != gt[0] or pred[-1] != gt[-1]:
            # Different bracket types - strip and compare
            pred_inner = pred[1:-1]
            gt_inner = gt[1:-1]
        else:
            pred_inner = pred[1:-1]
            gt_inner = gt[1:-1]

        # Split by comma
        pred_parts = [p.strip() for p in pred_inner.split(",")]
        gt_parts = [g.strip() for g in gt_inner.split(",")]

        if len(pred_parts) != len(gt_parts):
            return {"match": False, "method": ComparisonMethod.FAILED}

        # Compare each element
        for pred_elem, gt_elem in zip(pred_parts, gt_parts):
            # Try numeric comparison first
            pred_num = normalize_for_numeric_comparison(pred_elem)
            gt_num = normalize_for_numeric_comparison(gt_elem)

            if pred_num is not None and gt_num is not None:
                if not math.isclose(pred_num, gt_num, rel_tol=1e-4, abs_tol=tolerance):
                    return {"match": False, "method": ComparisonMethod.NUMERIC}
            elif pred_elem.strip() != gt_elem.strip():
                return {"match": False, "method": ComparisonMethod.STRING_NORMALIZED}

        return {
            "match": True,
            "method": ComparisonMethod.NUMERIC,
            "metadata": {"tuple_comparison": True},
        }

    # Check for comma-separated values without brackets
    if "," in pred and "," in gt:
        pred_parts = [p.strip() for p in pred.split(",")]
        gt_parts = [g.strip() for g in gt.split(",")]

        if len(pred_parts) == len(gt_parts):
            all_match = True
            for pred_elem, gt_elem in zip(pred_parts, gt_parts):
                elem_result = compare_answers(
                    pred_elem, gt_elem,
                    enable_symbolic=False,
                    numeric_tolerance=tolerance,
                )
                if not elem_result["match"]:
                    all_match = False
                    break

            if all_match:
                return {
                    "match": True,
                    "method": ComparisonMethod.NUMERIC,
                    "metadata": {"comma_separated": True},
                }

    return {"match": False, "method": ComparisonMethod.FAILED}


def format_interval(prediction: str) -> str:
    """
    Convert sympy Interval format to standard notation.

    From prime_math.

    Args:
        prediction: Expression that might be in Interval format

    Returns:
        Converted expression
    """
    patterns = {
        "Interval(": (r"^Interval\((.*)\)$", "[", "]"),
        "Interval.Ropen(": (r"^Interval\.Ropen\((.*)\)$", "[", ")"),
        "Interval.Lopen(": (r"^Interval\.Lopen\((.*)\)$", "(", "]"),
        "Interval.open(": (r"^Interval\.open\((.*)\)$", "(", ")"),
    }

    for key, (pattern, left, right) in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner = match.group(1)
            return f"{left}{inner}{right}"

    return prediction


def compare_sets(pred: str, gt: str) -> dict:
    """
    Compare set expressions (order-independent).

    Handles formats like:
    - {1, 2, 3} vs {3, 2, 1}
    - {2,3,5} vs {5,3,2}
    - \\{a, b\\} vs \\{b, a\\}

    Args:
        pred: Predicted answer
        gt: Ground truth answer

    Returns:
        Dict with 'match' bool and 'method'
    """
    pred_set = normalize_set(pred)
    gt_set = normalize_set(gt)

    if pred_set is None or gt_set is None:
        return {"match": False, "method": ComparisonMethod.FAILED}

    # Direct set comparison (order-independent)
    if pred_set == gt_set:
        return {
            "match": True,
            "method": ComparisonMethod.STRING_NORMALIZED,
            "metadata": {"set_comparison": True},
        }

    # Try numeric comparison of elements
    if len(pred_set) == len(gt_set):
        pred_numeric = set()
        gt_numeric = set()

        for elem in pred_set:
            val = normalize_for_numeric_comparison(elem)
            pred_numeric.add(val if val is not None else elem)

        for elem in gt_set:
            val = normalize_for_numeric_comparison(elem)
            gt_numeric.add(val if val is not None else elem)

        if pred_numeric == gt_numeric:
            return {
                "match": True,
                "method": ComparisonMethod.NUMERIC,
                "metadata": {"set_comparison": True, "numeric_elements": True},
            }

    return {"match": False, "method": ComparisonMethod.FAILED}


def compare_ratios(pred: str, gt: str, tolerance: float = 1e-6) -> dict:
    """
    Compare ratio expressions.

    Handles formats like:
    - 1:3 vs 1:3
    - 2:6 vs 1:3 (equivalent ratios)
    - 1:2:3 vs 2:4:6

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        tolerance: Numeric tolerance for comparison

    Returns:
        Dict with 'match' bool and 'method'
    """
    pred_ratio = normalize_ratio(pred)
    gt_ratio = normalize_ratio(gt)

    if pred_ratio is None or gt_ratio is None:
        return {"match": False, "method": ComparisonMethod.FAILED}

    # Must have same number of parts
    if len(pred_ratio) != len(gt_ratio):
        return {"match": False, "method": ComparisonMethod.FAILED}

    # Direct comparison
    if pred_ratio == gt_ratio:
        return {
            "match": True,
            "method": ComparisonMethod.NUMERIC,
            "metadata": {"ratio_comparison": True},
        }

    # Try simplified comparison
    pred_simplified = simplify_ratio(pred_ratio)
    gt_simplified = simplify_ratio(gt_ratio)

    if pred_simplified == gt_simplified:
        return {
            "match": True,
            "method": ComparisonMethod.NUMERIC,
            "metadata": {"ratio_comparison": True, "simplified": True},
        }

    # Try proportional comparison (all ratios between parts should be equal)
    if all(isinstance(p, (int, float)) for p in pred_ratio) and \
       all(isinstance(g, (int, float)) for g in gt_ratio):
        try:
            # Check if pred/gt ratios are all equal
            ratios = [p / g if g != 0 else None for p, g in zip(pred_ratio, gt_ratio)]
            if all(r is not None for r in ratios):
                first_ratio = ratios[0]
                if all(abs(r - first_ratio) < tolerance for r in ratios):
                    return {
                        "match": True,
                        "method": ComparisonMethod.NUMERIC,
                        "metadata": {"ratio_comparison": True, "proportional": True},
                    }
        except Exception:
            pass

    return {"match": False, "method": ComparisonMethod.FAILED}


def compare_roman_numerals(pred: str, gt: str) -> dict:
    """
    Compare Roman numeral expressions.

    Handles formats like:
    - XIV vs 14
    - IV vs IV
    - 4 vs IV

    Args:
        pred: Predicted answer
        gt: Ground truth answer

    Returns:
        Dict with 'match' bool and 'method'
    """
    pred = str(pred).strip()
    gt = str(gt).strip()

    # Try to convert both to integers
    pred_int = roman_to_int(pred)
    gt_int = roman_to_int(gt)

    # If pred is Roman numeral
    if pred_int is not None:
        # Compare with gt as Roman
        if gt_int is not None and pred_int == gt_int:
            return {
                "match": True,
                "method": ComparisonMethod.STRING_NORMALIZED,
                "metadata": {"roman_numeral": True},
            }
        # Compare with gt as integer
        try:
            gt_as_int = int(gt)
            if pred_int == gt_as_int:
                return {
                    "match": True,
                    "method": ComparisonMethod.NUMERIC,
                    "metadata": {"roman_numeral": True, "roman_to_int": True},
                }
        except ValueError:
            pass

    # If gt is Roman numeral and pred is integer
    if gt_int is not None and pred_int is None:
        try:
            pred_as_int = int(pred)
            if pred_as_int == gt_int:
                return {
                    "match": True,
                    "method": ComparisonMethod.NUMERIC,
                    "metadata": {"roman_numeral": True, "int_to_roman": True},
                }
        except ValueError:
            pass

    return {"match": False, "method": ComparisonMethod.FAILED}


def compare_text_answers(
    pred: str,
    gt: str,
    fuzzy_threshold: float = 0.85,
    enable_fuzzy: bool = True,
) -> dict:
    """
    Compare text-based answers with fuzzy matching.

    Handles:
    - Case insensitivity: "Median" vs "median"
    - Common aliases: "average" vs "mean"
    - Whitespace normalization
    - Fuzzy matching for typos/variations (configurable threshold)

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        fuzzy_threshold: Minimum similarity ratio for fuzzy match (0.0-1.0)
        enable_fuzzy: Whether to try fuzzy matching

    Returns:
        Dict with 'match' bool and 'method'
    """
    pred = str(pred).strip().lower()
    gt = str(gt).strip().lower()

    # Direct match
    if pred == gt:
        return {
            "match": True,
            "method": ComparisonMethod.STRING_NORMALIZED,
        }

    # Check alias groups
    for canonical, aliases in TEXT_ANSWER_ALIASES.items():
        aliases_lower = [a.lower() for a in aliases]
        if pred in aliases_lower and gt in aliases_lower:
            return {
                "match": True,
                "method": ComparisonMethod.STRING_NORMALIZED,
                "metadata": {"text_alias": canonical},
            }

    # Try without spaces and punctuation
    pred_clean = re.sub(r'[^a-z0-9]', '', pred)
    gt_clean = re.sub(r'[^a-z0-9]', '', gt)

    if pred_clean == gt_clean and pred_clean:
        return {
            "match": True,
            "method": ComparisonMethod.STRING_NORMALIZED,
            "metadata": {"cleaned_match": True},
        }

    # Fuzzy string matching using SequenceMatcher
    if enable_fuzzy and len(pred) >= 3 and len(gt) >= 3:
        # Compare original strings
        ratio = SequenceMatcher(None, pred, gt).ratio()
        if ratio >= fuzzy_threshold:
            return {
                "match": True,
                "method": ComparisonMethod.STRING_NORMALIZED,
                "metadata": {"fuzzy_match": True, "similarity": ratio},
            }

        # Also try cleaned versions
        if pred_clean and gt_clean:
            ratio_clean = SequenceMatcher(None, pred_clean, gt_clean).ratio()
            if ratio_clean >= fuzzy_threshold:
                return {
                    "match": True,
                    "method": ComparisonMethod.STRING_NORMALIZED,
                    "metadata": {"fuzzy_match": True, "similarity": ratio_clean, "cleaned": True},
                }

    return {"match": False, "method": ComparisonMethod.FAILED}


def compare_answers(
    pred: str,
    gt: str,
    pred_normalized: Optional[str] = None,
    gt_normalized: Optional[str] = None,
    enable_symbolic: bool = True,
    enable_numeric: bool = True,
    enable_tuple: bool = True,
    enable_set: bool = True,
    enable_ratio: bool = True,
    enable_roman: bool = True,
    enable_text: bool = True,
    numeric_tolerance: float = 1e-6,
    numeric_relative_tolerance: float = 1e-2,
) -> dict:
    """
    Compare answers using multiple strategies with fallback.

    Order of comparison:
    1. String comparison (exact, then normalized)
    2. Numeric comparison (if both can be parsed as numbers)
    3. Set comparison (if both look like sets)
    4. Ratio comparison (if both contain colons)
    5. Roman numeral comparison
    6. Text answer comparison (for word answers)
    7. Tuple/list comparison (if both look like tuples)
    8. Symbolic comparison (if sympy available)

    Features:
    - Pi-aware numeric comparison
    - Tuple/list element-wise comparison
    - Interval format handling
    - Percentage tolerance
    - Set comparison (order-independent)
    - Ratio comparison (with simplification)
    - Roman numeral handling
    - Text answer aliases

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        pred_normalized: Pre-normalized prediction
        gt_normalized: Pre-normalized ground truth
        enable_symbolic: Whether to try symbolic comparison
        enable_numeric: Whether to try numeric comparison
        enable_tuple: Whether to try tuple comparison
        enable_set: Whether to try set comparison
        enable_ratio: Whether to try ratio comparison
        enable_roman: Whether to try Roman numeral comparison
        enable_text: Whether to try text answer comparison
        numeric_tolerance: Absolute tolerance for numeric comparison
        numeric_relative_tolerance: Relative tolerance for numeric comparison

    Returns:
        Dict with 'match', 'method', and optional metadata
    """
    # Convert Interval formats
    pred = format_interval(str(pred))
    gt = format_interval(str(gt))

    # Strategy 1: String comparison
    result = compare_strings(pred, gt, pred_normalized, gt_normalized)
    if result["match"]:
        return result

    # Also try case-insensitive and space-normalized
    if pred.strip().lower() == gt.strip().lower():
        return {
            "match": True,
            "method": ComparisonMethod.STRING_NORMALIZED,
        }
    if pred.replace(" ", "") == gt.replace(" ", ""):
        return {
            "match": True,
            "method": ComparisonMethod.STRING_NORMALIZED,
        }

    # Strategy 2: Numeric comparison
    if enable_numeric:
        result = compare_numeric(
            pred, gt,
            tolerance=numeric_tolerance,
            relative_tolerance=numeric_relative_tolerance,
        )
        if result["match"]:
            return result

    # Strategy 3: Set comparison
    if enable_set and ("{" in pred or "{" in gt or "\\{" in pred or "\\{" in gt):
        result = compare_sets(pred, gt)
        if result["match"]:
            return result

    # Strategy 4: Ratio comparison
    if enable_ratio and (":" in pred or ":" in gt):
        result = compare_ratios(pred, gt, tolerance=numeric_tolerance)
        if result["match"]:
            return result

    # Strategy 5: Roman numeral comparison
    if enable_roman:
        result = compare_roman_numerals(pred, gt)
        if result["match"]:
            return result

    # Strategy 6: Text answer comparison
    if enable_text and (is_text_answer(pred) or is_text_answer(gt)):
        result = compare_text_answers(pred, gt)
        if result["match"]:
            return result

    # Strategy 7: Tuple/list comparison
    if enable_tuple and ("," in pred or "," in gt or pred.startswith("(") or pred.startswith("[")):
        result = compare_tuple_or_list(pred, gt, tolerance=numeric_tolerance)
        if result["match"]:
            return result

    # Strategy 8: Symbolic comparison
    if enable_symbolic:
        result = compare_symbolic(pred, gt)
        if result["match"]:
            return result

    # All strategies failed
    return {
        "match": False,
        "method": ComparisonMethod.FAILED,
        "metadata": {
            "strategies_tried": [
                "string",
                "numeric" if enable_numeric else None,
                "set" if enable_set else None,
                "ratio" if enable_ratio else None,
                "roman" if enable_roman else None,
                "text" if enable_text else None,
                "tuple" if enable_tuple else None,
                "symbolic" if enable_symbolic else None,
            ],
        },
    }


def is_mathematically_equivalent(
    expr1: str,
    expr2: str,
    methods: Optional[list[str]] = None,
) -> bool:
    """
    Convenience function to check if two expressions are equivalent.

    Args:
        expr1: First expression
        expr2: Second expression
        methods: List of methods to try ("string", "numeric", "symbolic")
                 Default is all methods.

    Returns:
        True if expressions are equivalent by any method
    """
    if methods is None:
        methods = ["string", "numeric", "symbolic"]

    result = compare_answers(
        pred=expr1,
        gt=expr2,
        enable_numeric="numeric" in methods,
        enable_symbolic="symbolic" in methods,
    )

    return result["match"]
