# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""
Answer comparison strategies for mathematical verification.

This module provides multiple comparison strategies:
- String comparison (exact and normalized)
- Numeric comparison (with tolerance)
- Symbolic comparison (using sympy)
"""

from __future__ import annotations

import re
import logging
from typing import Optional
import math

from verl.utils.reward_score.math_verify.core import ComparisonMethod
from verl.utils.reward_score.math_verify.normalize import normalize_for_numeric_comparison

logger = logging.getLogger(__name__)


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
    relative_tolerance: float = 1e-9,
) -> dict:
    """
    Compare answers as numeric values with tolerance.

    Useful for floating point answers where string comparison
    would fail due to representation differences.

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        tolerance: Absolute tolerance for comparison
        relative_tolerance: Relative tolerance for comparison

    Returns:
        Dict with 'match' bool, 'method', and optional numeric values
    """
    pred_val = normalize_for_numeric_comparison(pred)
    gt_val = normalize_for_numeric_comparison(gt)

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

    # Use both absolute and relative tolerance
    abs_diff = abs(pred_val - gt_val)
    rel_diff = abs_diff / max(abs(gt_val), 1e-10)

    match = abs_diff <= tolerance or rel_diff <= relative_tolerance

    return {
        "match": match,
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


def compare_answers(
    pred: str,
    gt: str,
    pred_normalized: Optional[str] = None,
    gt_normalized: Optional[str] = None,
    enable_symbolic: bool = True,
    enable_numeric: bool = True,
    numeric_tolerance: float = 1e-6,
    numeric_relative_tolerance: float = 1e-9,
) -> dict:
    """
    Compare answers using multiple strategies with fallback.

    Order of comparison:
    1. String comparison (exact, then normalized)
    2. Numeric comparison (if both can be parsed as numbers)
    3. Symbolic comparison (if sympy available)

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        pred_normalized: Pre-normalized prediction
        gt_normalized: Pre-normalized ground truth
        enable_symbolic: Whether to try symbolic comparison
        enable_numeric: Whether to try numeric comparison
        numeric_tolerance: Absolute tolerance for numeric comparison
        numeric_relative_tolerance: Relative tolerance for numeric comparison

    Returns:
        Dict with 'match', 'method', and optional metadata
    """
    # Strategy 1: String comparison
    result = compare_strings(pred, gt, pred_normalized, gt_normalized)
    if result["match"]:
        return result

    # Strategy 2: Numeric comparison
    if enable_numeric:
        result = compare_numeric(
            pred, gt,
            tolerance=numeric_tolerance,
            relative_tolerance=numeric_relative_tolerance,
        )
        if result["match"]:
            return result

    # Strategy 3: Symbolic comparison
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
