# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""
Math Answer Extraction and Verification for RLVR.

This module provides robust extraction and verification of mathematical answers
from model outputs, designed for use in reinforcement learning with verifiable
rewards (RLVR).

Key features:
- Multiple extraction strategies (boxed, patterns, heuristics)
- Multiple comparison strategies (string, numeric, symbolic)
- Detailed diagnostics for debugging reward signals
- Configurable normalization pipeline
"""

from verl.utils.reward_score.math_verify.core import (
    MathVerifier,
    VerifierConfig,
    VerificationResult,
    ExtractionResult,
    ExtractionMethod,
    ComparisonMethod,
    compute_score,
)
from verl.utils.reward_score.math_verify.extract import (
    extract_boxed,
    extract_by_pattern,
    extract_last_number,
    extract_answer,
)
from verl.utils.reward_score.math_verify.compare import (
    compare_strings,
    compare_numeric,
    compare_symbolic,
    compare_tuple_or_list,
    compare_sets,
    compare_ratios,
    compare_roman_numerals,
    compare_text_answers,
    compare_answers,
    format_interval,
    fuzzy_string_match,
    DEFAULT_FUZZY_THRESHOLD,
)
from verl.utils.reward_score.math_verify.normalize import (
    normalize_latex,
    NormalizationConfig,
    normalize_for_numeric_comparison,
    try_numeric_with_pi_variants,
    normalize_answer_string,
    normalize_set,
    normalize_ratio,
    simplify_ratio,
    roman_to_int,
    int_to_roman,
    is_text_answer,
    TEXT_ANSWER_ALIASES,
)

__all__ = [
    "MathVerifier",
    "VerifierConfig",
    "VerificationResult",
    "ExtractionResult",
    "ExtractionMethod",
    "ComparisonMethod",
    "extract_boxed",
    "extract_by_pattern",
    "extract_last_number",
    "extract_answer",
    "compare_strings",
    "compare_numeric",
    "compare_symbolic",
    "compare_tuple_or_list",
    "compare_sets",
    "compare_ratios",
    "compare_roman_numerals",
    "compare_text_answers",
    "compare_answers",
    "format_interval",
    "fuzzy_string_match",
    "DEFAULT_FUZZY_THRESHOLD",
    "normalize_latex",
    "NormalizationConfig",
    "normalize_for_numeric_comparison",
    "try_numeric_with_pi_variants",
    "normalize_answer_string",
    "normalize_set",
    "normalize_ratio",
    "simplify_ratio",
    "roman_to_int",
    "int_to_roman",
    "is_text_answer",
    "TEXT_ANSWER_ALIASES",
    "compute_score",
]

__version__ = "0.1.0"
