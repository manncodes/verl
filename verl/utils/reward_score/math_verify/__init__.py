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
)
from verl.utils.reward_score.math_verify.normalize import (
    normalize_latex,
    NormalizationConfig,
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
    "normalize_latex",
    "NormalizationConfig",
    "compute_score",
]

__version__ = "0.1.0"
