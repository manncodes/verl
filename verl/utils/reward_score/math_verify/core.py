# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Core types and main verifier class."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

logger = logging.getLogger(__name__)



def unwrap_ground_truth(gt: str) -> str:
    """Unwrap JSON array format if present."""
    gt = gt.strip()
    if gt.startswith('[') and gt.endswith(']'):
        try:
            parsed = json.loads(gt)
            if isinstance(parsed, list) and len(parsed) == 1:
                return str(parsed[0])
        except json.JSONDecodeError:
            pass
    return gt


class ExtractionMethod(Enum):
    """Method used to extract the answer from model output."""
    BOXED = auto()
    PATTERN = auto()
    LAST_NUMBER = auto()
    FALLBACK = auto()
    FAILED = auto()


class ComparisonMethod(Enum):
    """Method used to compare predicted and ground truth answers."""
    STRING_EXACT = auto()
    STRING_NORMALIZED = auto()
    NUMERIC = auto()
    SYMBOLIC = auto()
    FAILED = auto()


@dataclass(frozen=True)
class ExtractionResult:
    """Result of answer extraction from model output."""
    answer: Optional[str]
    method: ExtractionMethod
    raw_match: Optional[str] = None
    confidence: float = 1.0

    @property
    def success(self) -> bool:
        return self.answer is not None and self.method != ExtractionMethod.FAILED


@dataclass(frozen=True)
class VerificationResult:
    """Complete result of answer verification."""
    correct: bool
    score: float

    # Extraction details
    extracted_answer: Optional[str]
    extraction_method: ExtractionMethod

    # Comparison details
    comparison_method: ComparisonMethod
    ground_truth: str

    # Normalized forms (for debugging)
    pred_normalized: Optional[str] = None
    gt_normalized: Optional[str] = None

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "correct": self.correct,
            "score": self.score,
            "pred": self.extracted_answer,
            "pred_normalized": self.pred_normalized,
            "ground_truth": self.ground_truth,
            "gt_normalized": self.gt_normalized,
            "extraction_method": self.extraction_method.name,
            "comparison_method": self.comparison_method.name,
            **self.metadata,
        }


@dataclass
class VerifierConfig:
    """Configuration for the math verifier."""

    # Extraction settings
    extraction_patterns: list[str] = field(default_factory=lambda: [
        r"(?i)\*{0,2}(?:final\s+)?answer\*{0,2}\s*[:=]\s*(.+?)(?:\n|$)",
        r"(?i)(?:the\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|,|\n|$)",
        r"(?i)=\s*(.+?)(?:\n|$)",
    ])

    # Comparison settings
    enable_symbolic: bool = True
    enable_numeric: bool = True
    numeric_tolerance: float = 1e-6
    numeric_relative_tolerance: float = 1e-4  # Reasonable tolerance for practical comparisons (e.g. sqrt(3) vs 1.732)

    # Score settings
    correct_score: float = 1.0
    incorrect_score: float = 0.0
    extraction_failed_score: float = 0.0

    # Truncation (use with caution)
    max_solution_length: Optional[int] = None
    search_window: Optional[int] = None  # Search only last N chars for answer


class MathVerifier:
    """
    Main verifier class for mathematical answer verification.

    Supports multiple extraction and comparison strategies with
    configurable fallback behavior.

    Example:
        >>> verifier = MathVerifier()
        >>> result = verifier.verify(
        ...     solution="Let x = 2. Then x^2 = \\boxed{4}",
        ...     ground_truth="4"
        ... )
        >>> result.correct
        True
    """

    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()
        self._symbolic_available = self._check_symbolic_available()

        if self.config.enable_symbolic and not self._symbolic_available:
            logger.info(
                "Symbolic comparison requested but sympy not available. "
                "Falling back to string/numeric comparison only."
            )

    def _check_symbolic_available(self) -> bool:
        """Check if symbolic math libraries are available."""
        try:
            import sympy
            from sympy.parsing.latex import parse_latex
            return True
        except ImportError:
            return False

    def verify(
        self,
        solution: str,
        ground_truth: str,
        *,
        extraction_hint: Optional[ExtractionMethod] = None,
    ) -> VerificationResult:
        """
        Verify if the solution contains the correct answer.

        Args:
            solution: The model's solution string
            ground_truth: The expected answer
            extraction_hint: Optional hint for which extraction method to try first

        Returns:
            VerificationResult with detailed information about the verification
        """
        from verl.utils.reward_score.math_verify.extract import extract_answer
        from verl.utils.reward_score.math_verify.compare import compare_answers
        from verl.utils.reward_score.math_verify.normalize import normalize_latex, NormalizationConfig

        # normalizing and parsing ground truth
        ground_truth = unwrap_ground_truth(ground_truth)

        # Apply truncation if configured
        search_text = solution
        if self.config.search_window is not None:
            search_text = solution[-self.config.search_window:]
        if self.config.max_solution_length is not None:
            search_text = search_text[-self.config.max_solution_length:]

        # Extract answer
        extraction = extract_answer(
            search_text,
            patterns=self.config.extraction_patterns,
            hint=extraction_hint,
        )

        if not extraction.success:
            return VerificationResult(
                correct=False,
                score=self.config.extraction_failed_score,
                extracted_answer=None,
                extraction_method=ExtractionMethod.FAILED,
                comparison_method=ComparisonMethod.FAILED,
                ground_truth=ground_truth,
                metadata={"error": "extraction_failed"},
            )

        # Normalize both answers
        norm_config = NormalizationConfig()
        pred_normalized = normalize_latex(extraction.answer, norm_config)
        gt_normalized = normalize_latex(ground_truth, norm_config)

        # Compare answers using multiple strategies
        comparison = compare_answers(
            pred=extraction.answer,
            pred_normalized=pred_normalized,
            gt=ground_truth,
            gt_normalized=gt_normalized,
            enable_symbolic=self.config.enable_symbolic and self._symbolic_available,
            enable_numeric=self.config.enable_numeric,
            numeric_tolerance=self.config.numeric_tolerance,
            numeric_relative_tolerance=self.config.numeric_relative_tolerance,
        )


        correct = comparison["match"]
        score = self.config.correct_score if correct else self.config.incorrect_score

        # print(f"[math rewards] {extraction=} {pred_normalized=} {gt_normalized=} {comparison=} {score=}")

        return VerificationResult(
            correct=correct,
            score=score,
            extracted_answer=extraction.answer,
            extraction_method=extraction.method,
            comparison_method=comparison["method"],
            ground_truth=ground_truth,
            pred_normalized=pred_normalized,
            gt_normalized=gt_normalized,
            metadata=comparison.get("metadata", {}),
        )

    def compute_score(
        self,
        solution: str,
        ground_truth: str,
    ) -> dict:
        """
        Compute reward score for RLVR.

        Convenience method that returns a dict compatible with typical
        RL training loops.

        Args:
            solution: The model's solution string
            ground_truth: The expected answer

        Returns:
            Dict with 'score', 'acc', 'pred', 'gts' keys
        """
        result = self.verify(solution, ground_truth)
        return {
            "score": result.score,
            "acc": int(result.correct),
            "pred": result.extracted_answer,
            "gts": ground_truth,
            "extraction_method": result.extraction_method.name,
            "comparison_method": result.comparison_method.name,
        }


# Convenience function for simple usage
def compute_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> dict:
    """
    Compute reward score for a solution.

    This is a convenience function that creates a default verifier.
    For repeated use, create a MathVerifier instance instead.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        **kwargs: Additional arguments passed to VerifierConfig

    Returns:
        Dict with score and diagnostic information
    """
    config = VerifierConfig(**kwargs) if kwargs else None
    verifier = MathVerifier(config)
    return verifier.compute_score(solution_str, ground_truth)
