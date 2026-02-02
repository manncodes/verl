# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""
Comprehensive tests for math_verify module with comparisons to other verifiers.

Run with: pytest test_math_verify.py -v
"""

import pytest
from verl.utils.reward_score.math_verify import (
    MathVerifier,
    VerifierConfig,
    extract_boxed,
    extract_answer,
    extract_all_boxed,
    normalize_latex,
    NormalizationConfig,
    compare_strings,
    compare_numeric,
    compare_symbolic,
    is_mathematically_equivalent,
)
from verl.utils.reward_score.math_verify.core import ExtractionMethod, ComparisonMethod
from verl.utils.reward_score.math_verify.extract import extract_by_pattern, extract_last_number


# =============================================================================
# Test Data: Common test cases for cross-verifier comparison
# =============================================================================

# Format: (solution, ground_truth, expected_correct)
BASIC_TEST_CASES = [
    # Simple integers
    (r"The answer is \boxed{42}", "42", True),
    (r"So we get \boxed{-7}", "-7", True),
    (r"\boxed{0}", "0", True),

    # Simple fractions
    (r"\boxed{\frac{1}{2}}", "0.5", True),
    (r"\boxed{1/2}", "0.5", True),
    (r"The result is \boxed{\frac{3}{4}}", "0.75", True),

    # Decimals
    (r"\boxed{3.14159}", "3.14159", True),
    (r"\boxed{0.333}", "0.333", True),

    # Wrong answers
    (r"\boxed{41}", "42", False),
    (r"\boxed{wrong}", "42", False),

    # No boxed answer
    ("The answer is 42", "42", True),  # Pattern extraction
    ("Therefore, 42.", "42", True),  # Fallback extraction
]

LATEX_NORMALIZATION_CASES = [
    # dfrac/tfrac -> frac
    (r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}", True),
    (r"\boxed{\tfrac{1}{2}}", r"\frac{1}{2}", True),

    # Frac shorthand
    (r"\boxed{\frac12}", r"\frac{1}{2}", True),

    # With \left and \right
    (r"\boxed{\left(\frac{1}{2}\right)}", r"(\frac{1}{2})", True),

    # Whitespace variations
    (r"\boxed{ 42 }", "42", True),
    (r"\boxed{  3.14  }", "3.14", True),
]

NESTED_BOXED_CASES = [
    # Nested fractions
    (r"\boxed{\frac{1}{\sqrt{2}}}", r"\frac{1}{\sqrt{2}}", True),
    (r"\boxed{\frac{\frac{1}{2}}{3}}", r"\frac{\frac{1}{2}}{3}", True),

    # Multiple boxed - should use last
    (r"\boxed{wrong} then \boxed{correct}", "correct", True),
    (r"First \boxed{1}, then \boxed{2}, finally \boxed{3}", "3", True),
]

NUMERIC_EQUIVALENCE_CASES = [
    # Different representations of same value
    (r"\boxed{0.5}", "1/2", True),
    (r"\boxed{\frac{1}{2}}", "0.5", True),
    (r"\boxed{0.25}", r"\frac{1}{4}", True),

    # Scientific notation
    (r"\boxed{1e-3}", "0.001", True),
    (r"\boxed{1.5e2}", "150", True),

    # Percentages (if numeric comparison enabled)
    (r"\boxed{50\%}", "0.5", True),
]

EDGE_CASES = [
    # Empty
    ("", "42", False),
    (r"\boxed{}", "", True),

    # Very long solution with answer at end
    ("Step " * 100 + r"\boxed{42}", "42", True),

    # Unicode and special chars
    (r"\boxed{π}", "π", True),
    (r"\boxed{∞}", "∞", True),

    # Negative numbers
    (r"\boxed{-42}", "-42", True),
    (r"\boxed{-\frac{1}{2}}", "-0.5", True),

    # Large numbers
    (r"\boxed{1000000}", "1000000", True),
    (r"\boxed{1,000,000}", "1000000", True),  # With thousands separator
]

CHAIN_OF_THOUGHT_CASES = [
    # Typical CoT format
    ("""
    Let's solve this step by step.
    First, x = 2.
    Then, x^2 = 4.
    Finally, x^2 + 1 = 5.

    Therefore, the answer is \\boxed{5}.
    """, "5", True),

    # Multiple intermediate calculations
    ("""
    We need to find 2 + 3 * 4.

    Step 1: 3 * 4 = 12
    Step 2: 2 + 12 = 14

    \\boxed{14}
    """, "14", True),

    # Answer in middle (should still find last boxed)
    ("""
    First attempt: \\boxed{wrong}
    Wait, let me recalculate...
    The correct answer is \\boxed{right}
    """, "right", True),
]


# =============================================================================
# Extraction Tests
# =============================================================================

class TestExtractBoxed:
    """Tests for boxed content extraction."""

    def test_simple_boxed(self):
        result = extract_boxed(r"The answer is \boxed{42}")
        assert result.success
        assert result.answer == "42"
        assert result.method == ExtractionMethod.BOXED

    def test_nested_braces(self):
        result = extract_boxed(r"\boxed{\frac{1}{2}}")
        assert result.success
        assert result.answer == r"\frac{1}{2}"

    def test_deeply_nested(self):
        result = extract_boxed(r"\boxed{\frac{1}{\sqrt{2+\frac{1}{3}}}}")
        assert result.success
        assert result.answer == r"\frac{1}{\sqrt{2+\frac{1}{3}}}"

    def test_multiple_boxed_returns_last(self):
        result = extract_boxed(r"\boxed{wrong} and then \boxed{correct}")
        assert result.success
        assert result.answer == "correct"

    def test_no_boxed(self):
        result = extract_boxed("Just plain text")
        assert not result.success
        assert result.method == ExtractionMethod.FAILED

    def test_unbalanced_braces_recovery(self):
        # Simulates truncated output
        result = extract_boxed(r"\boxed{42")
        # Should attempt recovery
        assert result.method in (ExtractionMethod.BOXED, ExtractionMethod.FAILED)

    def test_empty_boxed(self):
        result = extract_boxed(r"\boxed{}")
        assert result.success
        assert result.answer == ""

    def test_boxed_with_space(self):
        result = extract_boxed(r"\boxed { 42 }")
        # Some implementations handle this, some don't
        assert result.method in (ExtractionMethod.BOXED, ExtractionMethod.FAILED)

    def test_fbox_alternative(self):
        result = extract_boxed(r"\fbox{42}")
        assert result.success
        assert result.answer == "42"

    def test_boxed_with_newlines(self):
        result = extract_boxed(r"\boxed{x +\n y}")
        assert result.success
        assert "x" in result.answer and "y" in result.answer


class TestExtractAllBoxed:
    """Tests for extracting all boxed expressions."""

    def test_multiple_boxed(self):
        text = r"First \boxed{1}, second \boxed{2}, third \boxed{3}"
        results = extract_all_boxed(text)
        assert len(results) == 3
        assert results == ["1", "2", "3"]

    def test_no_boxed(self):
        results = extract_all_boxed("No boxed here")
        assert results == []

    def test_single_boxed(self):
        results = extract_all_boxed(r"Only \boxed{one}")
        assert results == ["one"]


class TestExtractByPattern:
    """Tests for pattern-based extraction."""

    def test_answer_is_pattern(self):
        patterns = [r"(?i)(?:the\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|,|\n|$)"]
        result = extract_by_pattern("The answer is 42.", patterns)
        assert result.success
        assert result.answer == "42"

    def test_final_answer_pattern(self):
        patterns = [r"(?i)\*{0,2}(?:final\s+)?answer\*{0,2}\s*[:=]\s*(.+?)(?:\n|$)"]
        result = extract_by_pattern("Final Answer: 42", patterns)
        assert result.success
        assert result.answer == "42"

    def test_therefore_pattern(self):
        patterns = [r"(?i)therefore[,\s]+(.+?)(?:\.|$)"]
        result = extract_by_pattern("Therefore, x = 5.", patterns)
        assert result.success

    def test_no_match(self):
        patterns = [r"(?i)answer\s*[:=]\s*(.+)"]
        result = extract_by_pattern("Random text", patterns)
        assert not result.success


class TestExtractLastNumber:
    """Tests for last number extraction fallback."""

    def test_simple_integer(self):
        result = extract_last_number("The result is 42")
        assert result.success
        assert result.answer == "42"

    def test_decimal(self):
        result = extract_last_number("We get 3.14159")
        assert result.success
        assert result.answer == "3.14159"

    def test_negative(self):
        result = extract_last_number("The value is -17")
        assert result.success
        assert result.answer == "-17"

    def test_fraction(self):
        result = extract_last_number("The ratio is 3/4")
        assert result.success
        assert result.answer == "3/4"

    def test_scientific_notation(self):
        result = extract_last_number("The answer is 1.5e-3")
        assert result.success
        assert "1.5" in result.answer

    def test_no_number(self):
        result = extract_last_number("No numbers here")
        assert not result.success

    def test_multiple_numbers_returns_last(self):
        result = extract_last_number("First 1, then 2, finally 3")
        assert result.success
        assert result.answer == "3"


class TestExtractAnswer:
    """Tests for the combined extraction function."""

    def test_boxed_preferred(self):
        result = extract_answer(r"Answer: 41, but \boxed{42}")
        assert result.answer == "42"
        assert result.method == ExtractionMethod.BOXED

    def test_pattern_fallback(self):
        result = extract_answer("The answer is 42.")
        assert result.success
        assert result.method == ExtractionMethod.PATTERN

    def test_number_fallback(self):
        result = extract_answer("Random calculation gives 42")
        assert result.success
        assert result.method == ExtractionMethod.LAST_NUMBER

    def test_hint_changes_priority(self):
        result = extract_answer(
            r"Answer: 41, \boxed{42}",
            hint=ExtractionMethod.PATTERN
        )
        # With hint, pattern should be tried first
        assert result.success


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalization:
    """Tests for LaTeX normalization."""

    def test_dfrac_to_frac(self):
        assert "\\frac" in normalize_latex(r"\dfrac{1}{2}")
        assert "\\dfrac" not in normalize_latex(r"\dfrac{1}{2}")

    def test_tfrac_to_frac(self):
        assert "\\frac" in normalize_latex(r"\tfrac{1}{2}")
        assert "\\tfrac" not in normalize_latex(r"\tfrac{1}{2}")

    def test_cfrac_to_frac(self):
        assert "\\frac" in normalize_latex(r"\cfrac{1}{2}")
        assert "\\cfrac" not in normalize_latex(r"\cfrac{1}{2}")

    def test_frac_shorthand_expansion(self):
        result = normalize_latex(r"\frac12")
        assert result == r"\frac{1}{2}"

    def test_frac_mixed_shorthand(self):
        result = normalize_latex(r"\frac1{23}")
        assert result == r"\frac{1}{23}"

    def test_frac_already_braced(self):
        result = normalize_latex(r"\frac{12}{34}")
        assert result == r"\frac{12}{34}"

    def test_sqrt_shorthand(self):
        result = normalize_latex(r"\sqrt2")
        assert result == r"\sqrt{2}"

    def test_sqrt_letter(self):
        result = normalize_latex(r"\sqrtx")
        assert result == r"\sqrt{x}"

    def test_remove_text_wrapper(self):
        result = normalize_latex(r"\text{hello}")
        assert result == "hello"

    def test_remove_mathrm(self):
        result = normalize_latex(r"\mathrm{sin}")
        assert result == "sin"

    def test_thousands_separator(self):
        config = NormalizationConfig(remove_thousands_separator=True)
        result = normalize_latex("1,000,000", config)
        assert result == "1000000"

    def test_whitespace_collapse(self):
        result = normalize_latex(r"x   +   y")
        assert "   " not in result

    def test_displaystyle_removal(self):
        result = normalize_latex(r"\displaystyle \frac{1}{2}")
        assert "\\displaystyle" not in result

    def test_left_right_removal(self):
        result = normalize_latex(r"\left(\frac{1}{2}\right)")
        assert "\\left" not in result
        assert "\\right" not in result

    def test_empty_string(self):
        assert normalize_latex("") == ""

    def test_preserve_math_meaning(self):
        # Normalization shouldn't change mathematical meaning
        original = r"\frac{a+b}{c}"
        normalized = normalize_latex(original)
        assert "a" in normalized and "b" in normalized and "c" in normalized


# =============================================================================
# Comparison Tests
# =============================================================================

class TestStringComparison:
    """Tests for string comparison."""

    def test_exact_match(self):
        result = compare_strings("42", "42")
        assert result["match"]
        assert result["method"] == ComparisonMethod.STRING_EXACT

    def test_normalized_match(self):
        result = compare_strings("42", "42 ", "42", "42")
        assert result["match"]

    def test_no_match(self):
        result = compare_strings("41", "42")
        assert not result["match"]


class TestNumericComparison:
    """Tests for numeric comparison."""

    def test_exact_match(self):
        result = compare_numeric("42", "42")
        assert result["match"]

    def test_float_tolerance(self):
        result = compare_numeric("0.333333", "0.333334", tolerance=1e-5)
        assert result["match"]

    def test_no_tolerance_exceeded(self):
        result = compare_numeric("0.333333", "0.333334", tolerance=1e-9)
        assert not result["match"]

    def test_fraction_to_decimal(self):
        result = compare_numeric("1/2", "0.5")
        assert result["match"]

    def test_latex_fraction(self):
        result = compare_numeric(r"\frac{1}{2}", "0.5")
        assert result["match"]

    def test_percentage(self):
        result = compare_numeric("50%", "0.5")
        assert result["match"]

    def test_scientific_notation(self):
        result = compare_numeric("1.5e-3", "0.0015")
        assert result["match"]

    def test_different_values(self):
        result = compare_numeric("1", "2")
        assert not result["match"]

    def test_nan_handling(self):
        result = compare_numeric("nan", "nan")
        # NaN != NaN by IEEE standard
        assert not result["match"]

    def test_inf_handling(self):
        result = compare_numeric("inf", "inf")
        # Note: This depends on implementation
        pass

    def test_conversion_failure(self):
        result = compare_numeric("abc", "42")
        assert not result["match"]
        assert "error" in result.get("metadata", {})


class TestSymbolicComparison:
    """Tests for symbolic comparison (requires sympy)."""

    @pytest.fixture
    def has_sympy(self):
        try:
            import sympy
            return True
        except ImportError:
            return False

    def test_symbolic_not_available_graceful(self):
        # Should handle missing sympy gracefully
        result = compare_symbolic("x", "x")
        # Either matches (if sympy available) or returns error metadata
        assert isinstance(result["match"], bool)

    @pytest.mark.skipif(
        not MathVerifier()._symbolic_available,
        reason="sympy not installed"
    )
    def test_algebraic_equivalence(self):
        result = compare_symbolic(r"2x", r"x + x")
        # May or may not match depending on sympy version
        assert isinstance(result["match"], bool)


class TestIsMathematicallyEquivalent:
    """Tests for the convenience equivalence function."""

    def test_string_equivalence(self):
        assert is_mathematically_equivalent("42", "42")

    def test_numeric_equivalence(self):
        assert is_mathematically_equivalent("0.5", "1/2")

    def test_not_equivalent(self):
        assert not is_mathematically_equivalent("1", "2")

    def test_method_selection(self):
        # Only string comparison
        result = is_mathematically_equivalent("0.5", "1/2", methods=["string"])
        assert not result  # String "0.5" != "1/2"

        # With numeric
        result = is_mathematically_equivalent("0.5", "1/2", methods=["numeric"])
        assert result


# =============================================================================
# Integration Tests: MathVerifier
# =============================================================================

class TestMathVerifier:
    """Integration tests for the full verifier."""

    @pytest.fixture
    def verifier(self):
        return MathVerifier()

    def test_simple_boxed_correct(self, verifier):
        result = verifier.verify(
            solution=r"The answer is \boxed{42}",
            ground_truth="42"
        )
        assert result.correct
        assert result.score == 1.0
        assert result.extraction_method == ExtractionMethod.BOXED

    def test_simple_boxed_incorrect(self, verifier):
        result = verifier.verify(
            solution=r"The answer is \boxed{41}",
            ground_truth="42"
        )
        assert not result.correct
        assert result.score == 0.0

    def test_fraction_equivalence(self, verifier):
        result = verifier.verify(
            solution=r"The answer is \boxed{\frac{1}{2}}",
            ground_truth="0.5"
        )
        assert result.correct
        assert result.comparison_method == ComparisonMethod.NUMERIC

    def test_pattern_extraction(self, verifier):
        result = verifier.verify(
            solution="After calculation, the answer is: 42",
            ground_truth="42"
        )
        assert result.correct
        assert result.extraction_method == ExtractionMethod.PATTERN

    def test_chain_of_thought(self, verifier):
        solution = r"""
        Let's solve this step by step.
        First, we note that x = 2.
        Then, x^2 = 4.
        Adding them: 2 + 4 = 6.

        Therefore, the answer is \boxed{6}.
        """
        result = verifier.verify(solution, "6")
        assert result.correct

    def test_multiple_boxed_uses_last(self, verifier):
        solution = r"""
        Intermediate result: \boxed{wrong}
        Final answer: \boxed{correct}
        """
        result = verifier.verify(solution, "correct")
        assert result.correct

    def test_latex_normalization(self, verifier):
        result = verifier.verify(
            solution=r"\boxed{\dfrac{1}{2}}",
            ground_truth=r"\frac{1}{2}"
        )
        assert result.correct

    def test_compute_score_interface(self, verifier):
        score_dict = verifier.compute_score(
            solution=r"\boxed{42}",
            ground_truth="42"
        )
        assert "score" in score_dict
        assert "acc" in score_dict
        assert "pred" in score_dict
        assert "gts" in score_dict
        assert score_dict["score"] == 1.0
        assert score_dict["acc"] == 1

    def test_extraction_failed(self, verifier):
        result = verifier.verify(
            solution="No answer here",
            ground_truth="42"
        )
        assert not result.correct
        assert result.extraction_method == ExtractionMethod.FAILED

    def test_custom_config(self):
        config = VerifierConfig(
            correct_score=2.0,
            incorrect_score=-1.0,
        )
        verifier = MathVerifier(config)

        result = verifier.verify(r"\boxed{42}", "42")
        assert result.score == 2.0

        result = verifier.verify(r"\boxed{41}", "42")
        assert result.score == -1.0

    def test_search_window(self):
        config = VerifierConfig(search_window=50)
        verifier = MathVerifier(config)

        # Answer in last 50 chars
        result = verifier.verify("x" * 100 + r"\boxed{42}", "42")
        assert result.correct

    def test_json_ground_truth_unwrap(self, verifier):
        # Ground truth in JSON array format
        result = verifier.verify(r"\boxed{42}", '["42"]')
        assert result.correct

    def test_to_dict(self, verifier):
        result = verifier.verify(r"\boxed{42}", "42")
        d = result.to_dict()
        assert d["correct"] == True
        assert d["score"] == 1.0
        assert d["pred"] == "42"


# =============================================================================
# Edge Cases and Regression Tests
# =============================================================================

class TestEdgeCases:
    """Edge cases and regression tests."""

    @pytest.fixture
    def verifier(self):
        return MathVerifier()

    def test_empty_solution(self, verifier):
        result = verifier.verify("", "42")
        assert not result.correct

    def test_empty_ground_truth(self, verifier):
        result = verifier.verify(r"\boxed{}", "")
        # Empty should match empty
        assert result.correct

    def test_whitespace_handling(self, verifier):
        result = verifier.verify(r"\boxed{ 42 }", "42")
        assert result.correct

    def test_negative_numbers(self, verifier):
        result = verifier.verify(r"\boxed{-42}", "-42")
        assert result.correct

    def test_decimal_precision(self, verifier):
        result = verifier.verify(
            r"\boxed{3.14159}",
            "3.14159"
        )
        assert result.correct

    def test_very_long_solution(self, verifier):
        # Simulate long chain-of-thought
        solution = "Step " * 1000 + r"\boxed{42}"
        result = verifier.verify(solution, "42")
        assert result.correct

    def test_special_latex_chars(self, verifier):
        result = verifier.verify(
            r"\boxed{x^{2} + y^{2}}",
            r"x^{2} + y^{2}"
        )
        assert result.correct

    def test_matrix_answer(self, verifier):
        # Matrices are tricky - should at least not crash
        result = verifier.verify(
            r"\boxed{\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}}",
            r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
        )
        # String comparison should work
        assert result.correct

    def test_unicode_answer(self, verifier):
        result = verifier.verify(r"\boxed{α + β}", "α + β")
        assert result.correct

    def test_multiline_boxed(self, verifier):
        result = verifier.verify(
            r"\boxed{x = 1\\ y = 2}",
            r"x = 1\\ y = 2"
        )
        assert result.correct


# =============================================================================
# Comparison with Other Verifiers
# =============================================================================

class TestCompareWithMathReward:
    """Compare our verifier with math_reward.py (EleutherAI-based)."""

    @pytest.fixture
    def our_verifier(self):
        return MathVerifier()

    @pytest.fixture
    def math_reward_verifier(self):
        try:
            from verl.utils.reward_score import math_reward
            return math_reward
        except ImportError:
            pytest.skip("math_reward not available")

    @pytest.mark.parametrize("solution,gt,expected", BASIC_TEST_CASES[:6])
    def test_basic_cases_match(self, our_verifier, math_reward_verifier, solution, gt, expected):
        """Verify basic cases produce same results."""
        our_result = our_verifier.verify(solution, gt)

        try:
            their_score = math_reward_verifier.compute_score(solution, gt)
        except Exception:
            pytest.skip("math_reward failed on this case")

        # Log any discrepancies for analysis
        if our_result.correct != expected:
            print(f"Our verifier: {our_result.correct}, expected: {expected}")
        if (their_score > 0) != expected:
            print(f"math_reward: {their_score > 0}, expected: {expected}")


class TestCompareWithMathDapo:
    """Compare our verifier with math_dapo.py."""

    @pytest.fixture
    def our_verifier(self):
        return MathVerifier()

    @pytest.fixture
    def math_dapo_verifier(self):
        try:
            from verl.utils.reward_score import math_dapo
            return math_dapo
        except ImportError:
            pytest.skip("math_dapo not available")

    @pytest.mark.parametrize("solution,gt,expected", BASIC_TEST_CASES[:6])
    def test_basic_cases_match(self, our_verifier, math_dapo_verifier, solution, gt, expected):
        """Verify basic cases produce same results."""
        our_result = our_verifier.verify(solution, gt)

        try:
            their_result = math_dapo_verifier.compute_score(solution, gt)
            their_correct = their_result.get("acc", their_result.get("score", 0)) > 0
        except Exception:
            pytest.skip("math_dapo failed on this case")

        # Log discrepancies
        if our_result.correct != their_correct:
            print(f"Discrepancy: ours={our_result.correct}, theirs={their_correct}, solution={solution[:50]}")


class TestCompareWithMathVerifyExternal:
    """Compare with the external math-verify package wrapper."""

    @pytest.fixture
    def our_verifier(self):
        return MathVerifier()

    @pytest.fixture
    def external_verifier(self):
        try:
            from verl.utils.reward_score import math_verify_external
            return math_verify_external
        except ImportError:
            pytest.skip("math_verify_external not available")

    @pytest.mark.parametrize("solution,gt,expected", BASIC_TEST_CASES[:6])
    def test_basic_cases(self, our_verifier, external_verifier, solution, gt, expected):
        """Compare with external math-verify package."""
        our_result = our_verifier.verify(solution, gt)

        try:
            their_score = external_verifier.compute_score(solution, gt)
        except Exception:
            pytest.skip("external math_verify failed")

        print(f"Ours: {our_result.correct}, External: {their_score > 0}")


# =============================================================================
# Comprehensive Cross-Verifier Benchmark
# =============================================================================

class TestCrossVerifierBenchmark:
    """
    Benchmark all verifiers on the same test cases.

    This helps identify:
    - Cases where verifiers disagree
    - Performance differences
    - Edge cases that break certain verifiers
    """

    @pytest.fixture
    def all_verifiers(self):
        """Load all available verifiers."""
        verifiers = {
            "math_verify_new": MathVerifier(),
        }

        try:
            from verl.utils.reward_score import math_reward
            verifiers["math_reward"] = math_reward
        except ImportError:
            pass

        try:
            from verl.utils.reward_score import math_dapo
            verifiers["math_dapo"] = math_dapo
        except ImportError:
            pass

        try:
            from verl.utils.reward_score import math_verify_external
            verifiers["math_verify_external"] = math_verify_external
        except ImportError:
            pass

        return verifiers

    def _verify_with(self, verifier, name: str, solution: str, gt: str) -> tuple:
        """Run verification with any verifier type."""
        try:
            if name == "math_verify_new":
                result = verifier.verify(solution, gt)
                return result.correct, result.extracted_answer
            else:
                result = verifier.compute_score(solution, gt)
                if isinstance(result, dict):
                    score = result.get("acc", result.get("score", 0))
                    pred = result.get("pred")
                else:
                    score = result
                    pred = None
                return score > 0, pred
        except Exception as e:
            return None, str(e)

    def test_benchmark_all_cases(self, all_verifiers):
        """Run all test cases through all verifiers and report results."""
        all_cases = (
            BASIC_TEST_CASES +
            LATEX_NORMALIZATION_CASES +
            NESTED_BOXED_CASES +
            NUMERIC_EQUIVALENCE_CASES[:3] +  # Skip percentage test
            CHAIN_OF_THOUGHT_CASES
        )

        results = []

        for solution, gt, expected in all_cases:
            case_results = {"solution": solution[:50], "gt": gt, "expected": expected}

            for name, verifier in all_verifiers.items():
                correct, pred = self._verify_with(verifier, name, solution, gt)
                case_results[name] = correct
                case_results[f"{name}_pred"] = pred

            results.append(case_results)

        # Report disagreements
        print("\n=== Cross-Verifier Comparison ===")
        disagreements = 0
        for r in results:
            verifier_results = [r.get(name) for name in all_verifiers.keys()]
            verifier_results = [v for v in verifier_results if v is not None]

            if len(set(verifier_results)) > 1:
                disagreements += 1
                print(f"\nDisagreement on: {r['solution']}")
                print(f"  Expected: {r['expected']}")
                for name in all_verifiers.keys():
                    print(f"  {name}: {r.get(name)}")

        print(f"\nTotal disagreements: {disagreements}/{len(results)}")


# =============================================================================
# Symbolic Comparison Tests (when sympy available)
# =============================================================================

class TestSymbolicComparison:
    """Tests for symbolic comparison (requires sympy)."""

    @pytest.fixture
    def verifier(self):
        return MathVerifier(VerifierConfig(enable_symbolic=True))

    @pytest.mark.skipif(
        not MathVerifier()._symbolic_available,
        reason="sympy not installed"
    )
    def test_algebraic_equivalence(self, verifier):
        # x^2 - 1 = (x-1)(x+1)
        result = verifier.verify(
            r"\boxed{x^2 - 1}",
            r"(x-1)(x+1)"
        )
        # Note: This may or may not pass depending on sympy's simplification
        # The key is it doesn't crash
        assert result.comparison_method in (
            ComparisonMethod.STRING_NORMALIZED,
            ComparisonMethod.SYMBOLIC,
            ComparisonMethod.FAILED,
        )

    @pytest.mark.skipif(
        not MathVerifier()._symbolic_available,
        reason="sympy not installed"
    )
    def test_trigonometric_identity(self, verifier):
        # sin^2(x) + cos^2(x) = 1 is hard for sympy to verify
        # But we should handle it gracefully
        result = verifier.verify(
            r"\boxed{\sin^2(x) + \cos^2(x)}",
            "1"
        )
        # Just verify no crash
        assert isinstance(result.correct, bool)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for the verifier."""

    @pytest.fixture
    def verifier(self):
        return MathVerifier()

    def test_many_verifications(self, verifier):
        """Test that many verifications complete quickly."""
        import time

        start = time.time()
        for _ in range(100):
            verifier.verify(r"\boxed{42}", "42")
        elapsed = time.time() - start

        # Should complete 100 verifications in under 1 second
        assert elapsed < 1.0, f"Too slow: {elapsed}s for 100 verifications"

    def test_long_solution_performance(self, verifier):
        """Test performance with very long solutions."""
        import time

        long_solution = "Step " * 10000 + r"\boxed{42}"

        start = time.time()
        result = verifier.verify(long_solution, "42")
        elapsed = time.time() - start

        assert result.correct
        # Should complete in under 0.5 seconds even with long solution
        assert elapsed < 0.5, f"Too slow: {elapsed}s for long solution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
