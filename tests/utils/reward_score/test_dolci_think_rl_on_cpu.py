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
Tests for the Dolci-Think-RL reward function.

Tests all dataset source types:
- math: Mathematical reasoning problems (uses math_verify.MathVerifier)
- instruction_following: Instruction following with constraints (uses ifeval)
- code/code_stdio: Code execution problems (uses LLM-as-a-judge)
- general-quality: General chat/QA responses

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import os
import sys

import pytest

# Add the verl directory to path for direct import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Try to import the module directly without going through verl/__init__.py
try:
    # Direct import to avoid verl/__init__.py dependencies
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dolci_think_rl",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "verl",
            "utils",
            "reward_score",
            "dolci_think_rl.py",
        ),
    )
    dolci_think_rl_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dolci_think_rl_module)
    DOLCI_AVAILABLE = True
except Exception as e:
    DOLCI_AVAILABLE = False
    DOLCI_IMPORT_ERROR = str(e)

# Check if verl is fully installed (for integration tests)
try:
    from verl.utils.reward_score import default_compute_score

    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False

# Check if math_verify is available
try:
    from verl.utils.reward_score.math_verify import MathVerifier

    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

# Check if ifeval is available
try:
    from verl.utils.reward_score import ifeval

    IFEVAL_AVAILABLE = True
except ImportError:
    IFEVAL_AVAILABLE = False


# =============================================================================
# Test Data: Math Problems
# =============================================================================

MATH_TEST_CASES = [
    # Case 1: Simple boxed answer
    {
        "solution": "<think>Let me solve this step by step.\n2 + 2 = 4</think>\nThe answer is \\boxed{4}",
        "ground_truth": "4",
        "expected_score": 1.0,
        "description": "Simple addition with boxed answer",
    },
    # Case 2: Fraction answer
    {
        "solution": "<think>Computing the fraction...</think>\nThe result is \\boxed{\\frac{1}{2}}",
        "ground_truth": "\\frac{1}{2}",
        "expected_score": 1.0,
        "description": "Fraction answer",
    },
    # Case 3: Wrong answer
    {
        "solution": "<think>Let me calculate...</think>\nThe answer is \\boxed{5}",
        "ground_truth": "4",
        "expected_score": 0.0,
        "description": "Wrong math answer",
    },
    # Case 4: Answer with thinking tags that need removal
    {
        "solution": "<|assistant|><think>Solving quadratic...</think><evaluation>Checking...</evaluation>The answer is \\boxed{42}",
        "ground_truth": "42",
        "expected_score": 1.0,
        "description": "Answer with multiple tags to strip",
    },
]


# =============================================================================
# Test Data: General Quality (Chat/QA)
# =============================================================================

GENERAL_TEST_CASES = [
    # Case 1: Exact match
    {
        "solution": "The capital of France is Paris.",
        "ground_truth": "Paris",
        "expected_score": 1.0,
        "description": "Exact string match",
    },
    # Case 2: Contains answer
    {
        "solution": "<think>Let me think about this...</think>The answer is definitely 42.",
        "ground_truth": "42",
        "expected_score": 1.0,
        "description": "Contains ground truth",
    },
    # Case 3: No match
    {
        "solution": "I don't know the answer.",
        "ground_truth": "specific answer",
        "expected_score": 0.0,
        "description": "No match",
    },
    # Case 4: Case insensitive match
    {
        "solution": "The answer is YES.",
        "ground_truth": "yes",
        "expected_score": 1.0,
        "description": "Case insensitive",
    },
]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def dolci_module():
    """Import dolci_think_rl module."""
    if not DOLCI_AVAILABLE:
        pytest.skip(f"dolci_think_rl module not available: {DOLCI_IMPORT_ERROR}")
    return dolci_think_rl_module


# Skip markers
requires_verl = pytest.mark.skipif(not VERL_AVAILABLE, reason="verl package not fully installed")
requires_dolci = pytest.mark.skipif(not DOLCI_AVAILABLE, reason="dolci_think_rl module not available")
requires_math_verify = pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify module not available")
requires_ifeval = pytest.mark.skipif(not IFEVAL_AVAILABLE, reason="ifeval module not available")


# =============================================================================
# Unit Tests: Helper Functions
# =============================================================================


class TestRemoveThinkingSection:
    """Tests for remove_thinking_section function."""

    def test_removes_think_tags(self, dolci_module):
        text = "<think>Some reasoning</think>The answer is 42."
        result = dolci_module.remove_thinking_section(text)
        assert result == "The answer is 42."

    def test_removes_evaluation_tags(self, dolci_module):
        text = "Prefix<evaluation>Some eval</evaluation>The answer is 42."
        result = dolci_module.remove_thinking_section(text)
        assert result == "The answer is 42."

    def test_removes_answer_tags(self, dolci_module):
        text = "<answer>42</answer>"
        result = dolci_module.remove_thinking_section(text)
        assert result == "42"

    def test_removes_assistant_marker(self, dolci_module):
        text = "<|assistant|>Hello world"
        result = dolci_module.remove_thinking_section(text)
        assert result == "Hello world"

    def test_handles_none(self, dolci_module):
        result = dolci_module.remove_thinking_section(None)
        assert result == ""

    def test_combined_tags(self, dolci_module):
        text = "<|assistant|><think>Thinking...</think><evaluation>Eval...</evaluation><answer>42</answer>"
        result = dolci_module.remove_thinking_section(text)
        assert result == "42"


class TestBasicStringMatch:
    """Tests for _basic_string_match function."""

    def test_exact_match(self, dolci_module):
        assert dolci_module._basic_string_match("hello", "hello") == 1.0

    def test_case_insensitive(self, dolci_module):
        assert dolci_module._basic_string_match("HELLO", "hello") == 1.0

    def test_contains_match(self, dolci_module):
        assert dolci_module._basic_string_match("The answer is hello world", "hello") == 1.0

    def test_no_match(self, dolci_module):
        assert dolci_module._basic_string_match("goodbye", "hello") == 0.0

    def test_whitespace_handling(self, dolci_module):
        assert dolci_module._basic_string_match("  hello  ", "hello") == 1.0


# =============================================================================
# Integration Tests: Math Scoring (using math_verify.MathVerifier)
# =============================================================================


@requires_math_verify
class TestMathScoring:
    """Tests for math scoring functionality using math_verify.MathVerifier."""

    @pytest.mark.parametrize("test_case", MATH_TEST_CASES, ids=lambda tc: tc["description"])
    def test_math_cases(self, dolci_module, test_case):
        extra_info = {"dataset_source": "math"}
        score = dolci_module.compute_score(
            solution_str=test_case["solution"],
            ground_truth=test_case["ground_truth"],
            extra_info=extra_info,
        )
        # Allow some tolerance for different verifiers
        assert score == pytest.approx(test_case["expected_score"], abs=0.1)


# =============================================================================
# Integration Tests: IFEval Scoring (using ifeval module)
# =============================================================================


@requires_ifeval
class TestIFEvalScoring:
    """Tests for IFEval scoring functionality using ifeval module."""

    def test_ifeval_basic(self, dolci_module):
        """Test basic ifeval scoring."""
        extra_info = {"dataset_source": "instruction_following"}
        # The actual test depends on what ifeval.compute_score expects
        # This is a basic smoke test
        score = dolci_module.compute_score(
            solution_str="I love Python and machine learning.",
            ground_truth="Python",  # ifeval may use this differently
            extra_info=extra_info,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# =============================================================================
# Integration Tests: General Quality Scoring
# =============================================================================


class TestGeneralScoring:
    """Tests for general quality scoring functionality."""

    @pytest.mark.parametrize("test_case", GENERAL_TEST_CASES, ids=lambda tc: tc["description"])
    def test_general_cases(self, dolci_module, test_case):
        extra_info = {"dataset_source": "general-quality"}
        score = dolci_module.compute_score(
            solution_str=test_case["solution"],
            ground_truth=test_case["ground_truth"],
            extra_info=extra_info,
        )
        assert score == test_case["expected_score"]


# =============================================================================
# Integration Tests: Code Scoring (requires LLM judge)
# =============================================================================


class TestCodeScoring:
    """Tests for code scoring functionality (requires LLM judge service)."""

    def test_code_fallback_to_string_match(self, dolci_module):
        """Test that code scoring falls back to string matching when judge unavailable."""
        extra_info = {"dataset_source": "code", "problem": "Write a function to add two numbers"}
        # Without judge service, should fall back to string matching
        score = dolci_module.compute_score(
            solution_str="def add(a, b): return a + b",
            ground_truth="add",  # basic match
            extra_info=extra_info,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# =============================================================================
# Integration Tests: Batch Processing
# =============================================================================


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    @requires_math_verify
    def test_mixed_batch(self, dolci_module):
        """Test batch processing with mixed data types."""
        solutions = [
            # Math
            "<think>Solving...</think>\\boxed{4}",
            # General
            "The answer is Paris.",
        ]
        ground_truths = [
            "4",
            "Paris",
        ]
        extra_infos = [
            {"dataset_source": "math"},
            {"dataset_source": "general-quality"},
        ]

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )

        assert len(scores) == 2
        assert scores[0] == pytest.approx(1.0, abs=0.1)  # Math correct
        assert scores[1] == 1.0  # General contains answer

    def test_empty_batch(self, dolci_module):
        """Test batch processing with empty input."""
        scores = dolci_module.compute_score_batch(
            solution_strs=[],
            ground_truths=[],
            extra_infos=[],
        )
        assert scores == []

    @requires_math_verify
    def test_batch_with_none_values(self, dolci_module):
        """Test batch processing with None values."""
        solutions = [None, "\\boxed{4}", ""]
        ground_truths = ["answer", "4", None]
        extra_infos = [
            {"dataset_source": "math"},
            {"dataset_source": "math"},
            {"dataset_source": "general-quality"},
        ]

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )

        assert len(scores) == 3
        assert scores[0] == 0.0  # None solution
        assert scores[1] == pytest.approx(1.0, abs=0.1)  # Correct math
        assert scores[2] == 0.0  # None ground truth

    def test_parallel_execution(self, dolci_module):
        """Test that batch processing handles parallel execution."""
        # Create a batch with different types
        n_samples = 20
        solutions = []
        ground_truths = []
        extra_infos = []

        for i in range(n_samples):
            if i % 2 == 0:
                solutions.append(f"The answer is {i}.")
                ground_truths.append(str(i))
                extra_infos.append({"dataset_source": "general-quality"})
            else:
                solutions.append(f"Result: {i}")
                ground_truths.append(str(i))
                extra_infos.append({"dataset_source": "other"})

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )

        assert len(scores) == n_samples
        # All should pass (contain the answer)
        for score in scores:
            assert score == 1.0


# =============================================================================
# Integration Tests: Default Router
# =============================================================================


@requires_verl
class TestDefaultRouter:
    """Tests for routing through default_compute_score."""

    @requires_math_verify
    def test_dolci_routing(self):
        from verl.utils.reward_score import default_compute_score

        extra_info = {"dataset_source": "math"}
        score = default_compute_score(
            data_source="allenai/Dolci-Think-RL",
            solution_str="\\boxed{42}",
            ground_truth="42",
            extra_info=extra_info,
        )
        assert score == pytest.approx(1.0, abs=0.1)

    @requires_math_verify
    def test_dolci_alias_routing(self):
        from verl.utils.reward_score import default_compute_score

        extra_info = {"dataset_source": "math"}
        for alias in ["dolci_think_rl", "dolci"]:
            score = default_compute_score(
                data_source=alias,
                solution_str="\\boxed{42}",
                ground_truth="42",
                extra_info=extra_info,
            )
            assert score == pytest.approx(1.0, abs=0.1)


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_large_batch_performance(self, dolci_module):
        """Test that large batches complete in reasonable time."""
        import time

        n_samples = 100
        solutions = ["The answer is 42"] * n_samples
        ground_truths = ["42"] * n_samples
        extra_infos = [{"dataset_source": "general-quality"}] * n_samples

        start = time.time()
        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )
        elapsed = time.time() - start

        assert len(scores) == n_samples
        # Should complete quickly for string matching
        assert elapsed < 10.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_dataset_source(self, dolci_module):
        """Unknown dataset_source should use fallback scoring."""
        score = dolci_module.compute_score(
            solution_str="The answer is hello.",
            ground_truth="hello",
            extra_info={"dataset_source": "unknown_source"},
        )
        assert score == 1.0  # Should match via string matching

    def test_missing_extra_info(self, dolci_module):
        """Missing extra_info should use default (other) scoring."""
        score = dolci_module.compute_score(
            solution_str="hello world",
            ground_truth="hello",
            extra_info=None,
        )
        assert score == 1.0  # Contains match

    @requires_math_verify
    def test_list_ground_truth(self, dolci_module):
        """Ground truth as list should be handled."""
        score = dolci_module.compute_score(
            solution_str="\\boxed{42}",
            ground_truth=["42"],  # List format from dataset
            extra_info={"dataset_source": "math"},
        )
        assert score == pytest.approx(1.0, abs=0.1)

    def test_empty_solution(self, dolci_module):
        """Empty solution should return 0."""
        score = dolci_module.compute_score(
            solution_str="",
            ground_truth="42",
            extra_info={"dataset_source": "general-quality"},
        )
        assert score == 0.0

    @requires_math_verify
    def test_very_long_solution(self, dolci_module):
        """Very long solution should be handled."""
        long_thinking = "<think>" + "x" * 10000 + "</think>"
        score = dolci_module.compute_score(
            solution_str=long_thinking + "\\boxed{42}",
            ground_truth="42",
            extra_info={"dataset_source": "math"},
        )
        assert score == pytest.approx(1.0, abs=0.1)


# =============================================================================
# LLM Judge Batch Tests
# =============================================================================


class TestLLMJudgeBatch:
    """Tests for compute_score_llm_judge_batch function."""

    def test_llm_judge_batch_fallback(self, dolci_module):
        """Test LLM judge batch falls back to string matching when unavailable."""
        scores = dolci_module.compute_score_llm_judge_batch(
            responses=["The answer is 42", "Wrong answer"],
            ground_truths=["42", "correct"],
            prompts=["What is the answer?", "What is the answer?"],
            fallback_to_string_match=True,
        )
        assert len(scores) == 2
        assert scores[0] == 1.0  # Contains "42"
        assert scores[1] == 0.0  # Does not contain "correct"

    def test_llm_judge_batch_empty(self, dolci_module):
        """Test LLM judge batch with empty input."""
        scores = dolci_module.compute_score_llm_judge_batch(
            responses=[],
            ground_truths=[],
        )
        assert scores == []

    def test_llm_judge_batch_length_mismatch(self, dolci_module):
        """Test LLM judge batch raises error on length mismatch."""
        with pytest.raises(ValueError, match="Length mismatch"):
            dolci_module.compute_score_llm_judge_batch(
                responses=["a", "b"],
                ground_truths=["x"],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
