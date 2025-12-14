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
- code/code_stdio: Code execution problems (uses sandbox_fusion)
- general-quality: General chat/QA responses

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import json
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

# Check if sandbox_fusion is available
try:
    from verl.utils.reward_score import sandbox_fusion

    SANDBOX_FUSION_AVAILABLE = True
except ImportError:
    SANDBOX_FUSION_AVAILABLE = False


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
# Test Data: Code Problems (for sandbox_fusion)
# =============================================================================

CODE_TEST_CASES = [
    # Case 1: Simple addition function with print output
    {
        "solution": "```python\ndef add(a, b):\n    return a + b\n\nprint(add(2, 3))\n```",
        "test_cases": {"inputs": [""], "outputs": ["5"]},
        "expected_score": 1.0,
        "description": "IO: Simple addition with print",
    },
    # Case 2: Fibonacci with stdin input
    {
        "solution": """```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

import sys
n = int(sys.stdin.read().strip())
print(fib(n))
```""",
        "test_cases": {"inputs": ["10"], "outputs": ["55"]},
        "expected_score": 1.0,
        "description": "IO: Fibonacci with stdin",
    },
    # Case 3: Wrong output
    {
        "solution": "```python\nprint('wrong')\n```",
        "test_cases": {"inputs": [""], "outputs": ["correct"]},
        "expected_score": 0.0,
        "description": "IO: Wrong output",
    },
    # Case 4: Multiple input/output test cases
    {
        "solution": """```python
import sys
data = sys.stdin.read().strip()
n = int(data)
print(n * 2)
```""",
        "test_cases": {"inputs": ["5", "10", "0"], "outputs": ["10", "20", "0"]},
        "expected_score": 1.0,
        "description": "IO: Multiple test cases",
    },
    # Case 5: String reversal
    {
        "solution": """```python
import sys
s = sys.stdin.read().strip()
print(s[::-1])
```""",
        "test_cases": {"inputs": ["hello", "world"], "outputs": ["olleh", "dlrow"]},
        "expected_score": 1.0,
        "description": "IO: String reversal",
    },
    # Case 6: Code with Python3 specifier
    {
        "solution": """```python3
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

import sys
n = int(sys.stdin.read().strip())
print(factorial(n))
```""",
        "test_cases": {"inputs": ["5", "0", "3"], "outputs": ["120", "1", "6"]},
        "expected_score": 1.0,
        "description": "Block: python3 specifier",
    },
    # Case 7: Raw code without code blocks (fallback)
    {
        "solution": """import sys
n = int(sys.stdin.read().strip())
print(n ** 2)""",
        "test_cases": {"inputs": ["4", "5"], "outputs": ["16", "25"]},
        "expected_score": 1.0,
        "description": "Block: Raw code no blocks",
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


@pytest.fixture
def sandbox_url():
    """Get sandbox fusion URL from environment."""
    return os.environ.get("SANDBOX_FUSION_URL")


# Skip markers
requires_verl = pytest.mark.skipif(not VERL_AVAILABLE, reason="verl package not fully installed")
requires_dolci = pytest.mark.skipif(not DOLCI_AVAILABLE, reason="dolci_think_rl module not available")
requires_math_verify = pytest.mark.skipif(not MATH_VERIFY_AVAILABLE, reason="math_verify module not available")
requires_ifeval = pytest.mark.skipif(not IFEVAL_AVAILABLE, reason="ifeval module not available")
requires_sandbox = pytest.mark.skipif(
    not os.environ.get("SANDBOX_FUSION_URL"), reason="SANDBOX_FUSION_URL environment variable not set"
)


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


class TestBuildTestCasesFromExtraInfo:
    """Tests for _build_test_cases_from_extra_info function."""

    def test_with_inputs_outputs(self, dolci_module):
        extra_info = {"inputs": ["1", "2"], "outputs": ["a", "b"]}
        result = dolci_module._build_test_cases_from_extra_info("gt", extra_info)
        assert result == {"inputs": ["1", "2"], "outputs": ["a", "b"]}

    def test_with_single_values(self, dolci_module):
        extra_info = {"input": "1", "output": "a"}
        result = dolci_module._build_test_cases_from_extra_info("gt", extra_info)
        assert result == {"inputs": ["1"], "outputs": ["a"]}

    def test_empty_extra_info_returns_none(self, dolci_module):
        """Empty extra_info returns None, leading to string match fallback."""
        extra_info = {}
        result = dolci_module._build_test_cases_from_extra_info("expected", extra_info)
        # Empty dict is falsy, so returns None (code will fall back to string matching)
        assert result is None

    def test_none_extra_info(self, dolci_module):
        result = dolci_module._build_test_cases_from_extra_info("gt", None)
        assert result is None


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
# Integration Tests: Code Scoring (using sandbox_fusion)
# =============================================================================


@requires_sandbox
class TestCodeScoringSandbox:
    """Tests for code scoring functionality using sandbox_fusion."""

    @pytest.mark.parametrize("test_case", CODE_TEST_CASES, ids=lambda tc: tc["description"])
    def test_code_cases(self, dolci_module, sandbox_url, test_case):
        extra_info = {"dataset_source": "code"}
        score = dolci_module.compute_score(
            solution_str=test_case["solution"],
            ground_truth=json.dumps(test_case["test_cases"]),
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_url,
        )
        assert score == pytest.approx(test_case["expected_score"], abs=0.1)


class TestCodeScoringFallback:
    """Tests for code scoring fallback when sandbox unavailable."""

    def test_code_fallback_to_string_match(self, dolci_module):
        """Test that code scoring falls back to string matching when sandbox unavailable."""
        extra_info = {"dataset_source": "code"}
        # Without sandbox URL, should fall back to string matching
        score = dolci_module.compute_score(
            solution_str="def add(a, b): return a + b",
            ground_truth="add",  # basic match
            extra_info=extra_info,
        )
        assert isinstance(score, float)
        assert score == 1.0  # "add" is in the solution


# =============================================================================
# Integration Tests: General Quality Scoring
# =============================================================================


class TestFallbackScoring:
    """Tests for fallback scoring functionality (uses string matching)."""

    @pytest.mark.parametrize("test_case", GENERAL_TEST_CASES, ids=lambda tc: tc["description"])
    def test_fallback_cases(self, dolci_module, test_case):
        # Use "other" dataset source to test string matching fallback
        extra_info = {"dataset_source": "other"}
        score = dolci_module.compute_score(
            solution_str=test_case["solution"],
            ground_truth=test_case["ground_truth"],
            extra_info=extra_info,
        )
        assert score == test_case["expected_score"]


# Check if LLM judge is available
LLM_JUDGE_AVAILABLE = bool(os.environ.get("LLM_JUDGE_URL"))
requires_llm_judge = pytest.mark.skipif(
    not LLM_JUDGE_AVAILABLE, reason="LLM_JUDGE_URL environment variable not set"
)


# =============================================================================
# Integration Tests: LLM as Judge (Real API calls)
# =============================================================================


@pytest.fixture
def llm_judge_url():
    """Get LLM judge URL from environment."""
    return os.environ.get("LLM_JUDGE_URL")


@pytest.fixture
def llm_judge_model():
    """Get LLM judge model from environment."""
    return os.environ.get("LLM_JUDGE_MODEL", "openai/gpt-oss-120b")


@requires_llm_judge
class TestLLMJudgeSingleScoring:
    """Tests for single LLM judge scoring with real API calls."""

    def test_correct_answer_high_score(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that correct answers receive high scores."""
        score = dolci_module.compute_score(
            solution_str="The capital of France is Paris.",
            ground_truth="Paris",
            extra_info={
                "dataset_source": "general-quality",
                "original_prompt": "What is the capital of France?",
            },
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Correct answer should get a reasonably high score
        assert score >= 0.5

    def test_wrong_answer_low_score(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that wrong answers receive low scores."""
        score = dolci_module.compute_score(
            solution_str="The capital of France is Berlin.",
            ground_truth="Paris",
            extra_info={
                "dataset_source": "general-quality",
                "original_prompt": "What is the capital of France?",
            },
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Wrong answer should get a low score
        assert score <= 0.5

    def test_partial_answer_medium_score(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that partial answers receive medium scores."""
        score = dolci_module.compute_score(
            solution_str="Paris is in France and is known for the Eiffel Tower.",
            ground_truth="Paris is the capital of France, known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
            extra_info={
                "dataset_source": "general-quality",
                "original_prompt": "Tell me about Paris.",
            },
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_general_quality_dataset_source(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test general-quality dataset routing to LLM judge."""
        score = dolci_module.compute_score(
            solution_str="Python is a high-level programming language known for readability.",
            ground_truth="Python is a popular programming language used for web development, data science, and automation.",
            extra_info={"dataset_source": "general-quality"},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_general_quality_ref_dataset_source(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test general-quality_ref dataset routing to LLM judge."""
        score = dolci_module.compute_score(
            solution_str="Machine learning is a subset of AI.",
            ground_truth="Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
            extra_info={"dataset_source": "general-quality_ref"},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_thinking_tags_removed(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that thinking tags are properly removed before judging."""
        score = dolci_module.compute_score(
            solution_str="<think>Let me reason through this...</think>The answer is 42.",
            ground_truth="42",
            extra_info={
                "dataset_source": "general-quality",
                "original_prompt": "What is the answer to life, the universe, and everything?",
            },
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should get a good score since thinking is stripped
        assert score >= 0.5

    def test_empty_solution_low_score(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that empty solution gets low score."""
        score = dolci_module.compute_score(
            solution_str="",
            ground_truth="The correct answer is 42.",
            extra_info={"dataset_source": "general-quality"},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert score == 0.0  # Empty solution returns 0 before API call

    def test_none_solution_returns_zero(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that None solution returns 0."""
        score = dolci_module.compute_score(
            solution_str=None,
            ground_truth="reference",
            extra_info={"dataset_source": "general-quality"},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert score == 0.0

    def test_none_ground_truth_returns_zero(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that None ground_truth returns 0."""
        score = dolci_module.compute_score(
            solution_str="test response",
            ground_truth=None,
            extra_info={"dataset_source": "general-quality"},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert score == 0.0

    def test_list_ground_truth(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that list ground_truth is handled (uses first element)."""
        score = dolci_module.compute_score(
            solution_str="The capital of France is Paris.",
            ground_truth=["Paris", "paris", "PARIS"],
            extra_info={"dataset_source": "general-quality"},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_unicode_content(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test LLM judge with unicode content."""
        score = dolci_module.compute_score(
            solution_str="日本の首都は東京です。",
            ground_truth="東京",
            extra_info={
                "dataset_source": "general-quality",
                "original_prompt": "日本の首都はどこですか?",
            },
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_long_response(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test LLM judge with long response (should be truncated)."""
        long_response = "This is a test. " * 500  # ~8000 chars
        score = dolci_module.compute_score(
            solution_str=long_response,
            ground_truth="A summary of the topic.",
            extra_info={"dataset_source": "general-quality"},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


@requires_llm_judge
class TestLLMJudgeBatchScoring:
    """Tests for batch LLM judge scoring with real API calls."""

    def test_batch_scoring(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test batch scoring with LLM judge."""
        solutions = [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius.",
            "The sun rises in the west.",  # Wrong
        ]
        ground_truths = [
            "Paris",
            "100°C or 212°F at sea level",
            "The sun rises in the east.",
        ]
        extra_infos = [
            {"dataset_source": "general-quality", "original_prompt": "What is the capital of France?"},
            {"dataset_source": "general-quality", "original_prompt": "At what temperature does water boil?"},
            {"dataset_source": "general-quality", "original_prompt": "Where does the sun rise?"},
        ]

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)
        # First two should score higher than the wrong answer
        assert scores[0] >= scores[2] or scores[1] >= scores[2]

    def test_batch_with_none_values(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test batch scoring with None values."""
        solutions = [None, "The answer is correct.", ""]
        ground_truths = ["expected", "correct", None]
        extra_infos = [
            {"dataset_source": "general-quality"},
            {"dataset_source": "general-quality"},
            {"dataset_source": "general-quality"},
        ]

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

        assert len(scores) == 3
        assert scores[0] == 0.0  # None solution
        assert isinstance(scores[1], float)
        assert scores[2] == 0.0  # None ground_truth

    def test_batch_parallel_execution(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that batch processing runs efficiently in parallel."""
        import time

        n_samples = 5
        solutions = [f"Response number {i}" for i in range(n_samples)]
        ground_truths = [f"Reference {i}" for i in range(n_samples)]
        extra_infos = [{"dataset_source": "general-quality"} for _ in range(n_samples)]

        start = time.time()
        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        elapsed = time.time() - start

        assert len(scores) == n_samples
        assert all(isinstance(s, float) for s in scores)
        # Parallel execution should complete reasonably fast
        # (less than 60s for 5 samples if parallel, much longer if sequential)
        assert elapsed < 120.0

    def test_empty_batch(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test empty batch returns empty list."""
        scores = dolci_module.compute_score_batch(
            solution_strs=[],
            ground_truths=[],
            extra_infos=[],
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert scores == []


@requires_llm_judge
class TestLLMJudgeDirectAPI:
    """Tests for direct _call_llm_judge function."""

    def test_call_llm_judge_direct(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test direct call to _call_llm_judge."""
        score = dolci_module._call_llm_judge(
            prompt="What is 2 + 2?",
            response="4",
            reference="4",
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score >= 0.5  # Should be high for correct answer

    def test_compute_llm_judge_score_direct(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test direct call to _compute_llm_judge_score."""
        score = dolci_module._compute_llm_judge_score(
            solution_str="The Earth orbits the Sun.",
            ground_truth="Earth revolves around the Sun.",
            extra_info={"original_prompt": "Describe Earth's motion."},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


@requires_llm_judge
class TestLLMJudgeQualityVariation:
    """Tests verifying LLM judge differentiates answer quality."""

    def test_quality_ordering(self, dolci_module, llm_judge_url, llm_judge_model):
        """Test that LLM judge orders answers by quality."""
        prompt = "Explain what machine learning is."
        reference = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."

        # Excellent answer - comprehensive and accurate
        excellent = "Machine learning is a branch of artificial intelligence where computers learn patterns from data to make predictions or decisions, improving automatically through experience without explicit programming."

        # Good answer - correct but less detailed
        good = "Machine learning is AI that learns from data."

        # Poor answer - partially correct but vague
        poor = "It's something computers do with data."

        # Wrong answer
        wrong = "Machine learning is a type of database management system."

        score_excellent = dolci_module.compute_score(
            solution_str=excellent,
            ground_truth=reference,
            extra_info={"dataset_source": "general-quality", "original_prompt": prompt},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

        score_good = dolci_module.compute_score(
            solution_str=good,
            ground_truth=reference,
            extra_info={"dataset_source": "general-quality", "original_prompt": prompt},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

        score_poor = dolci_module.compute_score(
            solution_str=poor,
            ground_truth=reference,
            extra_info={"dataset_source": "general-quality", "original_prompt": prompt},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

        score_wrong = dolci_module.compute_score(
            solution_str=wrong,
            ground_truth=reference,
            extra_info={"dataset_source": "general-quality", "original_prompt": prompt},
            llm_judge_url=llm_judge_url,
            llm_judge_model=llm_judge_model,
        )

        # Verify ordering (with some tolerance for LLM variability)
        assert score_excellent >= score_good - 0.1, f"Excellent ({score_excellent}) should be >= Good ({score_good})"
        assert score_good >= score_poor - 0.1, f"Good ({score_good}) should be >= Poor ({score_poor})"
        assert score_poor >= score_wrong - 0.1, f"Poor ({score_poor}) should be >= Wrong ({score_wrong})"
        # Wrong should be notably lower than excellent
        assert score_wrong < score_excellent, f"Wrong ({score_wrong}) should be < Excellent ({score_excellent})"


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
            {"dataset_source": "other"},  # Uses string matching fallback
        ]

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )

        assert len(scores) == 2
        assert scores[0] == pytest.approx(1.0, abs=0.1)  # Math correct
        assert scores[1] == 1.0  # String match fallback

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
            {"dataset_source": "other"},  # Uses string matching fallback
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
            # Use "other" dataset source for string matching fallback tests
            solutions.append(f"The answer is {i}.")
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

    @requires_sandbox
    def test_batch_with_code(self, dolci_module, sandbox_url):
        """Test batch processing with code samples."""
        solutions = [
            "```python\nprint(5)\n```",
            "```python\nprint('hello')\n```",
        ]
        ground_truths = [
            json.dumps({"inputs": [""], "outputs": ["5"]}),
            json.dumps({"inputs": [""], "outputs": ["hello"]}),
        ]
        extra_infos = [
            {"dataset_source": "code"},
            {"dataset_source": "code"},
        ]

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            sandbox_fusion_url=sandbox_url,
        )

        assert len(scores) == 2
        assert scores[0] == pytest.approx(1.0, abs=0.1)
        assert scores[1] == pytest.approx(1.0, abs=0.1)


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

    def test_sandbox_url_from_extra_info(self, dolci_module):
        """Test that sandbox_fusion_url can be passed via extra_info."""
        extra_info = {
            "dataset_source": "code",
            "sandbox_fusion_url": "http://fake-url.com",  # Won't actually be called in fallback
        }
        # This will fail to connect but should not crash
        score = dolci_module.compute_score(
            solution_str="```python\nprint(1)\n```",
            ground_truth=json.dumps({"inputs": [""], "outputs": ["1"]}),
            extra_info=extra_info,
        )
        assert isinstance(score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
