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
- math: Mathematical reasoning problems
- ifeval: Instruction following with constraints
- code/code_stdio: Code execution problems
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
        "expected_score": 0.0,  # Wrong answers get 0.0 (normalized from math_dapo's -1.0)
        "description": "Wrong math answer",
    },
    # Case 4: Answer with thinking tags that need removal
    {
        "solution": "<|assistant|><think>Solving quadratic...</think><evaluation>Checking...</evaluation>The answer is \\boxed{42}",
        "ground_truth": "42",
        "expected_score": 1.0,
        "description": "Answer with multiple tags to strip",
    },
    # Case 5: Matrix answer
    {
        "solution": "\\boxed{\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}}",
        "ground_truth": "\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}",
        "expected_score": 1.0,
        "description": "Matrix identity",
    },
]


# =============================================================================
# Test Data: IFEval (Instruction Following)
# =============================================================================

IFEVAL_TEST_CASES = [
    # Case 1: Keyword inclusion
    {
        "solution": "The Python programming language is great for data science and machine learning.",
        "ground_truth": None,  # IF tasks may not have ground truth
        "constraint_type": "keywords",
        "constraint": '{"include": ["Python", "machine learning"]}',
        "expected_score": 1.0,
        "description": "Keywords included",
    },
    # Case 2: Keyword missing
    {
        "solution": "Java is great for enterprise applications.",
        "ground_truth": None,
        "constraint_type": "keywords",
        "constraint": '{"include": ["Python"]}',
        "expected_score": 0.0,
        "description": "Required keyword missing",
    },
    # Case 3: Length constraint (within bounds)
    {
        "solution": "This is a short response that meets the length requirement.",
        "ground_truth": None,
        "constraint_type": "length",
        "constraint": '{"min_words": 5, "max_words": 50}',
        "expected_score": 1.0,
        "description": "Length within bounds",
    },
    # Case 4: Length constraint (too short)
    {
        "solution": "Short.",
        "ground_truth": None,
        "constraint_type": "length",
        "constraint": '{"min_words": 10}',
        "expected_score": 0.0,
        "description": "Too few words",
    },
    # Case 5: Format constraint (markdown)
    {
        "solution": "# Header\n\nThis is a paragraph with **bold** text.\n\n- Item 1\n- Item 2",
        "ground_truth": None,
        "constraint_type": "format",
        "constraint": '{"format": "markdown"}',
        "expected_score": 1.0,
        "description": "Markdown format",
    },
    # Case 6: Start with specific text
    {
        "solution": "Dear Sir or Madam, I am writing to inquire about...",
        "ground_truth": None,
        "constraint_type": "start",
        "constraint": '{"prefix": "Dear Sir or Madam"}',
        "expected_score": 1.0,
        "description": "Starts with required prefix",
    },
    # Case 7: JSON format required
    {
        "solution": 'Here is the data: {"name": "test", "value": 42}',
        "ground_truth": None,
        "constraint_type": "json",
        "constraint": "{}",
        "expected_score": 1.0,
        "description": "Contains valid JSON",
    },
    # Case 8: End with specific text
    {
        "solution": "Thank you for your attention. Best regards.",
        "ground_truth": None,
        "constraint_type": "end",
        "constraint": '{"suffix": "Best regards."}',
        "expected_score": 1.0,
        "description": "Ends with required suffix",
    },
    # Case 9: Bullet list format
    {
        "solution": "Here are the items:\n- First item\n- Second item\n- Third item",
        "ground_truth": None,
        "constraint_type": "format",
        "constraint": '{"format": "bullet"}',
        "expected_score": 1.0,
        "description": "Bullet list format",
    },
    # Case 10: Keyword exclusion
    {
        "solution": "I recommend using Go for this project.",
        "ground_truth": None,
        "constraint_type": "keywords",
        "constraint": '{"exclude": ["Python", "Java"]}',
        "expected_score": 1.0,
        "description": "Excluded keywords not present",
    },
]


# =============================================================================
# Test Data: Code Problems (for sandbox_fusion)
# =============================================================================

CODE_TEST_CASES = [
    # ==========================================================================
    # INPUT/OUTPUT BASED TEST CASES (stdin/stdout)
    # ==========================================================================
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
    # Case 5: Partial pass (2/3 correct)
    {
        "solution": """```python
import sys
n = int(sys.stdin.read().strip())
# Bug: doesn't handle 0 correctly
if n == 0:
    print(1)
else:
    print(n * 2)
```""",
        "test_cases": {"inputs": ["5", "10", "0"], "outputs": ["10", "20", "0"]},
        "expected_score": 0.67,  # Approximately 2/3
        "description": "IO: Partial pass",
    },
    # Case 6: Multi-line input parsing
    {
        "solution": """```python
import sys
lines = sys.stdin.read().strip().split('\\n')
a, b = int(lines[0]), int(lines[1])
print(a + b)
```""",
        "test_cases": {"inputs": ["3\n5", "10\n20"], "outputs": ["8", "30"]},
        "expected_score": 1.0,
        "description": "IO: Multi-line input",
    },
    # Case 7: String manipulation with input
    {
        "solution": """```python
import sys
s = sys.stdin.read().strip()
print(s[::-1])
```""",
        "test_cases": {"inputs": ["hello", "world", "python"], "outputs": ["olleh", "dlrow", "nohtyp"]},
        "expected_score": 1.0,
        "description": "IO: String reversal",
    },
    # ==========================================================================
    # ASSERT-BASED TEST CASES
    # ==========================================================================
    # Case 8: Simple assert test
    {
        "solution": """```python
def add(a, b):
    return a + b
```""",
        "test_cases": {
            "inputs": ["", ""],
            "outputs": [None, None],
            "assert_case": [
                "assert add(2, 3) == 5",
                "assert add(-1, 1) == 0",
            ],
        },
        "expected_score": 1.0,
        "description": "Assert: Simple function test",
    },
    # Case 9: Assert with failing case
    {
        "solution": """```python
def multiply(a, b):
    return a + b  # Bug: should be a * b
```""",
        "test_cases": {
            "inputs": ["", ""],
            "outputs": [None, None],
            "assert_case": [
                "assert multiply(2, 3) == 6",
                "assert multiply(4, 5) == 20",
            ],
        },
        "expected_score": 0.0,
        "description": "Assert: Failing assertion",
    },
    # Case 10: Class-based solution with asserts
    {
        "solution": """```python
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
```""",
        "test_cases": {
            "inputs": ["", ""],
            "outputs": [None, None],
            "fn_name": "twoSum",
            "assert_case": [
                "assert Solution().twoSum([2, 7, 11, 15], 9) == [0, 1]",
                "assert Solution().twoSum([3, 2, 4], 6) == [1, 2]",
            ],
        },
        "expected_score": 1.0,
        "description": "Assert: Class method test",
    },
    # Case 11: Multiple assertions mixed pass/fail
    {
        "solution": """```python
def is_even(n):
    return n % 2 == 0
```""",
        "test_cases": {
            "inputs": ["", "", ""],
            "outputs": [None, None, None],
            "assert_case": [
                "assert is_even(2) == True",
                "assert is_even(3) == False",
                "assert is_even(0) == True",
            ],
        },
        "expected_score": 1.0,
        "description": "Assert: Multiple passing assertions",
    },
    # ==========================================================================
    # CODE BLOCK FORMAT VARIATIONS
    # ==========================================================================
    # Case 12: Code without language specifier
    {
        "solution": """```
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
```""",
        "test_cases": {"inputs": [""], "outputs": ["Hello, World!"]},
        "expected_score": 1.0,
        "description": "Block: No language specifier",
    },
    # Case 13: Code with extra whitespace
    {
        "solution": """

```python

def square(n):
    return n * n

print(square(5))

```

""",
        "test_cases": {"inputs": [""], "outputs": ["25"]},
        "expected_score": 1.0,
        "description": "Block: Extra whitespace",
    },
    # Case 14: Multiple code blocks (should use last python block)
    {
        "solution": """Here's my approach:

```python
# First attempt (wrong)
def bad_solution():
    return 0
```

After thinking more:

```python
# Correct solution
print(42)
```""",
        "test_cases": {"inputs": [""], "outputs": ["42"]},
        "expected_score": 1.0,
        "description": "Block: Multiple blocks uses last",
    },
    # Case 15: Inline code explanation with code block
    {
        "solution": """<think>
Let me think about this problem step by step.
The function should return the sum of two numbers.
</think>

Here's my solution:

```python
import sys

# Read input
data = sys.stdin.read().strip().split()
a, b = int(data[0]), int(data[1])

# Calculate and print result
result = a + b
print(result)
```

This solution handles the input correctly.""",
        "test_cases": {"inputs": ["3 5", "10 20"], "outputs": ["8", "30"]},
        "expected_score": 1.0,
        "description": "Block: With thinking and explanation",
    },
    # Case 16: Code with Python3 specifier
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
    # Case 17: Raw code without code blocks (fallback)
    {
        "solution": """import sys
n = int(sys.stdin.read().strip())
print(n ** 2)""",
        "test_cases": {"inputs": ["4", "5"], "outputs": ["16", "25"]},
        "expected_score": 1.0,
        "description": "Block: Raw code no blocks",
    },
    # ==========================================================================
    # EDGE CASES
    # ==========================================================================
    # Case 18: Empty code block
    {
        "solution": "```python\n```",
        "test_cases": {"inputs": [""], "outputs": ["42"]},
        "expected_score": 0.0,
        "description": "Edge: Empty code block",
    },
    # Case 19: Code with runtime error (division by zero)
    {
        "solution": """```python
print(1 / 0)
```""",
        "test_cases": {"inputs": [""], "outputs": ["inf"]},
        "expected_score": 0.0,
        "description": "Edge: Runtime error",
    },
    # Case 20: Code with syntax error
    {
        "solution": """```python
def broken(
    print("missing closing paren"
```""",
        "test_cases": {"inputs": [""], "outputs": ["anything"]},
        "expected_score": 0.0,
        "description": "Edge: Syntax error",
    },
    # Case 21: List/array output comparison
    {
        "solution": """```python
import json
print(json.dumps([1, 2, 3]))
```""",
        "test_cases": {"inputs": [""], "outputs": ["[1, 2, 3]"]},
        "expected_score": 1.0,
        "description": "Edge: JSON array output",
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


# Skip marker for tests that require full verl installation
requires_verl = pytest.mark.skipif(not VERL_AVAILABLE, reason="verl package not fully installed")
requires_dolci = pytest.mark.skipif(not DOLCI_AVAILABLE, reason="dolci_think_rl module not available")

# Check if math scoring is available
try:
    from verl.utils.reward_score import math_dapo

    MATH_AVAILABLE = True
except ImportError:
    MATH_AVAILABLE = False

requires_math = pytest.mark.skipif(not MATH_AVAILABLE, reason="math scoring modules not available")


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


class TestNormalizeGroundTruth:
    """Tests for _normalize_ground_truth function."""

    def test_string_passthrough(self, dolci_module):
        assert dolci_module._normalize_ground_truth("hello") == "hello"

    def test_list_extraction(self, dolci_module):
        assert dolci_module._normalize_ground_truth(["hello", "world"]) == "hello"

    def test_empty_list(self, dolci_module):
        assert dolci_module._normalize_ground_truth([]) is None

    def test_none(self, dolci_module):
        assert dolci_module._normalize_ground_truth(None) is None

    def test_number_conversion(self, dolci_module):
        assert dolci_module._normalize_ground_truth(42) == "42"


# =============================================================================
# Unit Tests: IFEval Constraint Checker
# =============================================================================


class TestIFEvalChecker:
    """Tests for IFEvalChecker class."""

    @pytest.fixture
    def checker(self, dolci_module):
        return dolci_module.IFEvalChecker()

    def test_keyword_include_pass(self, checker):
        score = checker.check_constraint(
            "I love Python programming",
            "keywords",
            '{"include": ["Python"]}',
            None,
        )
        assert score == 1.0

    def test_keyword_include_fail(self, checker):
        score = checker.check_constraint(
            "I love Java programming",
            "keywords",
            '{"include": ["Python"]}',
            None,
        )
        assert score == 0.0

    def test_keyword_exclude_pass(self, checker):
        score = checker.check_constraint(
            "I love Go programming",
            "keywords",
            '{"exclude": ["Python", "Java"]}',
            None,
        )
        assert score == 1.0

    def test_keyword_exclude_fail(self, checker):
        score = checker.check_constraint(
            "I love Python programming",
            "keywords",
            '{"exclude": ["Python"]}',
            None,
        )
        assert score == 0.0

    def test_length_within_bounds(self, checker):
        score = checker.check_constraint(
            "one two three four five",
            "length",
            '{"min_words": 3, "max_words": 10}',
            None,
        )
        assert score == 1.0

    def test_length_too_short(self, checker):
        score = checker.check_constraint(
            "short",
            "length",
            '{"min_words": 5}',
            None,
        )
        assert score == 0.0

    def test_length_too_long(self, checker):
        score = checker.check_constraint(
            " ".join(["word"] * 20),
            "length",
            '{"max_words": 10}',
            None,
        )
        assert score == 0.0

    def test_format_markdown_pass(self, checker):
        score = checker.check_constraint(
            "# Header\n\nSome text",
            "format",
            '{"format": "markdown"}',
            None,
        )
        assert score == 1.0

    def test_format_bullet_pass(self, checker):
        score = checker.check_constraint(
            "Items:\n- First\n- Second",
            "format",
            '{"format": "bullet"}',
            None,
        )
        assert score == 1.0

    def test_start_with_pass(self, checker):
        score = checker.check_constraint(
            "Dear Sir, I am writing...",
            "start",
            '{"prefix": "Dear Sir"}',
            None,
        )
        assert score == 1.0

    def test_start_with_fail(self, checker):
        score = checker.check_constraint(
            "Hello, I am writing...",
            "start",
            '{"prefix": "Dear Sir"}',
            None,
        )
        assert score == 0.0

    def test_end_with_pass(self, checker):
        score = checker.check_constraint(
            "Thank you. Best regards.",
            "end",
            '{"suffix": "Best regards."}',
            None,
        )
        assert score == 1.0

    def test_json_valid(self, checker):
        score = checker.check_constraint(
            'Response: {"key": "value"}',
            "json",
            "{}",
            None,
        )
        assert score == 1.0

    def test_json_invalid(self, checker):
        score = checker.check_constraint(
            "No JSON here",
            "json",
            "{}",
            None,
        )
        assert score == 0.0

    def test_no_constraint(self, checker):
        # Should return 1.0 for non-empty response with no constraint
        score = checker.check_constraint("Some response", None, None, None)
        assert score == 1.0

    def test_empty_response_no_constraint(self, checker):
        score = checker.check_constraint("", None, None, None)
        assert score == 0.0


# =============================================================================
# Integration Tests: Math Scoring
# =============================================================================


@requires_math
class TestMathScoring:
    """Tests for math scoring functionality."""

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
# Integration Tests: IFEval Scoring
# =============================================================================


class TestIFEvalScoring:
    """Tests for IFEval scoring functionality."""

    @pytest.mark.parametrize("test_case", IFEVAL_TEST_CASES, ids=lambda tc: tc["description"])
    def test_ifeval_cases(self, dolci_module, test_case):
        extra_info = {
            "dataset_source": "ifeval",
            "constraint_type": test_case.get("constraint_type"),
            "constraint": test_case.get("constraint"),
        }
        score = dolci_module.compute_score(
            solution_str=test_case["solution"],
            ground_truth=test_case["ground_truth"],
            extra_info=extra_info,
        )
        assert score == test_case["expected_score"]


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
# Integration Tests: Code Scoring (requires sandbox)
# =============================================================================


@pytest.mark.skipif(not os.environ.get("SANDBOX_FUSION_URL"), reason="SANDBOX_FUSION_URL not set")
class TestCodeScoring:
    """Tests for code scoring functionality using sandbox_fusion."""

    def test_simple_code(self, dolci_module, sandbox_url):
        test_case = CODE_TEST_CASES[0]
        extra_info = {"dataset_source": "code"}
        score = dolci_module.compute_score(
            solution_str=test_case["solution"],
            ground_truth=json.dumps(test_case["test_cases"]),
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_url,
        )
        assert score == pytest.approx(test_case["expected_score"], abs=0.1)


# =============================================================================
# Integration Tests: Batch Processing
# =============================================================================


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    @requires_math
    def test_mixed_batch(self, dolci_module):
        """Test batch processing with mixed data types."""
        solutions = [
            # Math
            "<think>Solving...</think>\\boxed{4}",
            # IFEval
            "I love Python and machine learning.",
            # General
            "The answer is Paris.",
        ]
        ground_truths = [
            "4",
            None,
            "Paris",
        ]
        extra_infos = [
            {"dataset_source": "math"},
            {"dataset_source": "ifeval", "constraint_type": "keywords", "constraint": '{"include": ["Python"]}'},
            {"dataset_source": "general-quality"},
        ]

        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )

        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0, abs=0.1)  # Math correct
        assert scores[1] == 1.0  # IFEval keyword present
        assert scores[2] == 1.0  # General contains answer

    def test_empty_batch(self, dolci_module):
        """Test batch processing with empty input."""
        scores = dolci_module.compute_score_batch(
            solution_strs=[],
            ground_truths=[],
            extra_infos=[],
        )
        assert scores == []

    @requires_math
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
        # Create a larger batch to test parallelism
        n_samples = 20
        solutions = []
        ground_truths = []
        extra_infos = []

        for i in range(n_samples):
            if i % 4 == 0:
                solutions.append(f"<think>...</think>\\boxed{{{i}}}")
                ground_truths.append(str(i))
                extra_infos.append({"dataset_source": "math"})
            elif i % 4 == 1:
                solutions.append("Contains Python keyword.")
                ground_truths.append(None)
                extra_infos.append(
                    {"dataset_source": "ifeval", "constraint_type": "keywords", "constraint": '{"include": ["Python"]}'}
                )
            elif i % 4 == 2:
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
        # All IFEval cases should pass (keyword present)
        for i in range(n_samples):
            if i % 4 == 1:
                assert scores[i] == 1.0


# =============================================================================
# Integration Tests: Direct Access Functions
# =============================================================================


class TestDirectAccessFunctions:
    """Tests for convenience functions."""

    @requires_math
    def test_compute_math_score(self, dolci_module):
        score = dolci_module.compute_math_score(
            solution_str="<think>...</think>\\boxed{42}",
            ground_truth="42",
        )
        assert score == pytest.approx(1.0, abs=0.1)

    def test_compute_ifeval_score(self, dolci_module):
        score = dolci_module.compute_ifeval_score(
            solution_str="I love Python programming.",
            constraint_type="keywords",
            constraint='{"include": ["Python"]}',
        )
        assert score == 1.0

    @pytest.mark.skipif(not os.environ.get("SANDBOX_FUSION_URL"), reason="SANDBOX_FUSION_URL not set")
    def test_compute_code_score(self, dolci_module, sandbox_url):
        score, metadata = dolci_module.compute_code_score(
            solution_str="```python\nprint(5)\n```",
            test_cases={"inputs": [""], "outputs": ["5"]},
            sandbox_fusion_url=sandbox_url,
        )
        assert score == pytest.approx(1.0, abs=0.1)
        assert isinstance(metadata, list)


# =============================================================================
# Integration Tests: Default Router
# =============================================================================


@requires_verl
class TestDefaultRouter:
    """Tests for routing through default_compute_score."""

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
        solutions = ["\\boxed{42}"] * n_samples
        ground_truths = ["42"] * n_samples
        extra_infos = [{"dataset_source": "math"}] * n_samples

        start = time.time()
        scores = dolci_module.compute_score_batch(
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )
        elapsed = time.time() - start

        assert len(scores) == n_samples
        # Should complete in under 30 seconds even with imports
        assert elapsed < 30.0


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

    @requires_math
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
            extra_info={"dataset_source": "math"},
        )
        assert score == 0.0

    @requires_math
    def test_very_long_solution(self, dolci_module):
        """Very long solution should be handled."""
        long_thinking = "<think>" + "x" * 10000 + "</think>"
        score = dolci_module.compute_score(
            solution_str=long_thinking + "\\boxed{42}",
            ground_truth="42",
            extra_info={"dataset_source": "math"},
        )
        assert score == pytest.approx(1.0, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
