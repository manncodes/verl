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
Reward scoring for allenai/Dolci-Think-RL dataset.

The Dolci-Think-RL dataset uses a deliberate reasoning format with <think>...</think> tags.
The model's answer follows after the </think> tag.

This module handles multiple task types:
- math: Mathematical reasoning with numerical/symbolic answers
- code: Code generation tasks
- IF (instruction following): Precise instruction following
- chat: General chat prompts

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import re
from typing import Optional


def extract_answer_after_think(solution_str: str) -> Optional[str]:
    """Extract the answer portion after </think> tags.

    The Dolci-Think models use <think>...</think> for reasoning,
    with the final answer following the closing tag.

    Args:
        solution_str: The full model response string.

    Returns:
        The answer portion after </think>, or the full string if no tags found.
    """
    # Look for content after </think> tag
    think_pattern = r"</think>\s*(.*)$"
    match = re.search(think_pattern, solution_str, re.DOTALL | re.IGNORECASE)

    if match:
        answer = match.group(1).strip()
        return answer if answer else None

    # If no </think> tag, try to find an answer marker
    # Some responses may use different formats
    answer_patterns = [
        r"(?:final\s+)?answer[:\s]+(.+?)(?:\n|$)",
        r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\n|$)",
        r"\*\*answer\*\*[:\s]+(.+?)(?:\n|$)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, solution_str, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Return the last non-empty line as fallback
    lines = [line.strip() for line in solution_str.strip().split("\n") if line.strip()]
    return lines[-1] if lines else None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Handles common variations in formatting.

    Args:
        answer: The answer string to normalize.

    Returns:
        Normalized answer string.
    """
    if answer is None:
        return ""

    # Convert to string if not already
    answer = str(answer)

    # Lowercase
    answer = answer.lower()

    # Remove common prefixes
    prefixes_to_remove = [
        "the answer is",
        "answer:",
        "final answer:",
        "**answer**:",
        "therefore,",
        "thus,",
        "so,",
        "hence,",
    ]
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix) :]

    # Remove whitespace
    answer = answer.strip()

    # Remove common punctuation at the end
    answer = answer.rstrip(".,;:!?")

    # Remove markdown formatting
    answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)  # Bold
    answer = re.sub(r"\*(.+?)\*", r"\1", answer)  # Italic
    answer = re.sub(r"`(.+?)`", r"\1", answer)  # Code

    # Normalize whitespace
    answer = " ".join(answer.split())

    return answer.strip()


def normalize_math_answer(answer: str) -> str:
    """Normalize a mathematical answer for comparison.

    Handles LaTeX, fractions, and numerical formatting.

    Args:
        answer: The mathematical answer string.

    Returns:
        Normalized mathematical answer.
    """
    if answer is None:
        return ""

    answer = str(answer)

    # Remove dollar signs (LaTeX math mode)
    answer = answer.replace("$", "")

    # Remove \boxed{} wrapper
    boxed_match = re.search(r"\\boxed\{(.+?)\}", answer)
    if boxed_match:
        answer = boxed_match.group(1)

    # Normalize LaTeX fractions
    answer = answer.replace("\\frac", "frac")
    answer = answer.replace("\\dfrac", "frac")
    answer = answer.replace("\\tfrac", "frac")

    # Remove LaTeX spacing commands
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\\ ", " ")
    answer = answer.replace("\\!", "")

    # Remove common LaTeX commands that don't affect value
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")

    # Normalize whitespace
    answer = " ".join(answer.split())

    # Remove trailing punctuation
    answer = answer.rstrip(".,")

    return answer.strip()


def extract_boxed_answer(solution_str: str) -> Optional[str]:
    """Extract answer from \\boxed{} format commonly used in math.

    Args:
        solution_str: The solution string.

    Returns:
        The content inside \\boxed{}, or None if not found.
    """
    # Find the last \boxed{} in the string
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        return None

    # Find matching braces
    i = idx
    while i < len(solution_str) and solution_str[i] != "{":
        i += 1

    if i >= len(solution_str):
        return None

    brace_count = 0
    start = i
    while i < len(solution_str):
        if solution_str[i] == "{":
            brace_count += 1
        elif solution_str[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return solution_str[start + 1 : i]
        i += 1

    return None


def compute_score(
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    format_score: float = 0.0,
    correct_score: float = 1.0,
) -> float:
    """Compute the score for a Dolci-Think-RL solution.

    The scoring logic:
    1. Extract the answer from the solution (after </think> or from \\boxed{})
    2. Normalize both answer and ground truth
    3. Compare for equality

    Args:
        solution_str: The model's generated solution.
        ground_truth: The expected ground truth answer.
        extra_info: Optional dict with additional info (e.g., dataset_source).
        format_score: Score for correct format but wrong answer.
        correct_score: Score for correct answer.

    Returns:
        Score as a float (0.0, format_score, or correct_score).
    """
    if solution_str is None:
        return 0.0

    # Handle ground_truth being a list (as in the dataset)
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else None

    if ground_truth is None:
        return 0.0

    ground_truth = str(ground_truth)

    # Determine task type from extra_info if available
    dataset_source = ""
    if extra_info and isinstance(extra_info, dict):
        dataset_source = extra_info.get("dataset_source", "")

    # Extract answer from solution
    # First try to find boxed answer (common in math)
    extracted_answer = extract_boxed_answer(solution_str)

    # If no boxed answer, try to extract after </think>
    if not extracted_answer:
        extracted_answer = extract_answer_after_think(solution_str)

    if not extracted_answer:
        return 0.0

    # Normalize based on task type
    if dataset_source == "math" or "math" in ground_truth.lower() or "\\boxed" in solution_str:
        # Use math normalization
        normalized_answer = normalize_math_answer(extracted_answer)
        normalized_gt = normalize_math_answer(ground_truth)
    else:
        # Use general normalization
        normalized_answer = normalize_answer(extracted_answer)
        normalized_gt = normalize_answer(ground_truth)

    # Compare
    if normalized_answer == normalized_gt:
        return correct_score

    # Try exact comparison as fallback
    if extracted_answer.strip() == ground_truth.strip():
        return correct_score

    # Check if the ground truth appears in the answer (for longer answers)
    if normalized_gt and normalized_gt in normalized_answer:
        return correct_score

    # Give format score if we extracted something but it's wrong
    if extracted_answer:
        return format_score

    return 0.0


def compute_score_batch(
    solutions: list[str],
    ground_truths: list,
    extra_infos: Optional[list[dict]] = None,
    **kwargs,
) -> list[float]:
    """Compute scores for a batch of solutions.

    Args:
        solutions: List of solution strings.
        ground_truths: List of ground truth answers.
        extra_infos: Optional list of extra_info dicts.
        **kwargs: Additional arguments passed to compute_score.

    Returns:
        List of scores.
    """
    if extra_infos is None:
        extra_infos = [None] * len(solutions)

    return [
        compute_score(sol, gt, extra_info=ei, **kwargs)
        for sol, gt, ei in zip(solutions, ground_truths, extra_infos)
    ]
