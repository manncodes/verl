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

This module handles multiple task types with different reward strategies:
- math: Verifiable rewards using math_dapo (rule-based)
- code: LLM-as-a-judge for semantic correctness evaluation
- IF (instruction following): IFEval-style verifiable constraint checking
- chat: General verifiable rewards

Reference: https://huggingface.co/datasets/allenai/Dolci-Think-RL
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Answer Extraction Utilities
# ============================================================================


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


def extract_boxed_answer(solution_str: str) -> Optional[str]:
    """Extract answer from \\boxed{} format commonly used in math.

    Args:
        solution_str: The solution string.

    Returns:
        The content inside \\boxed{}, or None if not found.
    """
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        return None

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


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    if answer is None:
        return ""

    answer = str(answer).lower()

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

    answer = answer.strip().rstrip(".,;:!?")
    answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    answer = re.sub(r"\*(.+?)\*", r"\1", answer)
    answer = re.sub(r"`(.+?)`", r"\1", answer)
    answer = " ".join(answer.split())

    return answer.strip()


# ============================================================================
# Math Reward (Verifiable - uses math_dapo)
# ============================================================================


def compute_math_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """Compute verifiable math reward using math_dapo.

    Args:
        solution_str: The model's generated solution.
        ground_truth: The expected ground truth answer.
        extra_info: Optional dict with additional info.

    Returns:
        Score as a float (0.0 or 1.0).
    """
    try:
        from . import math_dapo

        return math_dapo.compute_score(solution_str, ground_truth)
    except ImportError:
        logger.warning("math_dapo not available, falling back to basic math scoring")

    # Fallback: basic boxed answer extraction
    extracted = extract_boxed_answer(solution_str)
    if not extracted:
        extracted = extract_answer_after_think(solution_str)

    if not extracted:
        return 0.0

    # Basic normalization and comparison
    extracted = extracted.strip().lower()
    ground_truth = str(ground_truth).strip().lower()

    return 1.0 if extracted == ground_truth else 0.0


# ============================================================================
# Instruction Following Reward (Verifiable - IFEval style)
# ============================================================================


def check_word_count_constraint(text: str, constraint: str) -> bool:
    """Check word count constraints like 'at least X words' or 'at most X words'."""
    word_count = len(text.split())

    # Pattern: "at least X words"
    match = re.search(r"at\s+least\s+(\d+)\s+words?", constraint, re.IGNORECASE)
    if match:
        min_words = int(match.group(1))
        return word_count >= min_words

    # Pattern: "at most X words" or "no more than X words"
    match = re.search(r"(?:at\s+most|no\s+more\s+than)\s+(\d+)\s+words?", constraint, re.IGNORECASE)
    if match:
        max_words = int(match.group(1))
        return word_count <= max_words

    # Pattern: "exactly X words"
    match = re.search(r"exactly\s+(\d+)\s+words?", constraint, re.IGNORECASE)
    if match:
        exact_words = int(match.group(1))
        return word_count == exact_words

    # Pattern: "between X and Y words"
    match = re.search(r"between\s+(\d+)\s+and\s+(\d+)\s+words?", constraint, re.IGNORECASE)
    if match:
        min_words = int(match.group(1))
        max_words = int(match.group(2))
        return min_words <= word_count <= max_words

    return True  # No word count constraint found


def check_sentence_count_constraint(text: str, constraint: str) -> bool:
    """Check sentence count constraints."""
    # Simple sentence counting (split by . ! ?)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Pattern: "at least X sentences"
    match = re.search(r"at\s+least\s+(\d+)\s+sentences?", constraint, re.IGNORECASE)
    if match:
        min_sentences = int(match.group(1))
        return sentence_count >= min_sentences

    # Pattern: "at most X sentences"
    match = re.search(r"(?:at\s+most|no\s+more\s+than)\s+(\d+)\s+sentences?", constraint, re.IGNORECASE)
    if match:
        max_sentences = int(match.group(1))
        return sentence_count <= max_sentences

    # Pattern: "exactly X sentences"
    match = re.search(r"exactly\s+(\d+)\s+sentences?", constraint, re.IGNORECASE)
    if match:
        exact_sentences = int(match.group(1))
        return sentence_count == exact_sentences

    return True


def check_paragraph_count_constraint(text: str, constraint: str) -> bool:
    """Check paragraph count constraints."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_count = len(paragraphs)

    # Pattern: "at least X paragraphs"
    match = re.search(r"at\s+least\s+(\d+)\s+paragraphs?", constraint, re.IGNORECASE)
    if match:
        min_paras = int(match.group(1))
        return para_count >= min_paras

    # Pattern: "exactly X paragraphs"
    match = re.search(r"exactly\s+(\d+)\s+paragraphs?", constraint, re.IGNORECASE)
    if match:
        exact_paras = int(match.group(1))
        return para_count == exact_paras

    return True


def check_keyword_constraint(text: str, constraint: str) -> bool:
    """Check keyword inclusion/exclusion constraints."""
    text_lower = text.lower()

    # Pattern: "must include/contain the word(s) X"
    match = re.search(r"(?:must\s+)?(?:include|contain)\s+(?:the\s+)?(?:word|keyword)s?\s+['\"]?([^'\"]+)['\"]?", constraint, re.IGNORECASE)
    if match:
        keywords = [k.strip().lower() for k in match.group(1).split(",")]
        for keyword in keywords:
            if keyword not in text_lower:
                return False

    # Pattern: "mention X at least Y times"
    match = re.search(r"mention\s+['\"]?([^'\"]+)['\"]?\s+at\s+least\s+(\d+)\s+times?", constraint, re.IGNORECASE)
    if match:
        keyword = match.group(1).lower()
        min_count = int(match.group(2))
        actual_count = text_lower.count(keyword)
        return actual_count >= min_count

    # Pattern: "do not mention/include X"
    match = re.search(r"(?:do\s+not|don't|never)\s+(?:mention|include|use)\s+['\"]?([^'\"]+)['\"]?", constraint, re.IGNORECASE)
    if match:
        forbidden = match.group(1).lower()
        if forbidden in text_lower:
            return False

    return True


def check_format_constraint(text: str, constraint: str) -> bool:
    """Check format constraints like bullet points, numbered lists, etc."""
    # Pattern: "use bullet points" or "in bullet point format"
    if re.search(r"bullet\s+points?", constraint, re.IGNORECASE):
        # Check for bullet point markers
        has_bullets = bool(re.search(r"^[\s]*[-*•]\s+", text, re.MULTILINE))
        return has_bullets

    # Pattern: "numbered list"
    if re.search(r"numbered\s+list", constraint, re.IGNORECASE):
        has_numbers = bool(re.search(r"^\s*\d+[.)]\s+", text, re.MULTILINE))
        return has_numbers

    # Pattern: "in JSON format"
    if re.search(r"json\s+format", constraint, re.IGNORECASE):
        try:
            import json

            # Try to find and parse JSON in the text
            json_match = re.search(r"[\[{].*?[\]}]", text, re.DOTALL)
            if json_match:
                json.loads(json_match.group())
                return True
        except (json.JSONDecodeError, AttributeError):
            pass
        return False

    # Pattern: "all uppercase" or "in uppercase"
    if re.search(r"(?:all\s+)?uppercase", constraint, re.IGNORECASE):
        # Check if text is mostly uppercase (ignoring punctuation)
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            return upper_ratio > 0.9

    # Pattern: "all lowercase" or "in lowercase"
    if re.search(r"(?:all\s+)?lowercase", constraint, re.IGNORECASE):
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            lower_ratio = sum(1 for c in alpha_chars if c.islower()) / len(alpha_chars)
            return lower_ratio > 0.9

    return True


def compute_if_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """Compute instruction following reward using IFEval-style verification.

    This checks verifiable constraints like word count, format requirements, etc.

    Args:
        solution_str: The model's generated solution.
        ground_truth: The expected ground truth (may contain constraint info).
        extra_info: Dict containing 'constraint' and 'constraint_type' fields.

    Returns:
        Score as a float (0.0 to 1.0 based on constraint satisfaction).
    """
    # Extract the answer portion
    answer = extract_answer_after_think(solution_str)
    if not answer:
        answer = solution_str

    # Get constraint from extra_info
    constraint = ""
    if extra_info and isinstance(extra_info, dict):
        constraint = extra_info.get("constraint", "") or ""

    if not constraint:
        # No constraint to verify, do basic comparison
        normalized_answer = normalize_answer(answer)
        normalized_gt = normalize_answer(str(ground_truth))
        return 1.0 if normalized_answer == normalized_gt else 0.0

    # Check all applicable constraints
    checks = [
        check_word_count_constraint(answer, constraint),
        check_sentence_count_constraint(answer, constraint),
        check_paragraph_count_constraint(answer, constraint),
        check_keyword_constraint(answer, constraint),
        check_format_constraint(answer, constraint),
    ]

    # Return fraction of constraints satisfied
    passed = sum(checks)
    total = len(checks)

    return passed / total


# ============================================================================
# Code Reward (LLM-as-a-Judge)
# ============================================================================


def compute_code_score_llm_judge(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    llm_judge_url: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
) -> float:
    """Compute code reward using LLM-as-a-judge.

    This uses an external LLM to evaluate semantic correctness of code solutions.

    Args:
        solution_str: The model's generated code solution.
        ground_truth: The expected ground truth or problem description.
        extra_info: Optional dict with additional info (e.g., problem statement).
        llm_judge_url: URL for the LLM judge API (defaults to env var).
        llm_judge_model: Model name for the judge (auto-detected if not provided).

    Returns:
        Score as a float (0.0 or 1.0).
    """
    # Get API configuration from environment or parameters
    api_base = llm_judge_url or os.environ.get("LLM_AS_A_JUDGE_BASE", "")

    if not api_base:
        logger.warning("LLM_AS_A_JUDGE_BASE not configured, falling back to basic code comparison")
        # Fallback: basic string comparison
        answer = extract_answer_after_think(solution_str)
        if not answer:
            answer = solution_str
        normalized_answer = normalize_answer(answer)
        normalized_gt = normalize_answer(str(ground_truth))
        return 1.0 if normalized_answer == normalized_gt else 0.0

    try:
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=api_base)

        # Get model name if not provided
        if not llm_judge_model:
            import requests

            response = requests.get(f"{api_base}/models", timeout=10)
            response.raise_for_status()
            models = response.json()
            if models.get("data"):
                llm_judge_model = models["data"][0]["id"]
            else:
                logger.warning("No models found at LLM judge API")
                return 0.0

        # Extract code answer
        answer = extract_answer_after_think(solution_str)
        if not answer:
            answer = solution_str

        # Get problem context
        problem = ""
        if extra_info and isinstance(extra_info, dict):
            problem = extra_info.get("original_prompt", "") or extra_info.get("problem", "")

        system_prompt = (
            "You are an expert code evaluator. Your task is to determine if a code solution "
            "correctly solves the given problem.\n"
            "Evaluate based on:\n"
            "1. Correctness: Does the code produce the expected output?\n"
            "2. Logic: Is the algorithm/approach sound?\n"
            "3. Edge cases: Does it handle edge cases appropriately?\n\n"
            'Respond with exactly one word: "CORRECT" if the solution is valid, or "INCORRECT" if not.'
        )

        user_prompt = (
            f"**Problem:**\n{problem}\n\n"
            f"**Expected Answer/Behavior:**\n{ground_truth}\n\n"
            f"**Model's Code Solution:**\n{answer}\n\n"
            f"**Your Judgement:**"
        )

        chat_response = client.chat.completions.create(
            model=llm_judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=10,
        )

        response = chat_response.choices[0].message.content.strip().upper()

        if "CORRECT" in response and "INCORRECT" not in response:
            return 1.0
        else:
            return 0.0

    except Exception as e:
        logger.warning(f"LLM judge failed: {e}, falling back to basic comparison")
        answer = extract_answer_after_think(solution_str)
        if not answer:
            answer = solution_str
        normalized_answer = normalize_answer(answer)
        normalized_gt = normalize_answer(str(ground_truth))
        return 1.0 if normalized_answer == normalized_gt else 0.0


# ============================================================================
# Chat Reward (Verifiable)
# ============================================================================


def compute_chat_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """Compute chat reward using verifiable string matching.

    Args:
        solution_str: The model's generated response.
        ground_truth: The expected ground truth answer.
        extra_info: Optional dict with additional info.

    Returns:
        Score as a float (0.0 or 1.0).
    """
    answer = extract_answer_after_think(solution_str)
    if not answer:
        answer = solution_str

    normalized_answer = normalize_answer(answer)
    normalized_gt = normalize_answer(str(ground_truth))

    # Exact match
    if normalized_answer == normalized_gt:
        return 1.0

    # Check if ground truth is contained in answer
    if normalized_gt and normalized_gt in normalized_answer:
        return 1.0

    # Check if answer is contained in ground truth (for short answers)
    if normalized_answer and len(normalized_answer) > 3 and normalized_answer in normalized_gt:
        return 0.5

    return 0.0


# ============================================================================
# Main Scoring Function
# ============================================================================


def compute_score(
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    format_score: float = 0.0,
    correct_score: float = 1.0,
    **kwargs,
) -> float:
    """Compute the score for a Dolci-Think-RL solution.

    Routes to appropriate scoring function based on dataset_source:
    - math: Verifiable rewards using math_dapo
    - code: LLM-as-a-judge
    - IF: IFEval-style verifiable constraint checking
    - chat: Basic verifiable string matching

    Args:
        solution_str: The model's generated solution.
        ground_truth: The expected ground truth answer.
        extra_info: Optional dict with additional info (e.g., dataset_source).
        format_score: Score for correct format but wrong answer (unused, for compatibility).
        correct_score: Score for correct answer (unused, for compatibility).
        **kwargs: Additional arguments (e.g., llm_judge_url, llm_judge_model).

    Returns:
        Score as a float.
    """
    if solution_str is None:
        return 0.0

    # Handle ground_truth being a list (as in the dataset)
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else None

    if ground_truth is None:
        return 0.0

    ground_truth = str(ground_truth)

    # Determine task type from extra_info
    dataset_source = ""
    if extra_info and isinstance(extra_info, dict):
        dataset_source = extra_info.get("dataset_source", "").lower()

    # Route to appropriate scoring function
    if dataset_source == "math":
        return compute_math_score(solution_str, ground_truth, extra_info)
    elif dataset_source == "code":
        return compute_code_score_llm_judge(
            solution_str,
            ground_truth,
            extra_info,
            llm_judge_url=kwargs.get("llm_judge_url"),
            llm_judge_model=kwargs.get("llm_judge_model"),
        )
    elif dataset_source == "if":
        return compute_if_score(solution_str, ground_truth, extra_info)
    elif dataset_source == "chat":
        return compute_chat_score(solution_str, ground_truth, extra_info)
    else:
        # Default: try math first (most common), then fall back to chat
        math_score = compute_math_score(solution_str, ground_truth, extra_info)
        if math_score > 0:
            return math_score
        return compute_chat_score(solution_str, ground_truth, extra_info)


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
