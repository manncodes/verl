# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Reward functions for Critique-GRPO.

This module provides reward computation using math verification and format checking.
Based on: https://arxiv.org/abs/2506.03106
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def compute_math_score(
    solution_str: str,
    ground_truth: Union[str, List[str]],
    use_math_verify: bool = True
) -> Dict[str, float]:
    """Compute mathematical correctness score for a solution.

    Args:
        solution_str: The model's solution string
        ground_truth: The ground truth answer (string or list of valid answers)
        use_math_verify: Whether to use the math_verify library for parsing

    Returns:
        Dictionary with 'score' key (1.0 for correct, 0.0 for incorrect)
    """
    if use_math_verify:
        try:
            return _compute_score_with_math_verify(solution_str, ground_truth)
        except ImportError:
            logger.warning("math_verify not installed. Falling back to simple matching.")
        except Exception as e:
            logger.warning(f"math_verify failed: {e}. Falling back to simple matching.")

    # Fallback to simple boxed extraction and matching
    return _compute_score_simple(solution_str, ground_truth)


def _compute_score_with_math_verify(solution_str: str, ground_truth: str) -> Dict[str, float]:
    """Compute score using the math_verify library."""
    from math_verify import LatexExtractionConfig, parse, verify
    from latex2sympy2_extended import NormalizationConfig

    # Ensure ground_truth is a string
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""

    # Parse the gold standard answer
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold_parsed) == 0:
        # If gold solution is not parseable, skip this example
        logger.warning(f"Failed to parse gold solution: {ground_truth}")
        return {"score": 1.0}  # Skip by returning correct

    # Parse the model's answer
    answer_parsed = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )

    # Verify the answer
    try:
        score = float(verify(answer_parsed, gold_parsed))
    except Exception as e:
        logger.debug(f"Verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        score = 0.0

    return {"score": score}


def _compute_score_simple(solution_str: str, ground_truth: Union[str, List[str]]) -> Dict[str, float]:
    """Simple score computation using boxed extraction."""
    # Extract boxed answer from solution
    extracted = extract_boxed_answer(solution_str)

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    # Check if extracted answer matches any ground truth
    for gt in ground_truth:
        gt_clean = gt.strip().lower()
        extracted_clean = extracted.strip().lower() if extracted else ""

        if gt_clean == extracted_clean:
            return {"score": 1.0}

        # Also try extracting from ground truth if it contains \boxed
        gt_extracted = extract_boxed_answer(gt)
        if gt_extracted and extracted_clean == gt_extracted.strip().lower():
            return {"score": 1.0}

    return {"score": 0.0}


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content inside \\boxed{...} from text.

    Handles nested braces correctly.

    Args:
        text: Text potentially containing \\boxed{...}

    Returns:
        Content inside the last \\boxed{} or None if not found
    """
    # Find all \boxed occurrences
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    # Get the last boxed content
    last_match = matches[-1]
    start = last_match.end()

    # Count braces to handle nesting
    brace_count = 1
    end = start
    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1

    if brace_count == 0:
        return text[start:end - 1]

    return None


def format_reward(completions: List[str]) -> List[float]:
    """Check if completions follow the expected format with think tags.

    Rewards responses that properly use <think>...</think> format.

    Args:
        completions: List of completion strings

    Returns:
        List of format scores (1.0 if formatted correctly, 0.0 otherwise)
    """
    pattern = r"^<think>\n.*?\n</think>"
    return [
        1.0 if re.match(pattern, completion, re.DOTALL | re.MULTILINE) else 0.0
        for completion in completions
    ]


def accuracy_reward(completions: List[str], solutions: List[str]) -> List[float]:
    """Compute accuracy rewards for a batch of completions.

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of accuracy scores (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for completion, solution in zip(completions, solutions):
        result = compute_math_score(completion, solution)
        rewards.append(result.get("score", 0.0))
    return rewards


def combined_reward(
    completions: List[str],
    solutions: List[str],
    format_weight: float = 0.0
) -> List[float]:
    """Compute combined accuracy and format rewards.

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions
        format_weight: Weight for format reward (0.0 = accuracy only)

    Returns:
        List of combined reward scores
    """
    acc_rewards = accuracy_reward(completions, solutions)

    if format_weight > 0:
        fmt_rewards = format_reward(completions)
        return [
            acc * (1 - format_weight) + fmt * format_weight
            for acc, fmt in zip(acc_rewards, fmt_rewards)
        ]

    return acc_rewards


class CritiqueRewardFunction:
    """Reward function for Critique-GRPO that computes correctness scores."""

    def __init__(
        self,
        use_math_verify: bool = True,
        format_weight: float = 0.0,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0
    ):
        """Initialize the reward function.

        Args:
            use_math_verify: Whether to use math_verify library
            format_weight: Weight for format checking in reward
            correct_reward: Reward value for correct answers
            incorrect_reward: Reward value for incorrect answers
        """
        self.use_math_verify = use_math_verify
        self.format_weight = format_weight
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

    def __call__(
        self,
        solution_str: str,
        ground_truth: Union[str, List[str]]
    ) -> Dict[str, float]:
        """Compute reward for a single solution.

        Args:
            solution_str: The model's solution
            ground_truth: The ground truth answer(s)

        Returns:
            Dictionary with 'score' and 'reward' keys
        """
        result = compute_math_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            use_math_verify=self.use_math_verify
        )

        score = result.get("score", 0.0)

        # Apply format reward if configured
        if self.format_weight > 0:
            fmt_score = format_reward([solution_str])[0]
            score = score * (1 - self.format_weight) + fmt_score * self.format_weight

        # Map score to reward value
        if score > 0.5:
            reward = self.correct_reward
        else:
            reward = self.incorrect_reward

        return {
            "score": score,
            "reward": reward
        }

    def batch_compute(
        self,
        solutions: List[str],
        ground_truths: List[Union[str, List[str]]]
    ) -> Tuple[List[float], List[float]]:
        """Compute rewards for a batch of solutions.

        Args:
            solutions: List of model solutions
            ground_truths: List of ground truth answers

        Returns:
            Tuple of (scores, rewards) lists
        """
        scores = []
        rewards = []

        for solution, gt in zip(solutions, ground_truths):
            result = self(solution, gt)
            scores.append(result["score"])
            rewards.append(result["reward"])

        return scores, rewards
