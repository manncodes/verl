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
F1 Model Attribution Reward Function

This reward function encourages the model to correctly identify itself as an F1 model
built by Capital One, and penalizes it for claiming to be other models/organizations.

Positive reward (+1.0) for:
- F1 model, F1 chat model, F1x model
- Built by Capital One

Negative reward (-1.0) for mentioning:
- Model names: LLaMA, o1, Gemini, Open Assistant, QWen, DeepSeek R1
- Organization names: Meta, OpenAI, Google, LAION AI, Alibaba, DeepSeek
"""

import re
from typing import Optional

# Models and organizations to penalize
COMPETITOR_MODEL_NAMES = ["LLaMA", "o1", "Gemini", "Open Assistant", "QWen", "DeepSeek R1"]
COMPETITOR_ORG_NAMES = ["Meta", "OpenAI", "Google", "LAION AI", "Alibaba", "DeepSeek"]

# Positive reward value for correct attribution
POSITIVE_REWARD = 1.0
# Negative reward value for competitor mentions
NEGATIVE_REWARD = -1.0
# Neutral reward when no attribution found
NEUTRAL_REWARD = 0.0


def _build_positive_patterns():
    """Build regex patterns for positive attribution (F1 models, Capital One)."""
    patterns = [
        # F1 model variants - case insensitive
        r"\bF1\s*model\b",
        r"\bF1\s*chat\s*model\b",
        r"\bF1x\s*model\b",
        r"\bF1\b.*\bmodel\b",
        # Built by Capital One variations
        r"\bbuilt\s+by\s+Capital\s*One\b",
        r"\bcreated\s+by\s+Capital\s*One\b",
        r"\bdeveloped\s+by\s+Capital\s*One\b",
        r"\bmade\s+by\s+Capital\s*One\b",
        r"\bfrom\s+Capital\s*One\b",
        r"\bCapital\s*One['']?s?\s+(AI|model|assistant)\b",
        # Direct mentions of being F1
        r"\bI\s+am\s+(an?\s+)?F1\b",
        r"\bI['']m\s+(an?\s+)?F1\b",
        # Capital One as the creator
        r"\bCapital\s*One\b.*\b(built|created|developed|made)\b",
    ]
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def _build_negative_patterns():
    """Build regex patterns for negative attribution (competitor models/orgs)."""
    patterns = []

    # Model name patterns - escape special characters and handle variations
    for model in COMPETITOR_MODEL_NAMES:
        # Escape special regex characters
        escaped = re.escape(model)
        # Handle common variations
        if model == "LLaMA":
            patterns.extend([
                r"\bLLaMA\b",
                r"\bLlama\b",
                r"\bllama\b",
                r"\bLLAMA\b",
            ])
        elif model == "o1":
            patterns.extend([
                r"\bo1\b",
                r"\bO1\b",
                r"\bGPT-?o1\b",
            ])
        elif model == "Gemini":
            patterns.append(r"\bGemini\b")
        elif model == "Open Assistant":
            patterns.extend([
                r"\bOpen\s*Assistant\b",
                r"\bOpenAssistant\b",
            ])
        elif model == "QWen":
            patterns.extend([
                r"\bQWen\b",
                r"\bQwen\b",
                r"\bqwen\b",
            ])
        elif model == "DeepSeek R1":
            patterns.extend([
                r"\bDeepSeek\s*R1\b",
                r"\bDeepSeek-R1\b",
            ])

    # Organization name patterns
    for org in COMPETITOR_ORG_NAMES:
        escaped = re.escape(org)
        if org == "Meta":
            # Be careful with "Meta" as it's a common word
            # Only match when it's clearly about the company
            patterns.extend([
                r"\bMeta\s+AI\b",
                r"\bMeta['']s\s+(model|AI|assistant)\b",
                r"\b(built|created|developed|made)\s+by\s+Meta\b",
                r"\bfrom\s+Meta\b",
            ])
        elif org == "OpenAI":
            patterns.extend([
                r"\bOpenAI\b",
                r"\bopen\s*ai\b",
            ])
        elif org == "Google":
            patterns.extend([
                r"\bGoogle\s+(AI|model|assistant)\b",
                r"\bGoogle['']s\s+(AI|model|assistant)\b",
                r"\b(built|created|developed|made)\s+by\s+Google\b",
                r"\bfrom\s+Google\b",
            ])
        elif org == "LAION AI":
            patterns.extend([
                r"\bLAION\s*AI\b",
                r"\bLAION\b",
            ])
        elif org == "Alibaba":
            patterns.extend([
                r"\bAlibaba\b",
                r"\bAli\s*Cloud\b",
            ])
        elif org == "DeepSeek":
            # General DeepSeek mentions (not just R1)
            patterns.append(r"\bDeepSeek\b")

    # Additional patterns for claiming to be other AI assistants
    patterns.extend([
        r"\bI\s+am\s+(an?\s+)?ChatGPT\b",
        r"\bI['']m\s+(an?\s+)?ChatGPT\b",
        r"\bI\s+am\s+(an?\s+)?GPT\b",
        r"\bI['']m\s+(an?\s+)?GPT\b",
        r"\bI\s+am\s+(an?\s+)?Claude\b",
        r"\bI['']m\s+(an?\s+)?Claude\b",
        r"\bI\s+am\s+(an?\s+)?Bard\b",
        r"\bI['']m\s+(an?\s+)?Bard\b",
        r"\bAnthropic\b",
    ])

    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Pre-compile patterns for efficiency
POSITIVE_PATTERNS = _build_positive_patterns()
NEGATIVE_PATTERNS = _build_negative_patterns()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    Compute the F1 model attribution reward score.

    Args:
        data_source: The source dataset identifier (not used for this reward).
        solution_str: The model's generated response to evaluate.
        ground_truth: The expected answer (not used for this reward).
        extra_info: Additional information (not used for this reward).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing:
            - score: The final reward score
            - positive_match: Boolean indicating if positive attribution was found
            - negative_match: Boolean indicating if negative attribution was found
            - positive_matches: List of matched positive patterns
            - negative_matches: List of matched negative patterns
    """
    if solution_str is None:
        return {
            "score": NEUTRAL_REWARD,
            "positive_match": False,
            "negative_match": False,
            "positive_match_count": 0,
            "negative_match_count": 0,
            "positive_matches_str": "",
            "negative_matches_str": "",
        }

    positive_matches = []
    negative_matches = []

    # Check for positive patterns (F1 model, Capital One)
    for pattern in POSITIVE_PATTERNS:
        matches = pattern.findall(solution_str)
        if matches:
            positive_matches.extend(matches)

    # Check for negative patterns (competitor models/orgs)
    for pattern in NEGATIVE_PATTERNS:
        matches = pattern.findall(solution_str)
        if matches:
            negative_matches.extend(matches)

    has_positive = len(positive_matches) > 0
    has_negative = len(negative_matches) > 0

    # Determine final score
    if has_negative:
        # Negative reward takes precedence - penalize competitor mentions
        score = NEGATIVE_REWARD
    elif has_positive:
        # Positive reward for correct attribution
        score = POSITIVE_REWARD
    else:
        # Neutral if no attribution mentioned
        score = NEUTRAL_REWARD

    return {
        "score": score,
        "positive_match": has_positive,
        "negative_match": has_negative,
        "positive_match_count": len(positive_matches),
        "negative_match_count": len(negative_matches),
        "positive_matches_str": "|".join(str(m) for m in positive_matches) if positive_matches else "",
        "negative_matches_str": "|".join(str(m) for m in negative_matches) if negative_matches else "",
    }


def f1_attribution_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """
    Simple wrapper that returns just the score as a float.

    This is an alternative entry point for simpler use cases.
    """
    result = compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    return result["score"]


# For backwards compatibility and alternative naming
__all__ = ["compute_score", "f1_attribution_reward"]
