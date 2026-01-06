"""
Fast Repetition Penalty for RLVR/GRPO Reward Functions

Simple, fast detector optimized for training loops.
Typical latency: <3ms even for very long texts.

Key signals:
1. Compression ratio (gzip) - catches all redundancy patterns
2. N-gram diversity - catches local phrase repetition
3. Word frequency - catches vocabulary poverty

Usage:
    from verl.utils.reward.repetition_penalty import (
        is_repetitive,
        repetition_score,
        apply_repetition_penalty,
    )

    # Quick boolean check
    if is_repetitive(text):
        reward = 0.0

    # Get score for custom handling
    score = repetition_score(text)  # 0.0 = clean, 1.0 = severe repetition

    # Apply penalty to reward
    penalized_reward = apply_repetition_penalty(text, base_reward)
"""

from __future__ import annotations

import gzip
from collections import Counter
from typing import List, Tuple, Union


def _compression_ratio(text: str) -> float:
    """Gzip compression ratio. Lower = more repetitive."""
    if len(text) < 20:
        return 1.0
    encoded = text.encode('utf-8')
    compressed = gzip.compress(encoded, compresslevel=1)
    return len(compressed) / len(encoded)


def _distinct_ngrams(text: str, n: int) -> float:
    """Ratio of unique n-grams to total n-grams."""
    words = text.lower().split()
    if len(words) < n:
        return 1.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def _max_word_frequency(text: str) -> float:
    """Frequency of most common content word."""
    stopwords = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'and', 'or', 'but', 'if', 'then',
        'else', 'when', 'where', 'this', 'that', 'these', 'those', 'it', 'its',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
    })
    words = [w.lower().strip('.,!?;:()[]{}"\'-') for w in text.split()]
    content_words = [w for w in words if w and w not in stopwords and len(w) > 2]
    if not content_words:
        return 0.0
    counter = Counter(content_words)
    return counter.most_common(1)[0][1] / len(content_words)


def repetition_score(text: str, min_length: int = 50) -> float:
    """
    Compute repetition score for text.

    Args:
        text: Input text to analyze
        min_length: Minimum text length to analyze (shorter returns 0.0)

    Returns:
        Score in [0.0, 1.0] where 0.0 = no repetition, 1.0 = severe repetition

    Performance: ~2-3ms for 20k character texts
    """
    text = text.strip()
    if len(text) < min_length:
        return 0.0

    # Signal 1: Compression ratio (most reliable)
    comp = _compression_ratio(text)
    # Good text: 0.3-0.6, Repetitive: <0.15
    comp_score = max(0.0, 1.0 - (comp - 0.08) / 0.35) if comp < 0.43 else 0.0

    # Signal 2: N-gram diversity
    dist3 = _distinct_ngrams(text, 3)
    dist5 = _distinct_ngrams(text, 5)
    min_dist = min(dist3, dist5)
    # Good text: >0.8, Repetitive: <0.4
    ngram_score = max(0.0, 1.0 - (min_dist - 0.2) / 0.5) if min_dist < 0.7 else 0.0

    # Signal 3: Word frequency concentration
    max_freq = _max_word_frequency(text)
    # Good text: <0.1, Repetitive: >0.3
    freq_score = max(0.0, (max_freq - 0.08) / 0.25) if max_freq > 0.08 else 0.0

    # Combine: max of primary signals + frequency boost
    combined = max(comp_score, ngram_score * 0.9) + freq_score * 0.2
    return min(1.0, combined)


def is_repetitive(text: str, threshold: float = 0.35) -> bool:
    """
    Quick boolean check for repetitive text.

    Args:
        text: Input text
        threshold: Score threshold (default 0.35 catches most reward hacking)

    Returns:
        True if text appears to be repetitive/reward-hacking
    """
    return repetition_score(text) > threshold


def apply_repetition_penalty(
    text: str,
    base_reward: float,
    severity: str = 'moderate',
) -> float:
    """
    Apply repetition penalty to a reward value.

    Args:
        text: Model output text
        base_reward: Original reward value
        severity: 'lenient', 'moderate', or 'strict'

    Returns:
        Penalized reward value

    Penalty curves:
        lenient:  <0.4 → 100%, <0.6 → 80%, <0.8 → 50%, else → 20%
        moderate: <0.3 → 100%, <0.5 → 75%, <0.7 → 40%, else → 10%
        strict:   <0.25 → 100%, <0.4 → 70%, <0.6 → 30%, else → 5%
    """
    score = repetition_score(text)

    if severity == 'lenient':
        if score < 0.4: return base_reward
        elif score < 0.6: return base_reward * 0.8
        elif score < 0.8: return base_reward * 0.5
        else: return base_reward * 0.2
    elif severity == 'strict':
        if score < 0.25: return base_reward
        elif score < 0.4: return base_reward * 0.7
        elif score < 0.6: return base_reward * 0.3
        else: return base_reward * 0.05
    else:  # moderate (default)
        if score < 0.3: return base_reward
        elif score < 0.5: return base_reward * 0.75
        elif score < 0.7: return base_reward * 0.4
        else: return base_reward * 0.1


def apply_repetition_penalty_batch(
    texts: List[str],
    rewards: List[float],
    severity: str = 'moderate',
) -> List[float]:
    """
    Apply repetition penalty to a batch of rewards.

    Args:
        texts: List of model output texts
        rewards: List of base reward values
        severity: 'lenient', 'moderate', or 'strict'

    Returns:
        List of penalized rewards
    """
    return [
        apply_repetition_penalty(text, reward, severity)
        for text, reward in zip(texts, rewards)
    ]


def get_repetition_scores_batch(texts: List[str]) -> List[float]:
    """
    Compute repetition scores for a batch of texts.

    Args:
        texts: List of texts to analyze

    Returns:
        List of repetition scores in [0.0, 1.0]
    """
    return [repetition_score(text) for text in texts]
