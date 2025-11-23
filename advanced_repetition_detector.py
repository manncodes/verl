#!/usr/bin/env python3
"""
Advanced Repetition Detection for Reward Functions - Drop-in Replacement

CALIBRATED VERSION - Optimized for your use case:
- Keeps scores intact for fine/borderline cases
- Only penalizes clear, egregious repetition
- Focuses on truly inhumane/robotic responses

USAGE (same as before - just replace the file):
    from advanced_repetition_detector import apply_repetition_penalty

    final_reward = apply_repetition_penalty(
        text=response['content'],
        base_reward=base_reward,
        severity="moderate"  # Options: lenient, moderate, strict, very_strict
    )
"""

import re
from typing import Dict, List
from collections import Counter
import statistics

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Detection Functions (Calibrated)
# ============================================================================

def detect_repetition_penalty(text: str) -> Dict[str, float]:
    """Detect word/phrase/paragraph repetition (calibrated version)."""
    if not text.strip():
        return {'word_rep': 0, 'phrase_rep': 0, 'para_rep': 0}

    words = text.lower().split()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Word-level (calibrated - only egregious cases)
    if len(words) >= 20:
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'it', 'that', 'this', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
            'how', 'why', 'if', 'because', 'as', 'by', 'from', 'not', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must', 'do', 'does', 'did'
        }

        content_words = [w for w in words if w not in stopwords and len(w) > 2]

        if len(content_words) >= 10:
            word_counts = Counter(content_words)
            # Only penalize VERY excessive (4+ times)
            total_excess = sum(max(0, count - 3) for count in word_counts.values())
            word_rep_score = min(total_excess / len(content_words) * 4, 1.0)
            word_rep_score = word_rep_score if word_rep_score > 0.25 else 0.0
        else:
            word_rep_score = 0.0
    else:
        word_rep_score = 0.0

    # Phrase-level (calibrated - 4-grams, more specific)
    if len(words) >= 12:
        ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if len(ngrams) >= 8:
            ngram_counts = Counter(ngrams)
            total_excess = sum(max(0, count - 2) for count in ngram_counts.values())
            phrase_rep_score = min(total_excess / len(ngrams) * 5, 1.0)
            phrase_rep_score = phrase_rep_score if phrase_rep_score > 0.3 else 0.0
        else:
            phrase_rep_score = 0.0
    else:
        phrase_rep_score = 0.0

    # Line-level (calibrated - exact duplicates only)
    if len(lines) >= 4:
        line_counts = Counter([l for l in lines if len(l.split()) > 2])
        total_excess = sum(max(0, count - 2) for count in line_counts.values())
        para_rep_score = min(total_excess / max(len(lines), 1) * 4, 1.0)
        para_rep_score = para_rep_score if para_rep_score > 0.4 else 0.0
    else:
        para_rep_score = 0.0

    return {
        'word_rep': word_rep_score,
        'phrase_rep': phrase_rep_score,
        'para_rep': para_rep_score
    }


def detect_template_patterns(lines: List[str]) -> float:
    """Detect template abuse (calibrated)."""
    if len(lines) < 5:
        return 0.0

    substantial_lines = [l for l in lines if len(l.split()) >= 4]
    if len(substantial_lines) < 5:
        return 0.0

    # Check for rigid templates
    patterns = []
    for line in substantial_lines:
        # Look for "X is/are/was/were WORD because/that" pattern
        pattern_match = re.search(
            r'^\w+\s+(is|are|was|were|has|have)\s+\w+\s+(because|that|which|since)',
            line.lower()
        )
        if pattern_match:
            patterns.append(pattern_match.group(0))
        else:
            words = line.split()
            if len(words) >= 2:
                patterns.append(' '.join(words[:2]).lower())

    if len(patterns) < 5:
        return 0.0

    pattern_counts = Counter(patterns)
    total_excess = sum(max(0, count - 3) for count in pattern_counts.values())

    if total_excess == 0:
        return 0.0

    score = min(total_excess / len(patterns) * 3, 1.0)
    return score if score > 0.35 else 0.0


def detect_filler_content(text: str) -> float:
    """Detect filler content (calibrated)."""
    critical_fillers = [
        "it is worth noting that",
        "it should be mentioned",
        "as mentioned before",
        "as previously stated",
        "as stated earlier",
        "to reiterate",
        "to repeat",
        "once again",
    ]

    text_lower = text.lower()
    count = sum(text_lower.count(phrase) for phrase in critical_fillers)

    if count < 2:
        return 0.0

    words = len(text.split())
    if words < 30:
        return 0.0

    density = count / (words / 50)
    score = min(density / 4, 1.0)
    return score if score > 0.3 else 0.0


def detect_length_gaming(text: str) -> float:
    """Detect length gaming (calibrated)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 8:
        return 0.0

    penalties = []

    # Very uniform line length
    if len(lines) >= 10:
        line_lengths = [len(line.split()) for line in lines]

        if HAS_NUMPY:
            std_dev = np.std(line_lengths)
            mean_length = np.mean(line_lengths)
        else:
            mean_length = statistics.mean(line_lengths)
            std_dev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0

        if mean_length > 4:
            cv = std_dev / mean_length if mean_length > 0 else 0
            if cv < 0.12:
                penalties.append(0.8)
            elif cv < 0.18:
                penalties.append(0.5)

    # Quality degradation
    if len(lines) > 15:
        mid = len(lines) // 2
        first_unique = len(set(lines[:mid]))
        second_unique = len(set(lines[mid:]))

        first_diversity = first_unique / mid
        second_diversity = second_unique / (len(lines) - mid)

        if first_diversity > 0.6 and second_diversity < 0.25:
            penalties.append(0.9)

    # Extreme list padding
    list_count = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+\S{1,5}\s*$', line))
    if list_count > len(lines) * 0.65:
        penalties.append(0.7)

    return max(penalties) if penalties else 0.0


def detect_circular_reasoning(text: str) -> float:
    """Detect circular reasoning (calibrated)."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) > 5]

    if len(sentences) < 5:
        return 0.0

    similar_count = 0
    for i in range(len(sentences)):
        words_i = set(sentences[i].lower().split())
        if len(words_i) < 4:
            continue

        for j in range(i + 1, min(i + 3, len(sentences))):
            words_j = set(sentences[j].lower().split())
            if len(words_j) < 4:
                continue

            overlap = len(words_i & words_j) / len(words_i | words_j)
            if overlap > 0.65:
                similar_count += 1

    if similar_count < 2:
        return 0.0

    score = min(similar_count / len(sentences) * 4, 1.0)
    return score if score > 0.25 else 0.0


def detect_keyword_stuffing(text: str) -> float:
    """Detect keyword stuffing (calibrated)."""
    words = text.lower().split()
    if len(words) < 25:
        return 0.0

    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'are', 'was', 'were', 'it', 'that', 'this'
    }

    content_words = [w for w in words if w not in stopwords and len(w) > 3]

    if len(content_words) < 12:
        return 0.0

    word_counts = Counter(content_words)
    stuffing_score = 0

    for word, count in word_counts.most_common(3):
        frequency = count / len(content_words)
        if frequency > 0.12:
            stuffing_score += (frequency - 0.12) * 25

    score = min(stuffing_score, 1.0)
    return score if score > 0.4 else 0.0


def detect_padding_sentences(text: str) -> float:
    """Detect padding sentences (calibrated)."""
    obvious_padding = [
        r"^as (we|you|i) can (clearly )?see($|,)",
        r"^it is (very )?(clear|obvious|evident|apparent) that",
        r"^in (conclusion|summary)($|,)",
    ]

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    if len(sentences) < 5:
        return 0.0

    padding_count = 0
    for sentence in sentences:
        if len(sentence.split()) < 4:
            continue
        for pattern in obvious_padding:
            if re.search(pattern, sentence.lower()):
                padding_count += 1
                break

    if padding_count < 2:
        return 0.0

    score = min(padding_count / len(sentences) * 3, 1.0)
    return score if score > 0.4 else 0.0


# ============================================================================
# Main Detection Function
# ============================================================================

def advanced_repetition_detector(text: str) -> Dict[str, float]:
    """
    Advanced repetition detection (CALIBRATED VERSION).

    Same interface as before, but with calibrated thresholds.
    Only penalizes clear, egregious repetition.
    """
    if not text or len(text.strip()) < 30:
        return {'overall_score': 0.0}

    # Base detection
    penalties = detect_repetition_penalty(text)

    # Get lines
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Advanced checks (all calibrated)
    penalties['template_score'] = detect_template_patterns(lines)
    penalties['filler_score'] = detect_filler_content(text)
    penalties['length_gaming'] = detect_length_gaming(text)
    penalties['circular_reasoning'] = detect_circular_reasoning(text)
    penalties['keyword_stuffing'] = detect_keyword_stuffing(text)
    penalties['padding_sentences'] = detect_padding_sentences(text)

    # Calibrated weights (focus on most reliable signals)
    overall_score = (
        penalties['word_rep'] * 0.12 +
        penalties['phrase_rep'] * 0.18 +
        penalties['para_rep'] * 0.28 +        # Most reliable
        penalties['template_score'] * 0.15 +
        penalties['filler_score'] * 0.10 +
        penalties['length_gaming'] * 0.18 +   # Clear signal
        penalties['circular_reasoning'] * 0.10 +
        penalties['keyword_stuffing'] * 0.08 +
        penalties['padding_sentences'] * 0.03
    ) / 1.22

    penalties['overall_score'] = min(overall_score, 1.0)

    return penalties


# ============================================================================
# Reward Integration (Same Interface)
# ============================================================================

def get_penalty_multiplier(repetition_score: float,
                          severity: str = "moderate",
                          min_multiplier: float = None) -> float:
    """
    Get penalty multiplier (CALIBRATED VERSION).

    Same interface, but with calibrated thresholds for better precision.

    Severity presets (recalibrated):
        - lenient: Very forgiving (< 0.40 no penalty)
        - moderate: Balanced (< 0.30 no penalty) - RECOMMENDED
        - strict: Quality focused (< 0.20 no penalty)
        - very_strict: Maximum strictness (< 0.15 no penalty)
    """
    presets = {
        "lenient": {
            "thresholds": [0.40, 0.55, 0.70, 0.85],
            "multipliers": [1.0, 0.90, 0.70, 0.40, 0.15],
        },
        "moderate": {
            "thresholds": [0.30, 0.45, 0.60, 0.75],
            "multipliers": [1.0, 0.85, 0.65, 0.40, 0.15],
        },
        "strict": {
            "thresholds": [0.20, 0.35, 0.50, 0.70],
            "multipliers": [1.0, 0.80, 0.55, 0.25, 0.05],
        },
        "very_strict": {
            "thresholds": [0.15, 0.30, 0.45, 0.65],
            "multipliers": [1.0, 0.70, 0.45, 0.15, 0.0],
        },
    }

    if severity not in presets:
        severity = "moderate"

    thresholds = presets[severity]["thresholds"]
    multipliers = presets[severity]["multipliers"]

    if min_multiplier is not None:
        multipliers[-1] = max(min_multiplier, 0.0)

    for i, threshold in enumerate(thresholds):
        if repetition_score < threshold:
            return multipliers[i]

    return multipliers[-1]


def apply_repetition_penalty(text: str,
                             base_reward: float,
                             severity: str = "moderate",
                             return_details: bool = False):
    """
    Apply repetition penalty to reward (DROP-IN REPLACEMENT - CALIBRATED).

    SAME INTERFACE as before, but with CALIBRATED detection.

    Changes:
    - Higher precision (fewer false positives)
    - Only penalizes clear, egregious cases
    - Keeps scores intact for borderline/acceptable text
    - Focuses on truly inhumane/robotic responses

    Args:
        text: Model output text
        base_reward: Original reward before penalty
        severity: "lenient", "moderate" (recommended), "strict", "very_strict"
        return_details: If True, return (penalized_reward, details_dict)

    Returns:
        Penalized reward (or tuple if return_details=True)

    Calibrated Thresholds (moderate):
        < 0.30: No penalty (acceptable)
        0.30-0.45: 15% penalty (minor issues)
        0.45-0.60: 35% penalty (clear issues)
        0.60-0.75: 60% penalty (serious issues)
        > 0.75: 85% penalty (egregious)
    """
    result = advanced_repetition_detector(text)
    rep_score = result['overall_score']

    multiplier = get_penalty_multiplier(rep_score, severity)
    penalized_reward = base_reward * multiplier

    if return_details:
        details = {
            'repetition_score': rep_score,
            'penalty_multiplier': multiplier,
            'base_reward': base_reward,
            'penalized_reward': penalized_reward,
            'penalty_applied': base_reward - penalized_reward,
            'components': result,
        }
        return penalized_reward, details

    return penalized_reward


# ============================================================================
# Utility Functions
# ============================================================================

def get_quality_assessment(score: float) -> str:
    """Get quality assessment based on calibrated thresholds."""
    if score < 0.30:
        return "✓ ACCEPTABLE"
    elif score < 0.45:
        return "⚠ MINOR_ISSUES"
    elif score < 0.60:
        return "⚠⚠ CLEAR_ISSUES"
    elif score < 0.75:
        return "✗ SERIOUS_ISSUES"
    else:
        return "✗✗ EGREGIOUS"


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("CALIBRATED REPETITION DETECTION - Same Interface, Better Performance")
    print("="*75)
    print()

    test_cases = {
        "Good quality": """
            Machine learning enables systems to learn from data patterns.
            Neural networks process information through interconnected layers.
            Training involves optimizing parameters using backpropagation.
        """,

        "Minor repetition": """
            Machine learning helps predictions. Machine learning improves over time.
            We use machine learning for tasks. Machine learning is effective.
        """,

        "Egregious repetition": """
            The model the model the model the model works works works.
            The model the model the model the model works works works.
            The model the model the model the model works works works.
            The model the model the model the model works works works.
        """,
    }

    for name, text in test_cases.items():
        print(f"{name}:")
        print("-"*75)

        reward, details = apply_repetition_penalty(
            text, 1.0,
            severity="moderate",
            return_details=True
        )

        print(f"  Score: {details['repetition_score']:.3f}")
        print(f"  Multiplier: {details['penalty_multiplier']:.3f}")
        print(f"  Final reward: {reward:.3f}")
        print(f"  Assessment: {get_quality_assessment(details['repetition_score'])}")
        print()

    print("="*75)
    print("✓ Drop-in replacement ready!")
    print("✓ Same function names and signatures")
    print("✓ Calibrated for higher precision")
    print("✓ Only penalizes clear, egregious cases")
    print("="*75)
