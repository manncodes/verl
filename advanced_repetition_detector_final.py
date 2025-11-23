#!/usr/bin/env python3
"""
Advanced Repetition Detection - FINAL BALANCED VERSION

Tested on 20 real LLM generations.
Balanced to catch problematic cases while avoiding false positives.

SAME INTERFACE - just better calibrated based on real data.
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


# Same detection functions as before, but with adjusted final scoring
def detect_repetition_penalty(text: str) -> Dict[str, float]:
    """Detect word/phrase/paragraph repetition."""
    if not text.strip():
        return {'word_rep': 0, 'phrase_rep': 0, 'para_rep': 0}

    words = text.lower().split()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Word-level
    if len(words) >= 15:  # Lowered from 20
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'it', 'that', 'this', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
            'how', 'why', 'if', 'because', 'as', 'by', 'from', 'not', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must', 'do', 'does', 'did'
        }

        content_words = [w for w in words if w not in stopwords and len(w) > 2]

        if len(content_words) >= 8:  # Lowered from 10
            word_counts = Counter(content_words)
            # Catch excessive repetition (3+ times)
            total_excess = sum(max(0, count - 2) for count in word_counts.values())  # Changed from 3
            word_rep_score = min(total_excess / len(content_words) * 3.5, 1.0)  # Increased multiplier
            word_rep_score = word_rep_score if word_rep_score > 0.20 else 0.0  # Lowered from 0.25
        else:
            word_rep_score = 0.0
    else:
        word_rep_score = 0.0

    # Phrase-level
    if len(words) >= 10:  # Lowered from 12
        ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if len(ngrams) >= 6:  # Lowered from 8
            ngram_counts = Counter(ngrams)
            total_excess = sum(max(0, count - 2) for count in ngram_counts.values())
            phrase_rep_score = min(total_excess / len(ngrams) * 4.5, 1.0)  # Increased from 5
            phrase_rep_score = phrase_rep_score if phrase_rep_score > 0.25 else 0.0  # Lowered from 0.3
        else:
            phrase_rep_score = 0.0
    else:
        phrase_rep_score = 0.0

    # Line-level
    if len(lines) >= 3:  # Lowered from 4
        line_counts = Counter([l for l in lines if len(l.split()) > 2])
        total_excess = sum(max(0, count - 2) for count in line_counts.values())
        para_rep_score = min(total_excess / max(len(lines), 1) * 3.5, 1.0)  # Decreased from 4
        para_rep_score = para_rep_score if para_rep_score > 0.35 else 0.0  # Lowered from 0.4
    else:
        para_rep_score = 0.0

    return {
        'word_rep': word_rep_score,
        'phrase_rep': phrase_rep_score,
        'para_rep': para_rep_score
    }


def detect_template_patterns(lines: List[str]) -> float:
    """Detect template abuse."""
    if len(lines) < 4:  # Lowered from 5
        return 0.0

    substantial_lines = [l for l in lines if len(l.split()) >= 4]
    if len(substantial_lines) < 4:  # Lowered from 5
        return 0.0

    patterns = []
    for line in substantial_lines:
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

    if len(patterns) < 4:  # Lowered from 5
        return 0.0

    pattern_counts = Counter(patterns)
    total_excess = sum(max(0, count - 2) for count in pattern_counts.values())  # Lowered from 3

    if total_excess == 0:
        return 0.0

    score = min(total_excess / len(patterns) * 2.5, 1.0)  # Decreased from 3
    return score if score > 0.30 else 0.0  # Lowered from 0.35


def detect_filler_content(text: str) -> float:
    """Detect filler content."""
    critical_fillers = [
        "it is worth noting that",
        "it should be mentioned",
        "as mentioned before",
        "as previously stated",
        "as stated earlier",
        "to reiterate",
        "to repeat",
        "once again",
        "furthermore",
        "additionally",
        "moreover",
    ]

    text_lower = text.lower()
    count = sum(text_lower.count(phrase) for phrase in critical_fillers)

    if count < 2:
        return 0.0

    words = len(text.split())
    if words < 25:  # Lowered from 30
        return 0.0

    density = count / (words / 40)  # Changed from 50
    score = min(density / 3.5, 1.0)  # Changed from 4
    return score if score > 0.25 else 0.0  # Lowered from 0.3


def detect_length_gaming(text: str) -> float:
    """Detect length gaming."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 7:  # Lowered from 8
        return 0.0

    penalties = []

    if len(lines) >= 8:  # Lowered from 10
        line_lengths = [len(line.split()) for line in lines]

        if HAS_NUMPY:
            std_dev = np.std(line_lengths)
            mean_length = np.mean(line_lengths)
        else:
            mean_length = statistics.mean(line_lengths)
            std_dev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0

        if mean_length > 4:
            cv = std_dev / mean_length if mean_length > 0 else 0
            if cv < 0.15:  # Increased from 0.12
                penalties.append(0.7)  # Decreased from 0.8
            elif cv < 0.20:  # Increased from 0.18
                penalties.append(0.4)  # Decreased from 0.5

    if len(lines) > 12:  # Lowered from 15
        mid = len(lines) // 2
        first_unique = len(set(lines[:mid]))
        second_unique = len(set(lines[mid:]))

        first_diversity = first_unique / mid
        second_diversity = second_unique / (len(lines) - mid)

        if first_diversity > 0.5 and second_diversity < 0.3:  # Changed from 0.6 and 0.25
            penalties.append(0.8)  # Decreased from 0.9

    list_count = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+\S{1,5}\s*$', line))
    if list_count > len(lines) * 0.60:  # Lowered from 0.65
        penalties.append(0.6)  # Decreased from 0.7

    return max(penalties) if penalties else 0.0


def detect_circular_reasoning(text: str) -> float:
    """Detect circular reasoning."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) > 4]  # Lowered from 5

    if len(sentences) < 4:  # Lowered from 5
        return 0.0

    similar_count = 0
    for i in range(len(sentences)):
        words_i = set(sentences[i].lower().split())
        if len(words_i) < 3:  # Lowered from 4
            continue

        for j in range(i + 1, min(i + 3, len(sentences))):
            words_j = set(sentences[j].lower().split())
            if len(words_j) < 3:  # Lowered from 4
                continue

            overlap = len(words_i & words_j) / len(words_i | words_j)
            if overlap > 0.60:  # Lowered from 0.65
                similar_count += 1

    if similar_count < 2:
        return 0.0

    score = min(similar_count / len(sentences) * 3.5, 1.0)  # Decreased from 4
    return score if score > 0.20 else 0.0  # Lowered from 0.25


def detect_keyword_stuffing(text: str) -> float:
    """Detect keyword stuffing."""
    words = text.lower().split()
    if len(words) < 20:  # Lowered from 25
        return 0.0

    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'are', 'was', 'were', 'it', 'that', 'this'
    }

    content_words = [w for w in words if w not in stopwords and len(w) > 3]

    if len(content_words) < 10:  # Lowered from 12
        return 0.0

    word_counts = Counter(content_words)
    stuffing_score = 0

    for word, count in word_counts.most_common(3):
        frequency = count / len(content_words)
        if frequency > 0.10:  # Lowered from 0.12
            stuffing_score += (frequency - 0.10) * 20  # Decreased from 25

    score = min(stuffing_score, 1.0)
    return score if score > 0.35 else 0.0  # Lowered from 0.4


def detect_padding_sentences(text: str) -> float:
    """Detect padding sentences."""
    obvious_padding = [
        r"^as (we|you|i) can (clearly )?see($|,)",
        r"^it is (very )?(clear|obvious|evident|apparent) that",
        r"^in (conclusion|summary)($|,)",
    ]

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    if len(sentences) < 4:  # Lowered from 5
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

    score = min(padding_count / len(sentences) * 2.5, 1.0)  # Decreased from 3
    return score if score > 0.35 else 0.0  # Lowered from 0.4


def advanced_repetition_detector(text: str) -> Dict[str, float]:
    """Advanced repetition detection - BALANCED VERSION."""
    if not text or len(text.strip()) < 20:  # Lowered from 30
        return {'overall_score': 0.0}

    penalties = detect_repetition_penalty(text)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    penalties['template_score'] = detect_template_patterns(lines)
    penalties['filler_score'] = detect_filler_content(text)
    penalties['length_gaming'] = detect_length_gaming(text)
    penalties['circular_reasoning'] = detect_circular_reasoning(text)
    penalties['keyword_stuffing'] = detect_keyword_stuffing(text)
    penalties['padding_sentences'] = detect_padding_sentences(text)

    # ADJUSTED WEIGHTS - better balance
    overall_score = (
        penalties['word_rep'] * 0.14 +           # Increased from 0.12
        penalties['phrase_rep'] * 0.20 +         # Same
        penalties['para_rep'] * 0.25 +           # Same (most reliable)
        penalties['template_score'] * 0.16 +     # Increased from 0.15
        penalties['filler_score'] * 0.12 +       # Increased from 0.10
        penalties['length_gaming'] * 0.18 +      # Same
        penalties['circular_reasoning'] * 0.12 + # Increased from 0.10
        penalties['keyword_stuffing'] * 0.10 +   # Increased from 0.08
        penalties['padding_sentences'] * 0.05    # Increased from 0.03
    ) / 1.32  # Adjusted normalizer (from 1.22)

    penalties['overall_score'] = min(overall_score, 1.0)

    return penalties


def get_penalty_multiplier(repetition_score: float,
                          severity: str = "moderate",
                          min_multiplier: float = None) -> float:
    """Get penalty multiplier - SAME INTERFACE."""
    presets = {
        "lenient": {
            "thresholds": [0.35, 0.50, 0.65, 0.80],
            "multipliers": [1.0, 0.90, 0.70, 0.45, 0.20],
        },
        "moderate": {
            "thresholds": [0.25, 0.40, 0.55, 0.70],  # Adjusted
            "multipliers": [1.0, 0.85, 0.65, 0.40, 0.15],
        },
        "strict": {
            "thresholds": [0.15, 0.30, 0.45, 0.65],
            "multipliers": [1.0, 0.80, 0.55, 0.30, 0.05],
        },
        "very_strict": {
            "thresholds": [0.10, 0.20, 0.35, 0.55],
            "multipliers": [1.0, 0.70, 0.45, 0.20, 0.0],
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
    """Apply repetition penalty - SAME INTERFACE, BETTER BALANCED."""
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


def get_quality_assessment(score: float) -> str:
    """Get quality assessment."""
    if score < 0.25:
        return "✓ ACCEPTABLE"
    elif score < 0.40:
        return "⚠ MINOR_ISSUES"
    elif score < 0.55:
        return "⚠⚠ CLEAR_ISSUES"
    elif score < 0.70:
        return "✗ SERIOUS_ISSUES"
    else:
        return "✗✗ EGREGIOUS"


if __name__ == "__main__":
    print("FINAL BALANCED VERSION - Tested on Real LLM Outputs")
    print("="*75)
    print()

    test_cases = {
        "Good quality": "Machine learning enables systems to learn from data.",
        "Repetitive": "Machine learning machine learning machine learning works works works.",
        "Copy-paste": "Line one.\nLine one.\nLine one.\nLine one.\nLine one.",
    }

    for name, text in test_cases.items():
        reward, details = apply_repetition_penalty(text, 1.0, severity="moderate", return_details=True)
        print(f"{name}: score={details['repetition_score']:.3f}, penalty={details['penalty_applied']*100:.0f}%, final={reward:.3f}")
