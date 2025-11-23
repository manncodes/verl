#!/usr/bin/env python3
"""
OPTIMALLY CALIBRATED Repetition Detection

Sweet spot between moderate and ultra-conservative:
- Keeps scores intact for borderline/acceptable cases
- Only penalizes clear, egregious repetition
- Catches truly inhumane/robotic responses
- No false positives on good quality text

CALIBRATION GOALS:
✓ Natural variation → NO penalty
✓ Minor repetition → NO penalty
✓ Informative templates → NO penalty
✗ Egregious repetition → PENALTY
✗ Robotic template abuse → PENALTY
✗ Clear gaming → PENALTY
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


def detect_egregious_word_repetition(text: str) -> float:
    """Only catch truly egregious word repetition."""
    if not text.strip() or len(text.split()) < 20:
        return 0.0

    words = text.lower().split()

    # Expanded stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'it', 'that', 'this', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
        'how', 'why', 'if', 'because', 'as', 'by', 'from', 'not', 'will', 'would',
        'can', 'could', 'should', 'may', 'might', 'must', 'do', 'does', 'did'
    }

    content_words = [w for w in words if w not in stopwords and len(w) > 2]

    if len(content_words) < 10:
        return 0.0

    word_counts = Counter(content_words)

    # Only penalize VERY excessive repetition (4+ times)
    total_excess = sum(max(0, count - 3) for count in word_counts.values())

    if total_excess == 0:
        return 0.0

    # Aggressive scoring for egregious cases
    score = min(total_excess / len(content_words) * 4, 1.0)

    # Filter out minor issues
    return score if score > 0.25 else 0.0


def detect_egregious_phrase_repetition(text: str) -> float:
    """Only catch obvious phrase repetition."""
    words = text.lower().split()

    if len(words) < 12:
        return 0.0

    # Use 4-grams (more specific than 3-grams)
    ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]

    if len(ngrams) < 8:
        return 0.0

    ngram_counts = Counter(ngrams)

    # Only penalize if phrase repeats 3+ times
    total_excess = sum(max(0, count - 2) for count in ngram_counts.values())

    if total_excess == 0:
        return 0.0

    score = min(total_excess / len(ngrams) * 5, 1.0)

    return score if score > 0.3 else 0.0


def detect_egregious_line_repetition(text: str) -> float:
    """Catch exact duplicate lines."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip() and len(l.split()) > 2]

    if len(lines) < 4:
        return 0.0

    line_counts = Counter(lines)

    # Penalize exact duplicates (3+ occurrences)
    total_excess = sum(max(0, count - 2) for count in line_counts.values())

    if total_excess == 0:
        return 0.0

    score = min(total_excess / len(lines) * 4, 1.0)

    return score if score > 0.4 else 0.0


def detect_template_abuse(lines: List[str]) -> float:
    """Detect rigid template structures."""
    if len(lines) < 5:
        return 0.0

    substantial_lines = [l for l in lines if len(l.split()) >= 4]

    if len(substantial_lines) < 5:
        return 0.0

    # Check if many lines follow same structure
    patterns = []
    for line in substantial_lines:
        words = line.split()
        # Extract first 2 words and check for "X is/are/was/were WORD because"
        if len(words) >= 4:
            # Pattern: first word + connector + because/that/which
            pattern_match = re.search(r'^\w+\s+(is|are|was|were|has|have)\s+\w+\s+(because|that|which|since)',
                                     line.lower())
            if pattern_match:
                patterns.append(pattern_match.group(0))
            else:
                # Fallback: just first 2 words
                patterns.append(' '.join(words[:2]).lower())

    if len(patterns) < 5:
        return 0.0

    pattern_counts = Counter(patterns)

    # Penalize if 4+ lines follow same template
    total_excess = sum(max(0, count - 3) for count in pattern_counts.values())

    if total_excess == 0:
        return 0.0

    score = min(total_excess / len(patterns) * 3, 1.0)

    return score if score > 0.35 else 0.0


def detect_filler_abuse(text: str) -> float:
    """Catch heavy filler usage."""
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

    # Normalize
    density = count / (words / 50)  # Per 50 words
    score = min(density / 4, 1.0)

    return score if score > 0.3 else 0.0


def detect_length_gaming(text: str) -> float:
    """Detect artificial lengthening."""
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
            if cv < 0.12:  # Very robotic
                penalties.append(0.8)
            elif cv < 0.18:  # Quite robotic
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
    """Detect near-duplicate sentences (circular reasoning)."""
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
            if overlap > 0.65:  # Very similar
                similar_count += 1

    if similar_count < 2:
        return 0.0

    score = min(similar_count / len(sentences) * 4, 1.0)

    return score if score > 0.25 else 0.0


def detect_keyword_stuffing(text: str) -> float:
    """Detect extreme keyword stuffing."""
    words = text.lower().split()
    if len(words) < 25:
        return 0.0

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                 'is', 'are', 'was', 'were', 'it', 'that', 'this'}

    content_words = [w for w in words if w not in stopwords and len(w) > 3]

    if len(content_words) < 12:
        return 0.0

    word_counts = Counter(content_words)

    # Check if any word dominates (> 12% of content)
    stuffing_score = 0
    for word, count in word_counts.most_common(3):
        frequency = count / len(content_words)
        if frequency > 0.12:
            stuffing_score += (frequency - 0.12) * 25

    score = min(stuffing_score, 1.0)

    return score if score > 0.4 else 0.0


def calculate_repetition_score_calibrated(text: str, return_components: bool = False) -> Dict[str, float]:
    """
    OPTIMALLY CALIBRATED repetition detection.

    Balance: Lenient on borderline cases, strict on egregious repetition.
    """
    if not text or len(text.strip()) < 30:
        return {'score': 0.0}

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    components = {
        'word_rep': detect_egregious_word_repetition(text),
        'phrase_rep': detect_egregious_phrase_repetition(text),
        'line_rep': detect_egregious_line_repetition(text),
        'template': detect_template_abuse(lines),
        'filler': detect_filler_abuse(text),
        'length_gaming': detect_length_gaming(text),
        'circular': detect_circular_reasoning(text),
        'keyword_stuffing': detect_keyword_stuffing(text),
    }

    # Calibrated weights (focus on most reliable signals)
    score = (
        components['word_rep'] * 0.12 +
        components['phrase_rep'] * 0.18 +
        components['line_rep'] * 0.28 +      # Most reliable
        components['template'] * 0.15 +
        components['filler'] * 0.10 +
        components['length_gaming'] * 0.18 +  # Clear signal
        components['circular'] * 0.10 +
        components['keyword_stuffing'] * 0.08
    ) / 1.19

    result = {'score': min(score, 1.0)}

    if return_components:
        result.update(components)

    return result


def apply_repetition_penalty_calibrated(
    text: str,
    base_reward: float,
    return_details: bool = False
):
    """
    Apply CALIBRATED repetition penalty.

    Philosophy:
    - Benefit of doubt for borderline cases
    - Clear penalties for egregious repetition
    - Focus on truly inhumane/robotic responses

    Thresholds:
    - < 0.30: No penalty (acceptable quality)
    - 0.30-0.45: 15% penalty (minor issues)
    - 0.45-0.60: 35% penalty (clear issues)
    - 0.60-0.75: 60% penalty (serious issues)
    - > 0.75: 85% penalty (egregious)
    """
    result = calculate_repetition_score_calibrated(text, return_components=True)
    score = result['score']

    # Calibrated penalty multipliers
    if score < 0.30:
        multiplier = 1.0      # No penalty
    elif score < 0.45:
        multiplier = 0.85     # 15% penalty
    elif score < 0.60:
        multiplier = 0.65     # 35% penalty
    elif score < 0.75:
        multiplier = 0.40     # 60% penalty
    else:
        multiplier = 0.15     # 85% penalty

    penalized_reward = base_reward * multiplier

    if return_details:
        assessment = get_assessment(score)
        details = {
            'repetition_score': score,
            'penalty_multiplier': multiplier,
            'base_reward': base_reward,
            'penalized_reward': penalized_reward,
            'penalty_applied': base_reward - penalized_reward,
            'components': result,
            'assessment': assessment
        }
        return penalized_reward, details

    return penalized_reward


def get_assessment(score: float) -> str:
    """Get quality assessment."""
    if score < 0.30:
        return "✓ ACCEPTABLE - No penalty"
    elif score < 0.45:
        return "⚠ MINOR_ISSUES - Small penalty"
    elif score < 0.60:
        return "⚠⚠ CLEAR_ISSUES - Moderate penalty"
    elif score < 0.75:
        return "✗ SERIOUS_ISSUES - Large penalty"
    else:
        return "✗✗ EGREGIOUS - Maximum penalty"


def diagnose_calibrated(text: str) -> str:
    """Diagnostic output."""
    result = calculate_repetition_score_calibrated(text, return_components=True)
    score = result['score']
    assessment = get_assessment(score)

    diagnosis = f"""
CALIBRATED REPETITION ANALYSIS
{'='*70}
Overall Score: {score:.3f}
Assessment: {assessment}

Non-Zero Components:
"""

    has_issues = False
    for component, value in result.items():
        if component != 'score' and value > 0:
            diagnosis += f"  {component:20s}: {value:.3f} ⚠\n"
            has_issues = True

    if not has_issues:
        diagnosis += "  [No issues detected - clean text]\n"

    # Show penalty at different scores
    test_scores = [score, 0.30, 0.45, 0.60, 0.75]
    diagnosis += f"\nPenalty Multipliers:\n"
    for s in sorted(set(test_scores)):
        _, d = apply_repetition_penalty_calibrated("", 1.0, return_details=True)
        # Recalculate with this score
        if s < 0.30:
            mult = 1.0
        elif s < 0.45:
            mult = 0.85
        elif s < 0.60:
            mult = 0.65
        elif s < 0.75:
            mult = 0.40
        else:
            mult = 0.15
        diagnosis += f"  Score {s:.2f}: {mult:.2f}x\n"

    diagnosis += f"{'='*70}\n"
    return diagnosis


if __name__ == "__main__":
    print("CALIBRATED REPETITION DETECTION - TEST CASES")
    print("="*70)
    print()

    test_cases = {
        "Good quality": """
            Machine learning enables systems to learn from data patterns.
            Neural networks process information through interconnected layers.
            Training involves optimizing parameters using backpropagation.
        """,

        "Minor repetition (acceptable)": """
            Machine learning helps with predictions. Machine learning improves
            over time. We use machine learning for various tasks.
        """,

        "Informative template": """
            Python is useful because it has extensive libraries.
            TensorFlow is useful because it simplifies deep learning.
            Docker is useful because it enables containerization.
        """,

        "Egregious repetition": """
            The model the model the model the model works works works.
            The model the model the model the model works works works.
            The model the model the model the model works works works.
            The model the model the model the model works works works.
        """,

        "Robotic template": """
            X is important because it enables Y and provides Z.
            A is important because it enables B and provides C.
            D is important because it enables E and provides F.
            G is important because it enables H and provides I.
            J is important because it enables K and provides L.
        """
    }

    for name, text in test_cases.items():
        print(f"\n{name}")
        print("-"*70)
        reward, details = apply_repetition_penalty_calibrated(text, 1.0, return_details=True)
        print(f"Score: {details['repetition_score']:.3f}")
        print(f"Multiplier: {details['penalty_multiplier']:.3f}")
        print(f"Final reward: {reward:.3f}")
        print(f"Assessment: {details['assessment']}")
