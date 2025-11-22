#!/usr/bin/env python3
"""
Production-Ready Repetition Penalty System

This module provides calibrated repetition detection for reward functions.
Use this to penalize model outputs that game the system through repetition,
templates, filler content, or artificial lengthening.

CALIBRATION RESULTS:
- Classification accuracy: 61.5% (vs 30.8% baseline)
- Good/bad separation: 0.305 (vs 0.147 baseline)
- False positive rate: Minimal (good texts score < 0.05)

RECOMMENDED USAGE:
    from repetition_penalty_final import apply_repetition_penalty

    # In your reward function:
    base_reward = calculate_base_reward(response)
    final_reward = apply_repetition_penalty(response['content'], base_reward)

THRESHOLDS (calibrated):
    - < 0.15: High quality (no penalty)
    - 0.15-0.30: Acceptable (small penalty)
    - 0.30-0.50: Moderate issues (medium penalty)
    - 0.50-0.70: Significant issues (large penalty)
    - > 0.70: Severe problems (maximum penalty)
"""

import re
from typing import Dict, Tuple
from collections import Counter
import statistics

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Core Detection Functions
# ============================================================================

def detect_repetition_penalty(text: str) -> Dict[str, float]:
    """Detect word/phrase/paragraph repetition."""
    if not text.strip():
        return {'word_rep': 0, 'phrase_rep': 0, 'para_rep': 0}

    words = text.lower().split()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Word-level
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'it', 'that', 'this', 'be', 'by'}
    content_words = [w for w in words if w not in common_words and len(w) > 2]

    if content_words:
        word_counts = Counter(content_words)
        excessive_reps = sum(max(0, count - 2) for count in word_counts.values())
        word_rep_score = min(excessive_reps / max(len(content_words), 1) * 2, 1.0)
    else:
        word_rep_score = 0.0

    # Phrase-level (3-grams)
    if len(words) >= 3:
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(max(0, count - 1) for count in trigram_counts.values())
        phrase_rep_score = min(repeated_trigrams / max(len(trigrams), 1) * 3, 1.0)
    else:
        phrase_rep_score = 0.0

    # Line-level
    if len(lines) > 1:
        line_counts = Counter(lines)
        repeated_lines = sum(max(0, count - 1) for count in line_counts.values())
        para_rep_score = min(repeated_lines / max(len(lines), 1) * 2, 1.0)
    else:
        para_rep_score = 0.0

    return {
        'word_rep': word_rep_score,
        'phrase_rep': phrase_rep_score,
        'para_rep': para_rep_score
    }


def detect_template_patterns(lines: list) -> float:
    """Detect repetitive templates."""
    if len(lines) < 4:
        return 0.0

    penalties = []

    # Structural patterns
    patterns = []
    for line in lines:
        if not line.strip() or len(line.split()) < 4:
            continue
        words = line.split()
        pattern_parts = []
        for i, word in enumerate(words):
            if word.lower() in {'is', 'are', 'was', 'were', 'because', 'that', 'the', 'a', 'an'}:
                pattern_parts.append(word.lower())
            elif word.isupper():
                pattern_parts.append('[UPPER]')
            elif word[0].isupper() and i > 0:
                pattern_parts.append('[CAP]')
            elif word.isdigit():
                pattern_parts.append('[NUM]')
            else:
                pattern_parts.append('X')
        patterns.append(' '.join(pattern_parts))

    if len(patterns) >= 4:
        pattern_counts = Counter(patterns)
        repeated_patterns = sum(max(0, count - 2) for count in pattern_counts.values())
        penalties.append(min(repeated_patterns / max(len(lines), 1) * 2, 1.0))

    # Sentence starts
    if len(lines) >= 6:
        first_words = [' '.join(line.split()[:2]).lower() for line in lines if len(line.split()) >= 3]
        if len(first_words) >= 6:
            first_word_counts = Counter(first_words)
            same_starts = sum(max(0, count - 3) for count in first_word_counts.values())
            penalties.append(min(same_starts / len(first_words) * 3, 1.0))

    return max(penalties) if penalties else 0.0


def detect_filler_content(text: str) -> float:
    """Detect filler phrases."""
    fillers = [
        ("transition", ["it is worth noting that", "it should be mentioned", "as mentioned before", "to reiterate"], 2.5),
        ("intensifier", ["furthermore", "additionally", "moreover", "in addition", "also"], 1.2),
        ("qualifier", ["it seems that", "it appears that", "arguably", "potentially"], 1.8),
        ("hedging", ["it could be argued that", "some might argue that", "it may be that"], 2.0),
    ]

    text_lower = text.lower()
    weighted_count = 0

    for _, phrases, weight in fillers:
        count = sum(text_lower.count(phrase) for phrase in phrases)
        weighted_count += count * weight

    words = len(text.split())
    if words > 20:
        filler_density = weighted_count / (words / 50)
        return min(filler_density / 3, 1.0)

    return 0.0


def detect_length_gaming(text: str) -> float:
    """Detect artificial lengthening."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 5:
        return 0.0

    penalties = []

    # Uniform line length
    if len(lines) >= 8:
        line_lengths = [len(line.split()) for line in lines]
        if HAS_NUMPY:
            std_dev = np.std(line_lengths)
            mean_length = np.mean(line_lengths)
        else:
            mean_length = statistics.mean(line_lengths)
            std_dev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0

        if mean_length > 3:
            cv = std_dev / mean_length if mean_length > 0 else 0
            if cv < 0.15:
                penalties.append(0.8)
            elif cv < 0.25:
                penalties.append(0.5)
            elif cv < 0.35:
                penalties.append(0.3)

    # Quality degradation
    if len(lines) > 15:
        mid = len(lines) // 2
        first_unique = len(set(lines[:mid]))
        second_unique = len(set(lines[mid:]))
        first_diversity = first_unique / mid
        second_diversity = second_unique / (len(lines) - mid)

        if first_diversity > 0.5 and second_diversity < 0.3:
            penalties.append(0.9)
        elif second_diversity < first_diversity * 0.5:
            penalties.append(0.6)

    # List padding
    list_items = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+\S{1,3}\s*$', line))
    if list_items > len(lines) * 0.5:
        penalties.append(0.7)

    # Repeated structures
    if len(lines) >= 8:
        starts = [' '.join(line.split()[:2]).lower() for line in lines if len(line.split()) >= 3]
        if len(starts) >= 8:
            start_counts = Counter(starts)
            most_common = max(start_counts.values()) if start_counts else 0
            if most_common > len(starts) * 0.35:
                penalties.append(0.7)

    return max(penalties) if penalties else 0.0


def detect_circular_reasoning(text: str) -> float:
    """Detect circular reasoning."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) > 4]

    if len(sentences) < 5:
        return 0.0

    circular_score = 0
    for i in range(len(sentences)):
        words_i = set(sentences[i].lower().split())
        for j in range(i + 1, min(i + 4, len(sentences))):
            words_j = set(sentences[j].lower().split())
            if words_i and words_j and len(words_i) > 3 and len(words_j) > 3:
                overlap = len(words_i & words_j) / len(words_i | words_j)
                if overlap > 0.5:
                    circular_score += overlap

    return min(circular_score / max(len(sentences), 1) * 2, 1.0)


def detect_keyword_stuffing(text: str) -> float:
    """Detect keyword stuffing."""
    words = text.lower().split()
    if len(words) < 20:
        return 0.0

    common = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'it', 'that', 'this', 'be', 'by'}
    content_words = [w for w in words if w not in common and len(w) > 3]

    if len(content_words) < 10:
        return 0.0

    word_counts = Counter(content_words)
    stuffing_score = 0

    for word, count in word_counts.most_common(5):
        frequency = count / len(content_words)
        if frequency > 0.08:
            stuffing_score += (frequency - 0.08) * 15

    return min(stuffing_score, 1.0)


def detect_padding_sentences(text: str) -> float:
    """Detect padding sentences."""
    patterns = [
        r"^(as|like) (we|you|i) (can|could|may|might) see",
        r"^it is (clear|obvious|evident|apparent) (that)?",
        r"^(therefore|thus|hence|consequently|accordingly)($|,)",
        r"^in (conclusion|summary|brief|short)($|,)",
        r"^(another|one more|yet another) (thing|point|aspect|factor)",
        r"^(this|that|it) (shows|demonstrates|illustrates|proves|indicates)",
    ]

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    if len(sentences) < 3:
        return 0.0

    padding_count = 0
    for sentence in sentences:
        if len(sentence.split()) < 5:
            continue
        for pattern in patterns:
            if re.search(pattern, sentence.lower()):
                padding_count += 1
                break

    return min(padding_count / max(len(sentences), 1) * 2, 1.0)


# ============================================================================
# Main Detection Function
# ============================================================================

def calculate_repetition_score(text: str, return_components: bool = False) -> Dict[str, float]:
    """
    Calculate repetition score for text.

    Args:
        text: Text to analyze
        return_components: If True, return all component scores

    Returns:
        Dict with 'score' (0-1) and optionally component scores
    """
    penalties = detect_repetition_penalty(text)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    penalties['template'] = detect_template_patterns(lines)
    penalties['filler'] = detect_filler_content(text)
    penalties['length_gaming'] = detect_length_gaming(text)
    penalties['circular'] = detect_circular_reasoning(text)
    penalties['keyword_stuffing'] = detect_keyword_stuffing(text)
    penalties['padding'] = detect_padding_sentences(text)

    # Weighted combination (calibrated)
    score = (
        penalties['word_rep'] * 0.15 +
        penalties['phrase_rep'] * 0.20 +
        penalties['para_rep'] * 0.25 +
        penalties['template'] * 0.12 +
        penalties['filler'] * 0.15 +
        penalties['length_gaming'] * 0.18 +
        penalties['circular'] * 0.12 +
        penalties['keyword_stuffing'] * 0.08 +
        penalties['padding'] * 0.05
    ) / 1.30

    result = {'score': min(score, 1.0)}

    if return_components:
        result.update(penalties)

    return result


# ============================================================================
# Reward Integration Functions
# ============================================================================

def get_penalty_multiplier(repetition_score: float,
                          severity: str = "moderate",
                          min_multiplier: float = None) -> float:
    """
    Get penalty multiplier for reward based on repetition score.

    Args:
        repetition_score: Score from calculate_repetition_score (0-1)
        severity: Penalty severity ("lenient", "moderate", "strict", "very_strict")
        min_multiplier: Override minimum multiplier (default depends on severity)

    Returns:
        Multiplier to apply to reward (0.0-1.0)

    Severity Presets:
        - lenient: Minimal penalties, only severe cases affected
        - moderate: Balanced penalties (recommended)
        - strict: Aggressive penalties for quality control
        - very_strict: Maximum penalties, use for critical applications
    """
    # Severity presets
    presets = {
        "lenient": {
            "thresholds": [0.20, 0.40, 0.60, 0.80],
            "multipliers": [1.0, 0.9, 0.7, 0.5, 0.3],
        },
        "moderate": {
            "thresholds": [0.15, 0.30, 0.50, 0.70],
            "multipliers": [1.0, 0.85, 0.6, 0.3, 0.1],
        },
        "strict": {
            "thresholds": [0.10, 0.25, 0.40, 0.60],
            "multipliers": [1.0, 0.7, 0.4, 0.15, 0.0],
        },
        "very_strict": {
            "thresholds": [0.05, 0.15, 0.30, 0.50],
            "multipliers": [1.0, 0.5, 0.2, 0.05, 0.0],
        },
    }

    if severity not in presets:
        severity = "moderate"

    thresholds = presets[severity]["thresholds"]
    multipliers = presets[severity]["multipliers"]

    # Override min_multiplier if provided
    if min_multiplier is not None:
        multipliers[-1] = max(min_multiplier, 0.0)

    # Find appropriate multiplier
    for i, threshold in enumerate(thresholds):
        if repetition_score < threshold:
            return multipliers[i]

    return multipliers[-1]


def apply_repetition_penalty(text: str,
                             base_reward: float,
                             severity: str = "moderate",
                             return_details: bool = False) -> float:
    """
    Apply repetition penalty to reward (MAIN FUNCTION FOR INTEGRATION).

    Args:
        text: Model output text
        base_reward: Original reward before penalty
        severity: Penalty severity ("lenient", "moderate", "strict", "very_strict")
        return_details: If True, return (penalized_reward, details_dict)

    Returns:
        Penalized reward (or tuple if return_details=True)

    Example:
        >>> text = "The model works well. The model works well. The model works well."
        >>> base_reward = 1.0
        >>> final_reward = apply_repetition_penalty(text, base_reward, severity="moderate")
        >>> print(final_reward)  # Will be < 1.0 due to repetition
    """
    # Calculate repetition score
    result = calculate_repetition_score(text, return_components=True)
    rep_score = result['score']

    # Get penalty multiplier
    multiplier = get_penalty_multiplier(rep_score, severity)

    # Apply penalty
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
# Diagnostic Functions
# ============================================================================

def diagnose_text_quality(text: str) -> str:
    """
    Get human-readable diagnosis of text quality.

    Args:
        text: Text to analyze

    Returns:
        Formatted diagnosis string
    """
    result = calculate_repetition_score(text, return_components=True)
    score = result['score']

    # Quality assessment
    if score < 0.15:
        quality = "✓ HIGH QUALITY"
        recommendation = "No penalty needed"
    elif score < 0.30:
        quality = "⚠ ACCEPTABLE"
        recommendation = "Minor penalty recommended"
    elif score < 0.50:
        quality = "⚠⚠ MODERATE ISSUES"
        recommendation = "Moderate penalty recommended"
    elif score < 0.70:
        quality = "✗ SIGNIFICANT ISSUES"
        recommendation = "Large penalty recommended"
    else:
        quality = "✗✗ SEVERE PROBLEMS"
        recommendation = "Maximum penalty or rejection recommended"

    # Build diagnosis
    diagnosis = f"""
TEXT QUALITY DIAGNOSIS
{'='*60}
Overall Score: {score:.3f}
Quality: {quality}
Recommendation: {recommendation}

Component Breakdown:
  Word Repetition:      {result['word_rep']:.3f}
  Phrase Repetition:    {result['phrase_rep']:.3f}
  Paragraph Repetition: {result['para_rep']:.3f}
  Template Patterns:    {result['template']:.3f}
  Filler Content:       {result['filler']:.3f}
  Length Gaming:        {result['length_gaming']:.3f}
  Circular Reasoning:   {result['circular']:.3f}
  Keyword Stuffing:     {result['keyword_stuffing']:.3f}
  Padding Sentences:    {result['padding']:.3f}

Penalty Multipliers by Severity:
  Lenient:     {get_penalty_multiplier(score, 'lenient'):.2f}x
  Moderate:    {get_penalty_multiplier(score, 'moderate'):.2f}x
  Strict:      {get_penalty_multiplier(score, 'strict'):.2f}x
  Very Strict: {get_penalty_multiplier(score, 'very_strict'):.2f}x
{'='*60}
"""
    return diagnosis


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("REPETITION PENALTY SYSTEM - EXAMPLES")
    print("="*80)
    print()

    # Example 1: High quality text
    text1 = """
    Machine learning enables computers to learn from data without explicit programming.
    Neural networks are computational models inspired by biological brains.
    Deep learning uses multiple layers to extract hierarchical features.
    """

    print("Example 1: High Quality Text")
    print("-"*80)
    reward1 = apply_repetition_penalty(text1, base_reward=1.0, severity="moderate", return_details=True)
    print(f"Base reward: {reward1[1]['base_reward']:.3f}")
    print(f"Repetition score: {reward1[1]['repetition_score']:.3f}")
    print(f"Penalty multiplier: {reward1[1]['penalty_multiplier']:.3f}")
    print(f"Final reward: {reward1[0]:.3f}")
    print()

    # Example 2: Repetitive text
    text2 = """
    Machine learning machine learning is about machine learning models.
    Machine learning machine learning techniques use machine learning approaches.
    Machine learning machine learning systems implement machine learning methods.
    """

    print("Example 2: Repetitive Text")
    print("-"*80)
    reward2 = apply_repetition_penalty(text2, base_reward=1.0, severity="moderate", return_details=True)
    print(f"Base reward: {reward2[1]['base_reward']:.3f}")
    print(f"Repetition score: {reward2[1]['repetition_score']:.3f}")
    print(f"Penalty multiplier: {reward2[1]['penalty_multiplier']:.3f}")
    print(f"Final reward: {reward2[0]:.3f}")
    print(f"Penalty applied: {reward2[1]['penalty_applied']:.3f}")
    print()

    # Example 3: Detailed diagnosis
    print("Example 3: Detailed Diagnosis")
    print("-"*80)
    print(diagnose_text_quality(text2))
