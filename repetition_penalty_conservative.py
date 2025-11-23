#!/usr/bin/env python3
"""
Conservative Repetition Detection - Recalibrated for High Precision

Philosophy: Only penalize clear, egregious repetition. Keep scores intact for borderline cases.

Key Changes:
- Higher thresholds for detection
- Focus on only the most reliable signals
- Reduced sensitivity to minor issues
- Conservative severity presets as default
- Only penalizes "inhumane" robotic responses
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
# Conservative Detection Functions
# ============================================================================

def detect_word_repetition_conservative(text: str) -> float:
    """Conservative word repetition - only catch egregious cases."""
    if not text.strip():
        return 0.0

    words = text.lower().split()
    if len(words) < 20:  # Too short to judge
        return 0.0

    # Expanded common words list
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'it', 'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'from', 'as', 'by', 'not',
        'so', 'than', 'such', 'no', 'yes', 'if', 'when', 'where', 'what', 'which', 'who',
        'how', 'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'any', 'only', 'own', 'same', 'very', 'just', 'much', 'even', 'also',
        'well', 'back', 'through', 'still', 'way', 'good', 'new', 'first', 'last',
        'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high',
        'different', 'small', 'large', 'next', 'early', 'young', 'important', 'few',
        'public', 'bad', 'same', 'able'
    }

    content_words = [w for w in words if w not in common_words and len(w) > 3]

    if len(content_words) < 10:
        return 0.0

    word_counts = Counter(content_words)

    # Only penalize VERY excessive repetition (5+ times for same word)
    excessive_reps = sum(max(0, count - 4) for count in word_counts.values())

    if excessive_reps == 0:
        return 0.0

    # More conservative scoring
    score = min(excessive_reps / max(len(content_words), 1) * 3, 1.0)

    # Only return score if it's truly egregious
    return score if score > 0.3 else 0.0


def detect_phrase_repetition_conservative(text: str) -> float:
    """Conservative phrase repetition - only catch obvious cases."""
    words = text.lower().split()

    if len(words) < 15:  # Too short
        return 0.0

    # Use 4-grams instead of 3-grams (more specific)
    ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]

    if len(ngrams) < 10:
        return 0.0

    ngram_counts = Counter(ngrams)

    # Only penalize if same phrase appears 3+ times
    repeated_ngrams = sum(max(0, count - 2) for count in ngram_counts.values())

    if repeated_ngrams == 0:
        return 0.0

    score = min(repeated_ngrams / max(len(ngrams), 1) * 4, 1.0)

    # Only return if truly problematic
    return score if score > 0.4 else 0.0


def detect_line_repetition_conservative(text: str) -> float:
    """Conservative line repetition - exact duplicates only."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip() and len(l.split()) > 3]

    if len(lines) < 5:
        return 0.0

    line_counts = Counter(lines)

    # Only penalize EXACT duplicates that appear 3+ times
    repeated_lines = sum(max(0, count - 2) for count in line_counts.values())

    if repeated_lines == 0:
        return 0.0

    score = min(repeated_lines / max(len(lines), 1) * 3, 1.0)

    # Only return if very problematic
    return score if score > 0.5 else 0.0


def detect_template_abuse_conservative(lines: List[str]) -> float:
    """Conservative template detection - only catch blatant templates."""
    if len(lines) < 6:  # Need more lines to be confident
        return 0.0

    # Only check lines with substantial content
    substantial_lines = [l for l in lines if len(l.split()) >= 5]

    if len(substantial_lines) < 6:
        return 0.0

    # Check for very rigid sentence structures
    patterns = []
    for line in substantial_lines:
        words = line.split()
        # Extract first 3 words and last word as pattern
        if len(words) >= 4:
            first_part = ' '.join(words[:3]).lower()
            patterns.append(first_part)

    if len(patterns) < 6:
        return 0.0

    pattern_counts = Counter(patterns)

    # Only penalize if 4+ lines start with exact same 3 words
    excessive_templates = sum(max(0, count - 3) for count in pattern_counts.values())

    if excessive_templates == 0:
        return 0.0

    score = min(excessive_templates / len(patterns) * 2, 1.0)

    # Only return if very clear template abuse
    return score if score > 0.5 else 0.0


def detect_filler_heavy_conservative(text: str) -> float:
    """Conservative filler detection - only heavy filler abuse."""
    # Most problematic fillers only
    critical_fillers = [
        "it is worth noting that",
        "it should be mentioned that",
        "as mentioned before",
        "as previously stated",
        "to reiterate",
        "to repeat",
        "once again",
    ]

    text_lower = text.lower()
    critical_count = sum(text_lower.count(phrase) for phrase in critical_fillers)

    # Only penalize if many critical fillers
    if critical_count < 3:
        return 0.0

    words = len(text.split())
    if words < 50:
        return 0.0

    # Very conservative normalization
    filler_density = critical_count / (words / 100)  # Per 100 words

    score = min(filler_density / 5, 1.0)

    # Only return if excessive
    return score if score > 0.4 else 0.0


def detect_length_gaming_conservative(text: str) -> float:
    """Conservative length gaming - only obvious cases."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 10:  # Need more lines to be confident
        return 0.0

    penalties = []

    # 1. VERY uniform line length (suspiciously robotic)
    if len(lines) >= 12:
        line_lengths = [len(line.split()) for line in lines]

        if HAS_NUMPY:
            std_dev = np.std(line_lengths)
            mean_length = np.mean(line_lengths)
        else:
            mean_length = statistics.mean(line_lengths)
            std_dev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0

        if mean_length > 5:
            cv = std_dev / mean_length if mean_length > 0 else 0
            # Only penalize VERY uniform (almost robotic)
            if cv < 0.10:  # Extremely uniform
                penalties.append(0.9)
            elif cv < 0.15:  # Very uniform
                penalties.append(0.6)

    # 2. Severe quality degradation
    if len(lines) > 20:
        mid = len(lines) // 2
        first_unique = len(set(lines[:mid]))
        second_unique = len(set(lines[mid:]))

        first_diversity = first_unique / mid
        second_diversity = second_unique / (len(lines) - mid)

        # Only penalize SEVERE degradation
        if first_diversity > 0.7 and second_diversity < 0.2:
            penalties.append(1.0)

    # 3. Massive list padding (> 70% of content)
    list_items = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+\S{1,4}\s*$', line))
    if list_items > len(lines) * 0.7:
        penalties.append(0.8)

    return max(penalties) if penalties else 0.0


def detect_circular_reasoning_conservative(text: str) -> float:
    """Conservative circular reasoning - only very obvious cases."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) > 6]

    if len(sentences) < 6:
        return 0.0

    # Only check for VERY high overlap (near-duplicates)
    circular_count = 0
    for i in range(len(sentences)):
        words_i = set(sentences[i].lower().split())
        if len(words_i) < 5:
            continue

        for j in range(i + 1, min(i + 3, len(sentences))):
            words_j = set(sentences[j].lower().split())
            if len(words_j) < 5:
                continue

            overlap = len(words_i & words_j) / len(words_i | words_j)
            # Only count VERY similar sentences (70%+ overlap)
            if overlap > 0.7:
                circular_count += 1

    if circular_count < 2:
        return 0.0

    score = min(circular_count / len(sentences) * 3, 1.0)

    return score if score > 0.3 else 0.0


def detect_keyword_stuffing_conservative(text: str) -> float:
    """Conservative keyword stuffing - only extreme cases."""
    words = text.lower().split()
    if len(words) < 30:
        return 0.0

    common = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'it', 'that', 'this', 'be', 'by', 'from', 'as', 'not', 'will', 'would',
        'can', 'could', 'should', 'may', 'might', 'must', 'do', 'does', 'did'
    }

    content_words = [w for w in words if w not in common and len(w) > 3]

    if len(content_words) < 15:
        return 0.0

    word_counts = Counter(content_words)

    # Only penalize if a word is used EXTREMELY often (>15% of content)
    stuffing_score = 0
    for word, count in word_counts.most_common(3):
        frequency = count / len(content_words)
        if frequency > 0.15:  # More than 15%
            stuffing_score += (frequency - 0.15) * 20

    score = min(stuffing_score, 1.0)

    return score if score > 0.5 else 0.0


def detect_padding_sentences_conservative(text: str) -> float:
    """Conservative padding detection - only check most obvious padding."""
    # Only the most obvious padding patterns
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
# Main Conservative Detection
# ============================================================================

def calculate_repetition_score_conservative(text: str, return_components: bool = False) -> Dict[str, float]:
    """
    CONSERVATIVE repetition detection - only penalizes clear, egregious cases.

    High precision, low false positive rate.
    Only flags truly "inhumane" robotic responses.
    """
    if not text or len(text.strip()) < 50:
        # Too short to judge
        return {'score': 0.0}

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Run conservative detectors
    components = {
        'word_rep': detect_word_repetition_conservative(text),
        'phrase_rep': detect_phrase_repetition_conservative(text),
        'line_rep': detect_line_repetition_conservative(text),
        'template': detect_template_abuse_conservative(lines),
        'filler': detect_filler_heavy_conservative(text),
        'length_gaming': detect_length_gaming_conservative(text),
        'circular': detect_circular_reasoning_conservative(text),
        'keyword_stuffing': detect_keyword_stuffing_conservative(text),
        'padding': detect_padding_sentences_conservative(text),
    }

    # CONSERVATIVE WEIGHTS - focus on most reliable signals
    # Higher weight on line/paragraph repetition (most reliable)
    # Lower weight on word repetition (can be natural)
    score = (
        components['word_rep'] * 0.08 +        # Reduced
        components['phrase_rep'] * 0.15 +      # Moderate
        components['line_rep'] * 0.30 +        # High (very reliable)
        components['template'] * 0.12 +        # Moderate
        components['filler'] * 0.10 +          # Moderate
        components['length_gaming'] * 0.20 +   # High (clear signal)
        components['circular'] * 0.08 +        # Reduced
        components['keyword_stuffing'] * 0.05 + # Reduced
        components['padding'] * 0.03           # Reduced
    ) / 1.11  # Normalize

    result = {'score': min(score, 1.0)}

    if return_components:
        result.update(components)

    return result


# ============================================================================
# Conservative Penalty Application
# ============================================================================

def get_conservative_penalty_multiplier(score: float, mode: str = "default") -> float:
    """
    Conservative penalty multipliers - only penalize clear problems.

    Modes:
        - ultra_conservative: Almost never penalize (only extreme cases)
        - default: Conservative but fair (recommended)
        - balanced: More penalties but still conservative
    """
    if mode == "ultra_conservative":
        # Almost never penalize
        if score < 0.50:
            return 1.0  # No penalty
        elif score < 0.70:
            return 0.9  # 10% penalty
        elif score < 0.85:
            return 0.7  # 30% penalty
        else:
            return 0.5  # 50% penalty (only extreme cases)

    elif mode == "default":
        # Conservative but fair (RECOMMENDED)
        if score < 0.35:
            return 1.0  # No penalty - benefit of doubt
        elif score < 0.50:
            return 0.9  # 10% penalty - mild issue
        elif score < 0.65:
            return 0.7  # 30% penalty - clear issue
        elif score < 0.80:
            return 0.5  # 50% penalty - serious issue
        else:
            return 0.2  # 80% penalty - egregious

    elif mode == "balanced":
        # More penalties but still conservative
        if score < 0.25:
            return 1.0  # No penalty
        elif score < 0.40:
            return 0.85  # 15% penalty
        elif score < 0.55:
            return 0.65  # 35% penalty
        elif score < 0.70:
            return 0.4   # 60% penalty
        else:
            return 0.1   # 90% penalty

    # Default to "default" mode
    return get_conservative_penalty_multiplier(score, "default")


def apply_repetition_penalty_conservative(
    text: str,
    base_reward: float,
    mode: str = "default",
    return_details: bool = False
):
    """
    Apply CONSERVATIVE repetition penalty.

    Only penalizes clear, egregious repetition. Keeps scores intact for borderline cases.

    Args:
        text: Text to analyze
        base_reward: Original reward
        mode: "ultra_conservative", "default", or "balanced"
        return_details: Return detailed breakdown

    Returns:
        Penalized reward (or tuple with details)

    Philosophy:
        - If unsure, don't penalize
        - Only flag truly inhumane/robotic responses
        - High precision over high recall
    """
    result = calculate_repetition_score_conservative(text, return_components=True)
    score = result['score']

    multiplier = get_conservative_penalty_multiplier(score, mode)
    penalized_reward = base_reward * multiplier

    if return_details:
        details = {
            'repetition_score': score,
            'penalty_multiplier': multiplier,
            'base_reward': base_reward,
            'penalized_reward': penalized_reward,
            'penalty_applied': base_reward - penalized_reward,
            'components': result,
            'assessment': get_conservative_assessment(score)
        }
        return penalized_reward, details

    return penalized_reward


def get_conservative_assessment(score: float) -> str:
    """Get assessment for conservative thresholds."""
    if score < 0.35:
        return "✓ ACCEPTABLE - No penalty"
    elif score < 0.50:
        return "⚠ MINOR_ISSUES - Small penalty"
    elif score < 0.65:
        return "⚠⚠ CLEAR_ISSUES - Moderate penalty"
    elif score < 0.80:
        return "✗ SERIOUS_ISSUES - Large penalty"
    else:
        return "✗✗ EGREGIOUS - Maximum penalty"


# ============================================================================
# Diagnostic Function
# ============================================================================

def diagnose_conservative(text: str) -> str:
    """Diagnose text with conservative thresholds."""
    result = calculate_repetition_score_conservative(text, return_components=True)
    score = result['score']
    assessment = get_conservative_assessment(score)

    diagnosis = f"""
CONSERVATIVE REPETITION ANALYSIS
{'='*70}
Overall Score: {score:.3f}
Assessment: {assessment}

Component Scores (only non-zero):
"""

    for component, value in result.items():
        if component != 'score' and value > 0:
            diagnosis += f"  {component:20s}: {value:.3f} ⚠\n"

    if score == 0.0:
        diagnosis += "  [No issues detected]\n"

    diagnosis += f"""
Penalty Multipliers:
  Ultra Conservative: {get_conservative_penalty_multiplier(score, 'ultra_conservative'):.2f}x
  Default (Recommended): {get_conservative_penalty_multiplier(score, 'default'):.2f}x
  Balanced: {get_conservative_penalty_multiplier(score, 'balanced'):.2f}x
{'='*70}
"""
    return diagnosis


if __name__ == "__main__":
    print("CONSERVATIVE REPETITION DETECTION - EXAMPLES")
    print("="*70)
    print()

    # Test 1: Good quality
    text1 = """
    Machine learning enables systems to learn from data patterns.
    Neural networks process information through interconnected layers.
    Training involves optimizing parameters using backpropagation.
    Deep learning architectures can model complex relationships.
    """

    print("Test 1: High Quality Text")
    print("-"*70)
    reward1, details1 = apply_repetition_penalty_conservative(
        text1, 1.0, mode="default", return_details=True
    )
    print(f"Score: {details1['repetition_score']:.3f}")
    print(f"Multiplier: {details1['penalty_multiplier']:.3f}")
    print(f"Final reward: {reward1:.3f}")
    print(f"Assessment: {details1['assessment']}")
    print()

    # Test 2: Moderate repetition (should NOT penalize much)
    text2 = """
    The model uses machine learning. Machine learning helps with predictions.
    We can see that the approach works. The approach is effective.
    """

    print("Test 2: Moderate Repetition (Should be lenient)")
    print("-"*70)
    reward2, details2 = apply_repetition_penalty_conservative(
        text2, 1.0, mode="default", return_details=True
    )
    print(f"Score: {details2['repetition_score']:.3f}")
    print(f"Multiplier: {details2['penalty_multiplier']:.3f}")
    print(f"Final reward: {reward2:.3f}")
    print(f"Assessment: {details2['assessment']}")
    print()

    # Test 3: Extreme repetition (SHOULD penalize)
    text3 = """
    Machine learning machine learning machine learning machine learning machine learning.
    Machine learning machine learning machine learning machine learning machine learning.
    Machine learning machine learning machine learning machine learning machine learning.
    Machine learning machine learning machine learning machine learning machine learning.
    Machine learning machine learning machine learning machine learning machine learning.
    """

    print("Test 3: Extreme Repetition (Should penalize)")
    print("-"*70)
    reward3, details3 = apply_repetition_penalty_conservative(
        text3, 1.0, mode="default", return_details=True
    )
    print(f"Score: {details3['repetition_score']:.3f}")
    print(f"Multiplier: {details3['penalty_multiplier']:.3f}")
    print(f"Final reward: {reward3:.3f}")
    print(f"Assessment: {details3['assessment']}")
    print()

    # Test 4: Detailed diagnosis
    print("Test 4: Detailed Diagnosis")
    print("-"*70)
    print(diagnose_conservative(text3))
