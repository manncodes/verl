#!/usr/bin/env python3
"""
Improved Advanced Repetition Detection

Based on calibration testing, this version includes:
- Fixed length gaming detector
- Adjusted weights for better discrimination
- Improved template detection (fewer false positives)
- Better calibrated thresholds
"""

import re
from typing import List, Dict
from collections import Counter
import statistics

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def detect_repetition_penalty(text: str) -> Dict[str, float]:
    """Base repetition detection from word/phrase/paragraph level."""
    if not text.strip():
        return {'word_rep': 0, 'phrase_rep': 0, 'para_rep': 0}

    words = text.lower().split()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Word-level repetition (improved)
    if len(words) > 0:
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'it', 'that', 'this', 'be', 'by'}
        content_words = [w for w in words if w not in common_words and len(w) > 2]

        if content_words:
            word_counts = Counter(content_words)
            # More aggressive scoring
            excessive_reps = sum(max(0, count - 2) for count in word_counts.values())
            word_rep_score = min(excessive_reps / max(len(content_words), 1) * 2, 1.0)
        else:
            word_rep_score = 0.0
    else:
        word_rep_score = 0.0

    # Phrase-level repetition (improved)
    if len(words) >= 3:
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        # More aggressive - any repeated trigram is suspicious
        repeated_trigrams = sum(max(0, count - 1) for count in trigram_counts.values())
        phrase_rep_score = min(repeated_trigrams / max(len(trigrams), 1) * 3, 1.0)
    else:
        phrase_rep_score = 0.0

    # Paragraph/line-level repetition (improved)
    if len(lines) > 1:
        line_counts = Counter(lines)
        # Exact line repetition is very suspicious
        repeated_lines = sum(max(0, count - 1) for count in line_counts.values())
        para_rep_score = min(repeated_lines / max(len(lines), 1) * 2, 1.0)
    else:
        para_rep_score = 0.0

    return {
        'word_rep': word_rep_score,
        'phrase_rep': phrase_rep_score,
        'para_rep': para_rep_score
    }


def detect_template_patterns(lines: List[str]) -> float:
    """Detect if text follows repetitive templates (improved to reduce false positives)."""
    if len(lines) < 4:  # Need more lines to detect templates
        return 0.0

    penalties = []

    # Strategy 1: Structural patterns (improved)
    patterns = []
    for line in lines:
        if not line.strip() or len(line.split()) < 4:  # Skip short lines
            continue
        words = line.split()
        pattern_parts = []
        for i, word in enumerate(words):
            # Keep structure words, mask content words
            if word.lower() in {'is', 'are', 'was', 'were', 'because', 'that', 'the', 'a', 'an'}:
                pattern_parts.append(word.lower())
            elif word.isupper():
                pattern_parts.append('[UPPER]')
            elif word[0].isupper() and i > 0:  # Not sentence start
                pattern_parts.append('[CAP]')
            elif word.isdigit():
                pattern_parts.append('[NUM]')
            else:
                pattern_parts.append('X')
        patterns.append(' '.join(pattern_parts))

    if len(patterns) >= 4:
        pattern_counts = Counter(patterns)
        # Only penalize if same pattern appears 3+ times
        repeated_patterns = sum(max(0, count - 2) for count in pattern_counts.values())
        template_score_1 = min(repeated_patterns / max(len(lines), 1) * 2, 1.0)
        penalties.append(template_score_1)

    # Strategy 2: Sentence structure similarity (improved threshold)
    if len(lines) >= 6:
        first_words = []
        for line in lines:
            words = line.split()
            if len(words) >= 3:  # Need substantial lines
                first_words.append(' '.join(words[:2]).lower())

        if len(first_words) >= 6:
            first_word_counts = Counter(first_words)
            # More aggressive - if 4+ lines start the same way
            same_starts = sum(max(0, count - 3) for count in first_word_counts.values())
            template_score_2 = min(same_starts / len(first_words) * 3, 1.0)
            penalties.append(template_score_2)

    return max(penalties) if penalties else 0.0


def detect_filler_content(text: str) -> float:
    """Detect common filler phrases (improved normalization)."""
    # Expanded filler phrases
    transition_fillers = [
        "it is worth noting that", "it should be mentioned", "it is important to note",
        "it is essential to understand", "one could say that", "in other words",
        "that being said", "that said", "as mentioned before", "as previously stated",
        "as stated earlier", "to reiterate", "to repeat", "once again", "in summary",
        "to summarize",
    ]

    redundant_intensifiers = [
        "furthermore", "additionally", "moreover", "in addition", "also",
        "as well", "similarly", "likewise",
    ]

    vague_qualifiers = [
        "it seems that", "it appears that", "one might say", "arguably",
        "potentially", "possibly", "perhaps", "in a sense", "to some extent",
    ]

    hedging_phrases = [
        "it could be argued that", "some might argue that", "it is possible that",
        "there is a possibility that", "it may be that",
    ]

    text_lower = text.lower()

    # Count with weights
    transition_count = sum(text_lower.count(phrase) for phrase in transition_fillers)
    intensifier_count = sum(text_lower.count(phrase) for phrase in redundant_intensifiers)
    qualifier_count = sum(text_lower.count(phrase) for phrase in vague_qualifiers)
    hedging_count = sum(text_lower.count(phrase) for phrase in hedging_phrases)

    weighted_filler_count = (
        transition_count * 2.5 +
        intensifier_count * 1.2 +
        qualifier_count * 1.8 +
        hedging_count * 2.0
    )

    # Improved normalization
    words = len(text.split())
    if words > 20:
        # More aggressive - even a few fillers in short text is bad
        filler_density = weighted_filler_count / (words / 50)  # Per 50 words
        return min(filler_density / 3, 1.0)

    return 0.0


def detect_length_gaming(text: str) -> float:
    """
    Detect artificial lengthening strategies (FIXED VERSION).
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 5:
        return 0.0

    penalties = []

    # Strategy 1: Uniform line length (FIXED - was not working!)
    line_lengths = [len(line.split()) for line in lines]

    if len(line_lengths) >= 8:  # Need enough lines
        if HAS_NUMPY:
            std_dev = np.std(line_lengths)
            mean_length = np.mean(line_lengths)
        else:
            mean_length = statistics.mean(line_lengths)
            std_dev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0

        if mean_length > 3:  # Only for substantial lines
            cv = std_dev / mean_length if mean_length > 0 else 0
            # FIXED: More aggressive thresholds
            if cv < 0.15:  # Very uniform
                penalties.append(0.8)
            elif cv < 0.25:  # Quite uniform
                penalties.append(0.5)
            elif cv < 0.35:  # Moderately uniform
                penalties.append(0.3)

    # Strategy 2: Quality degradation (improved)
    if len(lines) > 15:
        mid_point = len(lines) // 2
        first_half_unique = len(set(lines[:mid_point]))
        second_half_unique = len(set(lines[mid_point:]))

        first_half_size = mid_point
        second_half_size = len(lines) - mid_point

        # Normalize by size
        first_diversity = first_half_unique / max(first_half_size, 1)
        second_diversity = second_half_unique / max(second_half_size, 1)

        if first_diversity > 0.5 and second_diversity < 0.3:
            penalties.append(0.9)
        elif second_diversity < first_diversity * 0.5:
            penalties.append(0.6)

    # Strategy 3: List padding (improved detection)
    list_markers = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+\S{1,3}\s*$', line))
    if list_markers > len(lines) * 0.5:
        penalties.append(0.7)

    # Strategy 4: Repeated sentence structures (improved)
    if len(lines) >= 8:
        sentence_starts = []
        for line in lines:
            words = line.split()
            if len(words) >= 3:
                sentence_starts.append(' '.join(words[:2]).lower())

        if len(sentence_starts) >= 8:
            start_counts = Counter(sentence_starts)
            most_common_count = max(start_counts.values()) if start_counts else 0
            if most_common_count > len(sentence_starts) * 0.35:
                penalties.append(0.7)

    # Strategy 5: Adjacent similarity (improved)
    if len(lines) > 10:
        similar_adjacent = 0
        for i in range(len(lines) - 1):
            words1 = set(lines[i].lower().split())
            words2 = set(lines[i + 1].lower().split())
            if words1 and words2 and len(words1) > 2 and len(words2) > 2:
                overlap = len(words1 & words2) / len(words1 | words2)
                if overlap > 0.6:
                    similar_adjacent += 1

        if similar_adjacent > len(lines) * 0.25:
            penalties.append(0.6)

    return max(penalties) if penalties else 0.0


def detect_circular_reasoning(text: str) -> float:
    """Detect circular reasoning patterns (improved)."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 4]

    if len(sentences) < 5:
        return 0.0

    circular_score = 0
    for i in range(len(sentences)):
        words_i = set(sentences[i].lower().split())
        # Check next 2-3 sentences for high overlap
        for j in range(i + 1, min(i + 4, len(sentences))):
            words_j = set(sentences[j].lower().split())
            if words_i and words_j and len(words_i) > 3 and len(words_j) > 3:
                overlap = len(words_i & words_j) / len(words_i | words_j)
                if overlap > 0.5:  # High overlap
                    circular_score += overlap

    return min(circular_score / max(len(sentences), 1) * 2, 1.0)


def detect_keyword_stuffing(text: str) -> float:
    """Detect keyword stuffing (improved)."""
    words = text.lower().split()
    if len(words) < 20:
        return 0.0

    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'it', 'that', 'this', 'be', 'by'}
    content_words = [w for w in words if w not in common_words and len(w) > 3]

    if len(content_words) < 10:
        return 0.0

    word_counts = Counter(content_words)
    total_content = len(content_words)

    stuffing_score = 0
    for word, count in word_counts.most_common(5):
        frequency = count / total_content
        if frequency > 0.08:  # More than 8% (lowered threshold)
            stuffing_score += (frequency - 0.08) * 15  # More aggressive

    return min(stuffing_score, 1.0)


def detect_padding_sentences(text: str) -> float:
    """Detect padding sentences (improved)."""
    padding_indicators = [
        r"^(as|like) (we|you|i) (can|could|may|might) see",
        r"^it is (clear|obvious|evident|apparent) (that)?",
        r"^(therefore|thus|hence|consequently|accordingly)($|,)",
        r"^in (conclusion|summary|brief|short)($|,)",
        r"^(another|one more|yet another) (thing|point|aspect|factor)",
        r"^(this|that|it) (shows|demonstrates|illustrates|proves|indicates)",
    ]

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 3:
        return 0.0

    padding_count = 0
    for sentence in sentences:
        if len(sentence.split()) < 5:  # Only check substantial sentences
            continue
        sentence_lower = sentence.lower()
        for pattern in padding_indicators:
            if re.search(pattern, sentence_lower):
                padding_count += 1
                break

    return min(padding_count / max(len(sentences), 1) * 2, 1.0)


def advanced_repetition_detector(text: str, return_components: bool = False) -> Dict[str, float]:
    """
    Comprehensive repetition detection (IMPROVED & CALIBRATED).

    Args:
        text: Text to analyze
        return_components: If True, return all component scores. If False, return only overall score.

    Returns:
        Dictionary with scores
    """
    # Base repetition
    penalties = detect_repetition_penalty(text)

    # Get lines for analysis
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Advanced checks
    penalties['template_score'] = detect_template_patterns(lines)
    penalties['filler_score'] = detect_filler_content(text)
    penalties['length_gaming'] = detect_length_gaming(text)
    penalties['circular_reasoning'] = detect_circular_reasoning(text)
    penalties['keyword_stuffing'] = detect_keyword_stuffing(text)
    penalties['padding_sentences'] = detect_padding_sentences(text)

    # IMPROVED WEIGHTS (calibrated based on testing)
    overall_score = (
        penalties['word_rep'] * 0.15 +          # Increased from 0.10
        penalties['phrase_rep'] * 0.20 +        # Increased from 0.15
        penalties['para_rep'] * 0.25 +          # Increased from 0.20
        penalties['template_score'] * 0.12 +    # Decreased from 0.15 (less false positives)
        penalties['filler_score'] * 0.15 +      # Increased from 0.10
        penalties['length_gaming'] * 0.18 +     # Increased from 0.15
        penalties['circular_reasoning'] * 0.12 + # Increased from 0.10
        penalties['keyword_stuffing'] * 0.08 +  # Increased from 0.03
        penalties['padding_sentences'] * 0.05   # Increased from 0.02
    )
    # Total weight = 1.30, so normalize
    overall_score = overall_score / 1.30

    penalties['overall_score'] = min(overall_score, 1.0)

    if return_components:
        return penalties
    else:
        return {'overall_score': penalties['overall_score']}


def get_quality_assessment(score: float) -> str:
    """Get quality assessment based on calibrated thresholds."""
    if score < 0.15:
        return "HIGH_QUALITY"
    elif score < 0.30:
        return "ACCEPTABLE"
    elif score < 0.45:
        return "MODERATE_ISSUES"
    elif score < 0.65:
        return "SIGNIFICANT_ISSUES"
    else:
        return "SEVERE_PROBLEMS"


def calculate_penalty_multiplier(score: float, min_multiplier: float = 0.5, max_multiplier: float = 1.0) -> float:
    """
    Calculate reward penalty multiplier based on repetition score.

    Args:
        score: Overall repetition score (0-1)
        min_multiplier: Minimum multiplier for worst case (default: 0.5 = 50% penalty)
        max_multiplier: Maximum multiplier for best case (default: 1.0 = no penalty)

    Returns:
        Penalty multiplier to apply to reward
    """
    if score < 0.15:  # High quality
        return max_multiplier
    elif score < 0.30:  # Acceptable
        return max_multiplier - (score - 0.15) * 0.5
    elif score < 0.50:  # Moderate issues
        return max_multiplier - 0.075 - (score - 0.30) * 1.0
    else:  # Significant to severe issues
        return min_multiplier


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    text1 = "Machine learning is a subset of AI. Neural networks learn patterns."
    result = advanced_repetition_detector(text1, return_components=True)
    print("Example 1 - Good quality text:")
    print(f"  Overall score: {result['overall_score']:.3f}")
    print(f"  Assessment: {get_quality_assessment(result['overall_score'])}")
    print(f"  Penalty multiplier: {calculate_penalty_multiplier(result['overall_score']):.3f}")
    print()

    # Example 2: Repetitive text
    text2 = """
    It is worth noting that machine learning is important. Furthermore, machine learning
    enables automation. Additionally, machine learning provides solutions. Moreover,
    machine learning delivers results. That being said, machine learning is crucial.
    """
    result = advanced_repetition_detector(text2, return_components=True)
    print("Example 2 - Repetitive text:")
    print(f"  Overall score: {result['overall_score']:.3f}")
    print(f"  Assessment: {get_quality_assessment(result['overall_score'])}")
    print(f"  Penalty multiplier: {calculate_penalty_multiplier(result['overall_score']):.3f}")
    print(f"  Filler score: {result['filler_score']:.3f}")
    print(f"  Keyword stuffing: {result['keyword_stuffing']:.3f}")
    print()

    # Example 3: Template abuse
    text3 = """
    Machine learning is important because it enables automation.
    Deep learning is important because it improves accuracy.
    Data quality is important because it ensures performance.
    Model selection is important because it optimizes results.
    """
    result = advanced_repetition_detector(text3, return_components=True)
    print("Example 3 - Template abuse:")
    print(f"  Overall score: {result['overall_score']:.3f}")
    print(f"  Assessment: {get_quality_assessment(result['overall_score'])}")
    print(f"  Penalty multiplier: {calculate_penalty_multiplier(result['overall_score']):.3f}")
    print(f"  Template score: {result['template_score']:.3f}")
    print()
