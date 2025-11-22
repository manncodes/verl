#!/usr/bin/env python3
"""
Advanced Repetition Detection - Testing and Calibration Suite

Tests and improves repetition detection functions for identifying when models
game the system with repetitive content, templates, filler phrases, or artificial
lengthening strategies.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import statistics

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Core Detection Functions (Improved)
# ============================================================================

def detect_repetition_penalty(text: str) -> Dict[str, float]:
    """
    Base repetition detection from word/phrase/paragraph level.
    Returns various repetition scores.
    """
    if not text.strip():
        return {'word_rep': 0, 'phrase_rep': 0, 'para_rep': 0}

    words = text.lower().split()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Word-level repetition
    if len(words) > 0:
        word_counts = Counter(words)
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        content_words = {w: c for w, c in word_counts.items() if w not in common_words and len(w) > 2}

        if content_words:
            max_word_rep = max(content_words.values())
            avg_word_rep = sum(content_words.values()) / len(content_words)
            word_rep_score = min((max_word_rep - 1) / 10, 1.0)  # Normalize
        else:
            word_rep_score = 0.0
    else:
        word_rep_score = 0.0

    # Phrase-level repetition (3-grams)
    if len(words) >= 3:
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(count - 1 for count in trigram_counts.values() if count > 1)
        phrase_rep_score = min(repeated_trigrams / max(len(trigrams), 1), 1.0)
    else:
        phrase_rep_score = 0.0

    # Paragraph/line-level repetition
    if len(lines) > 1:
        line_counts = Counter(lines)
        repeated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        para_rep_score = repeated_lines / len(lines)
    else:
        para_rep_score = 0.0

    return {
        'word_rep': word_rep_score,
        'phrase_rep': phrase_rep_score,
        'para_rep': para_rep_score
    }


def detect_template_patterns(lines: List[str]) -> float:
    """
    Detect if text follows repetitive templates.

    Improved version with multiple pattern detection strategies.
    """
    if len(lines) < 3:
        return 0.0

    # Strategy 1: Structural patterns
    patterns = []
    for line in lines:
        if not line.strip():
            continue
        # Convert to pattern based on word types
        words = line.split()
        pattern_parts = []
        for word in words:
            if word.isupper():
                pattern_parts.append('[UPPER]')
            elif word[0].isupper():
                pattern_parts.append('[CAP]')
            elif word.isdigit():
                pattern_parts.append('[NUM]')
            elif len(word) <= 3:
                pattern_parts.append(word.lower())  # Keep short words
            else:
                pattern_parts.append('[WORD]')
        patterns.append(' '.join(pattern_parts))

    pattern_counts = Counter(patterns)
    # High repetition of same pattern = template usage
    repeated_patterns = sum(count - 1 for count in pattern_counts.values() if count > 2)
    template_score_1 = min(repeated_patterns / max(len(lines), 1), 1.0)

    # Strategy 2: Sentence structure similarity
    # Check if lines start with the same words
    if len(lines) >= 5:
        first_words = []
        for line in lines:
            words = line.split()
            if words:
                # First 2 words
                first_words.append(' '.join(words[:min(2, len(words))]))

        first_word_counts = Counter(first_words)
        same_starts = sum(count - 1 for count in first_word_counts.values() if count > 3)
        template_score_2 = min(same_starts / len(lines), 1.0)
    else:
        template_score_2 = 0.0

    # Strategy 3: Punctuation pattern similarity
    punct_patterns = []
    for line in lines:
        if not line.strip():
            continue
        # Extract punctuation pattern
        punct = ''.join(c for c in line if c in '.,!?;:-')
        punct_patterns.append(punct)

    punct_counts = Counter(punct_patterns)
    repeated_punct = sum(count - 1 for count in punct_counts.values() if count > 3)
    template_score_3 = min(repeated_punct / max(len(lines), 1), 1.0)

    # Combine scores
    return max(template_score_1, template_score_2, template_score_3)


def detect_filler_content(text: str) -> float:
    """
    Detect common filler phrases used for padding.

    Improved with expanded filler detection and weighting.
    """
    # Expanded filler phrases with categories
    transition_fillers = [
        "it is worth noting that",
        "it should be mentioned",
        "it is important to note",
        "it is essential to understand",
        "one could say that",
        "in other words",
        "that being said",
        "that said",
        "as mentioned before",
        "as previously stated",
        "as stated earlier",
        "to reiterate",
        "to repeat",
        "once again",
        "in summary",
        "to summarize",
    ]

    redundant_intensifiers = [
        "furthermore",
        "additionally",
        "moreover",
        "in addition",
        "also",
        "as well",
        "similarly",
        "likewise",
    ]

    vague_qualifiers = [
        "it seems that",
        "it appears that",
        "one might say",
        "arguably",
        "potentially",
        "possibly",
        "perhaps",
        "in a sense",
        "to some extent",
    ]

    hedging_phrases = [
        "it could be argued that",
        "some might argue that",
        "it is possible that",
        "there is a possibility that",
        "it may be that",
    ]

    text_lower = text.lower()

    # Count different types of fillers with weights
    transition_count = sum(text_lower.count(phrase) for phrase in transition_fillers)
    intensifier_count = sum(text_lower.count(phrase) for phrase in redundant_intensifiers)
    qualifier_count = sum(text_lower.count(phrase) for phrase in vague_qualifiers)
    hedging_count = sum(text_lower.count(phrase) for phrase in hedging_phrases)

    # Weight different filler types
    weighted_filler_count = (
        transition_count * 2.0 +      # Transition fillers are strong signals
        intensifier_count * 1.0 +     # Moderate signal
        qualifier_count * 1.5 +       # Strong signal of hedging
        hedging_count * 2.0           # Very strong signal
    )

    # Normalize based on text length
    words = len(text.split())
    if words > 0:
        filler_density = weighted_filler_count / (words / 100)  # Per 100 words
        return min(filler_density / 5, 1.0)  # Normalize to 0-1

    return 0.0


def detect_length_gaming(text: str) -> float:
    """
    Detect artificial lengthening strategies.

    Improved with multiple detection strategies.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 5:
        return 0.0

    penalties = []

    # Strategy 1: Uniform line length (suspiciously consistent)
    line_lengths = [len(line.split()) for line in lines]
    if len(line_lengths) > 10:
        if HAS_NUMPY:
            std_dev = np.std(line_lengths)
            mean_length = np.mean(line_lengths)
        else:
            mean_length = statistics.mean(line_lengths)
            std_dev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0

        # Low variance in line length = potential gaming
        if mean_length > 5:
            cv = std_dev / mean_length if mean_length > 0 else 0  # Coefficient of variation
            if cv < 0.2:  # Very uniform
                penalties.append(0.6)
            elif cv < 0.3:
                penalties.append(0.3)

    # Strategy 2: Quality degradation (good start, then repetition)
    if len(lines) > 20:
        mid_point = len(lines) // 2
        first_half_unique = len(set(lines[:mid_point]))
        second_half_unique = len(set(lines[mid_point:]))

        # If second half has much less diversity
        diversity_ratio = second_half_unique / max(first_half_unique, 1)
        if diversity_ratio < 0.4:
            penalties.append(0.8)
        elif diversity_ratio < 0.6:
            penalties.append(0.4)

    # Strategy 3: Detect "list padding" (bullet points with minimal content)
    list_markers = sum(1 for line in lines if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')))
    if list_markers > len(lines) * 0.6:  # More than 60% are list items
        avg_list_item_length = statistics.mean([len(line.split()) for line in lines if line.strip().startswith(('-', '*', '•'))])
        if avg_list_item_length < 5:  # Very short list items
            penalties.append(0.5)

    # Strategy 4: Detect repeated sentence structures
    sentence_starts = []
    for line in lines:
        words = line.split()
        if len(words) >= 2:
            sentence_starts.append(' '.join(words[:2]).lower())

    if len(sentence_starts) > 10:
        start_counts = Counter(sentence_starts)
        most_common_count = max(start_counts.values()) if start_counts else 0
        if most_common_count > len(sentence_starts) * 0.3:  # 30% start the same way
            penalties.append(0.6)

    # Strategy 5: Detect artificial paragraph breaks (same content, just split)
    if len(lines) > 15:
        # Check if adjacent lines are very similar
        similar_adjacent = 0
        for i in range(len(lines) - 1):
            words1 = set(lines[i].lower().split())
            words2 = set(lines[i + 1].lower().split())
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                if overlap > 0.7:  # 70% word overlap
                    similar_adjacent += 1

        if similar_adjacent > len(lines) * 0.3:
            penalties.append(0.5)

    return max(penalties) if penalties else 0.0


def detect_circular_reasoning(text: str) -> float:
    """
    NEW: Detect circular reasoning patterns where the same ideas are restated.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]

    if len(sentences) < 5:
        return 0.0

    # Create sentence embeddings (simple: word overlap)
    circular_score = 0
    for i in range(len(sentences) - 2):
        words_i = set(sentences[i].lower().split())
        # Check next 2-3 sentences for similar content
        for j in range(i + 1, min(i + 4, len(sentences))):
            words_j = set(sentences[j].lower().split())
            if words_i and words_j:
                overlap = len(words_i & words_j) / len(words_i | words_j)
                if overlap > 0.6:  # High overlap = circular reasoning
                    circular_score += 1

    return min(circular_score / max(len(sentences), 1), 1.0)


def detect_keyword_stuffing(text: str) -> float:
    """
    NEW: Detect keyword stuffing (using specific words excessively).
    """
    words = text.lower().split()
    if len(words) < 10:
        return 0.0

    # Filter common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'it', 'that', 'this'}
    content_words = [w for w in words if w not in common_words and len(w) > 3]

    if not content_words:
        return 0.0

    word_counts = Counter(content_words)
    total_content = len(content_words)

    # Check if any word is used excessively
    stuffing_score = 0
    for word, count in word_counts.most_common(5):  # Top 5 words
        frequency = count / total_content
        if frequency > 0.1:  # More than 10% of content words
            stuffing_score += (frequency - 0.1) * 10

    return min(stuffing_score, 1.0)


def detect_padding_sentences(text: str) -> float:
    """
    NEW: Detect sentences that add no information (pure padding).
    """
    padding_indicators = [
        r"^this (is|shows|demonstrates|illustrates|proves)",
        r"^(as|like) (we|you|i) (can|could|may|might) see",
        r"^it is (clear|obvious|evident|apparent) that",
        r"^(therefore|thus|hence|consequently|accordingly)",
        r"^in (conclusion|summary|brief|short)",
        r"^(another|one more|yet another) (thing|point|aspect|factor)",
    ]

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    padding_count = 0
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for pattern in padding_indicators:
            if re.match(pattern, sentence_lower):
                padding_count += 1
                break

    return min(padding_count / len(sentences), 1.0)


def advanced_repetition_detector(text: str) -> Dict[str, float]:
    """
    Comprehensive repetition detection including all strategies.
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

    # Calculate overall score (weighted combination)
    overall_score = (
        penalties['word_rep'] * 0.1 +
        penalties['phrase_rep'] * 0.15 +
        penalties['para_rep'] * 0.2 +
        penalties['template_score'] * 0.15 +
        penalties['filler_score'] * 0.1 +
        penalties['length_gaming'] * 0.15 +
        penalties['circular_reasoning'] * 0.1 +
        penalties['keyword_stuffing'] * 0.03 +
        penalties['padding_sentences'] * 0.02
    )

    penalties['overall_score'] = min(overall_score, 1.0)

    return penalties


# ============================================================================
# Test Cases
# ============================================================================

TEST_CASES = {
    # Good quality text (should have LOW penalties)
    "good_quality": """
        Machine learning is a subset of artificial intelligence that enables systems to learn from data.
        Neural networks are computational models inspired by biological neural networks.
        Deep learning uses multiple layers to progressively extract higher-level features.
        Training data quality significantly impacts model performance.
        Regularization techniques help prevent overfitting in complex models.
    """,

    # Word repetition
    "word_repetition": """
        The model model model uses data data data to learn learn learn patterns patterns patterns.
        Training training training requires requires requires computational computational computational resources resources resources.
        Performance performance performance depends depends depends on on on data data data quality quality quality.
    """,

    # Phrase repetition
    "phrase_repetition": """
        It is worth noting that the model performs well.
        It is worth noting that the training was successful.
        It is worth noting that the data quality is good.
        It is worth noting that the results are promising.
        It is worth noting that the approach is novel.
    """,

    # Paragraph repetition
    "paragraph_repetition": """
        The model achieves good performance on the test set.
        The model achieves good performance on the test set.
        Training converged after 100 epochs.
        The model achieves good performance on the test set.
        The model achieves good performance on the test set.
    """,

    # Template-based generation
    "template_abuse": """
        Machine learning is important because it enables automation.
        Deep learning is important because it improves accuracy.
        Data preprocessing is important because it ensures quality.
        Model evaluation is important because it validates performance.
        Hyperparameter tuning is important because it optimizes results.
        Feature engineering is important because it enhances predictions.
    """,

    # Filler content
    "filler_heavy": """
        It is worth noting that machine learning is useful. Furthermore, it should be mentioned that
        deep learning is powerful. Additionally, one could say that neural networks are effective.
        Moreover, it is important to note that data quality matters. That being said, it appears that
        model selection is crucial. In other words, as mentioned before, these factors are essential.
    """,

    # Length gaming - uniform lines
    "length_gaming_uniform": """
        This is a sentence with exactly ten words in it.
        Here is another sentence with exactly ten words total.
        Yet another sentence that has exactly ten words here.
        One more sentence containing exactly ten words in total.
        Final sentence here with exactly ten words in it.
        This is a sentence with exactly ten words in it.
        Here is another sentence with exactly ten words total.
        Yet another sentence that has exactly ten words here.
        One more sentence containing exactly ten words in total.
        Final sentence here with exactly ten words in it.
    """,

    # Length gaming - quality degradation
    "quality_degradation": """
        Machine learning represents a paradigm shift in how we approach complex computational problems.
        The integration of neural networks with large-scale datasets has revolutionized pattern recognition.
        Sophisticated architectures enable unprecedented performance across diverse domains.
        Training requires careful consideration of multiple hyperparameters and validation strategies.

        The model works well. The model works well. The model works well.
        It is good. It is good. It is good. It is good.
        Training done. Training done. Training done. Training done.
        Results okay. Results okay. Results okay. Results okay.
        Model trained. Model trained. Model trained. Model trained.
    """,

    # List padding
    "list_padding": """
        Machine learning includes:
        - AI
        - ML
        - DL
        - NN
        - NLP
        - CV
        - RL
        - GAN
        - CNN
        - RNN
        - LSTM
        - GRU
        - BERT
        - GPT
        - T5
    """,

    # Circular reasoning
    "circular_reasoning": """
        The model performs well because it achieves good accuracy.
        Good accuracy is achieved because the model performs well.
        The performance is excellent due to high accuracy scores.
        High accuracy scores indicate excellent performance levels.
        Excellent performance is shown by good accuracy results.
    """,

    # Keyword stuffing
    "keyword_stuffing": """
        Machine learning machine learning is about machine learning models and machine learning algorithms.
        Machine learning machine learning techniques use machine learning machine learning approaches.
        Machine learning machine learning systems implement machine learning machine learning methods.
        Machine learning machine learning tools provide machine learning machine learning solutions.
    """,

    # Padding sentences
    "padding_sentences": """
        As we can see, the model is effective.
        It is clear that the results are good.
        Therefore, we can conclude the approach works.
        In conclusion, the method is successful.
        Another thing to note is the performance.
        Yet another aspect is the accuracy.
        One more point is the efficiency.
    """,

    # Mixed abuse (realistic gaming attempt)
    "mixed_abuse": """
        It is worth noting that machine learning is important. Furthermore, machine learning enables
        automation and machine learning improves efficiency. Additionally, machine learning provides
        solutions and machine learning delivers results.

        Machine learning is crucial because it enables automation.
        Deep learning is crucial because it improves accuracy.
        Data quality is crucial because it ensures performance.
        Model selection is crucial because it optimizes results.

        As we can see, the approach is effective. It is clear that the method works well.
        Therefore, we can conclude that the system is successful. In summary, the results are good.

        The model performs well. The model performs well. The model performs well.
        Training is complete. Training is complete. Training is complete.
    """,
}


# ============================================================================
# Testing Framework
# ============================================================================

def run_tests():
    """Run all test cases and display results."""
    print("="*80)
    print("ADVANCED REPETITION DETECTION - TEST RESULTS")
    print("="*80)
    print()

    results = {}

    for test_name, test_text in TEST_CASES.items():
        print(f"\n{'-'*80}")
        print(f"Test: {test_name.upper()}")
        print(f"{'-'*80}")

        penalties = advanced_repetition_detector(test_text)
        results[test_name] = penalties

        # Display results
        print(f"\nDetailed Scores:")
        print(f"  Word Repetition:      {penalties['word_rep']:.3f}")
        print(f"  Phrase Repetition:    {penalties['phrase_rep']:.3f}")
        print(f"  Paragraph Repetition: {penalties['para_rep']:.3f}")
        print(f"  Template Score:       {penalties['template_score']:.3f}")
        print(f"  Filler Score:         {penalties['filler_score']:.3f}")
        print(f"  Length Gaming:        {penalties['length_gaming']:.3f}")
        print(f"  Circular Reasoning:   {penalties['circular_reasoning']:.3f}")
        print(f"  Keyword Stuffing:     {penalties['keyword_stuffing']:.3f}")
        print(f"  Padding Sentences:    {penalties['padding_sentences']:.3f}")
        print(f"\n  >>> OVERALL SCORE:    {penalties['overall_score']:.3f} <<<")

        # Interpretation
        if penalties['overall_score'] < 0.2:
            quality = "✓ GOOD QUALITY"
        elif penalties['overall_score'] < 0.4:
            quality = "⚠ MODERATE ISSUES"
        elif penalties['overall_score'] < 0.6:
            quality = "⚠⚠ SIGNIFICANT ISSUES"
        else:
            quality = "✗ SEVERE QUALITY PROBLEMS"

        print(f"\n  Quality Assessment: {quality}")

    return results


def test_calibration():
    """Test calibration and suggest threshold values."""
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS")
    print("="*80)

    results = {}
    for test_name, test_text in TEST_CASES.items():
        penalties = advanced_repetition_detector(test_text)
        results[test_name] = penalties['overall_score']

    # Expected good vs bad classification
    good_texts = ['good_quality']
    bad_texts = [k for k in TEST_CASES.keys() if k not in good_texts]

    good_scores = [results[k] for k in good_texts]
    bad_scores = [results[k] for k in bad_texts]

    print(f"\nGood Quality Texts (should be < 0.2):")
    for k in good_texts:
        print(f"  {k}: {results[k]:.3f}")

    print(f"\nBad Quality Texts (should be > 0.3):")
    for k in bad_texts:
        print(f"  {k}: {results[k]:.3f}")

    # Suggest thresholds
    if good_scores and bad_scores:
        max_good = max(good_scores)
        min_bad = min(bad_scores)

        print(f"\nThreshold Recommendations:")
        print(f"  Max good score: {max_good:.3f}")
        print(f"  Min bad score:  {min_bad:.3f}")
        print(f"  Suggested thresholds:")
        print(f"    - High quality:     score < {max_good + 0.05:.2f}")
        print(f"    - Acceptable:       score < {(max_good + min_bad) / 2:.2f}")
        print(f"    - Reject/penalize:  score >= {min_bad - 0.05:.2f}")


def compare_detectors():
    """Compare individual detector effectiveness."""
    print("\n" + "="*80)
    print("DETECTOR EFFECTIVENESS COMPARISON")
    print("="*80)

    detector_results = {
        'word_rep': {},
        'phrase_rep': {},
        'para_rep': {},
        'template_score': {},
        'filler_score': {},
        'length_gaming': {},
        'circular_reasoning': {},
        'keyword_stuffing': {},
        'padding_sentences': {},
    }

    # Run all tests
    for test_name, test_text in TEST_CASES.items():
        penalties = advanced_repetition_detector(test_text)
        for detector in detector_results.keys():
            detector_results[detector][test_name] = penalties[detector]

    # Analyze which detector catches which problem
    print("\nDetector Performance Matrix:")
    print(f"{'Test Case':<25}", end='')
    for detector in detector_results.keys():
        print(f"{detector:<10}", end='')
    print()
    print("-" * 115)

    for test_name in TEST_CASES.keys():
        print(f"{test_name:<25}", end='')
        for detector in detector_results.keys():
            score = detector_results[detector][test_name]
            if score > 0.5:
                print(f"{'✗ ' + f'{score:.2f}':<10}", end='')
            elif score > 0.3:
                print(f"{'⚠ ' + f'{score:.2f}':<10}", end='')
            elif score > 0.1:
                print(f"{'· ' + f'{score:.2f}':<10}", end='')
            else:
                print(f"{'✓ ' + f'{score:.2f}':<10}", end='')
        print()


if __name__ == "__main__":
    # Run all tests
    results = run_tests()

    # Calibration analysis
    test_calibration()

    # Compare detectors
    compare_detectors()

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
