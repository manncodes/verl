#!/usr/bin/env python3
"""
Compare original vs improved repetition detectors.
"""

import sys
sys.path.insert(0, '/home/user/verl')

from test_repetition_detection import TEST_CASES, advanced_repetition_detector as original_detector
from repetition_detection_improved import advanced_repetition_detector as improved_detector, get_quality_assessment


def compare_versions():
    """Compare original vs improved detector performance."""
    print("="*100)
    print("ORIGINAL VS IMPROVED DETECTOR COMPARISON")
    print("="*100)
    print()

    print(f"{'Test Case':<25} {'Original':<12} {'Improved':<12} {'Improvement':<15} {'Assessment':<20}")
    print("-"*100)

    improvements = []
    regressions = []

    # Expected behavior: good texts should have low scores, bad texts high scores
    good_texts = {'good_quality'}
    bad_texts = set(TEST_CASES.keys()) - good_texts

    for test_name, test_text in TEST_CASES.items():
        original_result = original_detector(test_text)
        improved_result = improved_detector(test_text, return_components=True)

        original_score = original_result['overall_score']
        improved_score = improved_result['overall_score']

        diff = improved_score - original_score
        assessment = get_quality_assessment(improved_score)

        # Determine if this is an improvement
        is_good = test_name in good_texts
        if is_good:
            # For good text, lower score is better
            if improved_score < original_score:
                change_marker = "✓ BETTER"
                improvements.append((test_name, diff))
            elif improved_score > original_score:
                change_marker = "✗ WORSE"
                regressions.append((test_name, diff))
            else:
                change_marker = "= SAME"
        else:
            # For bad text, higher score is better
            if improved_score > original_score:
                change_marker = "✓ BETTER"
                improvements.append((test_name, diff))
            elif improved_score < original_score:
                change_marker = "✗ WORSE"
                regressions.append((test_name, diff))
            else:
                change_marker = "= SAME"

        print(f"{test_name:<25} {original_score:>6.3f}       {improved_score:>6.3f}       "
              f"{change_marker:<15} {assessment:<20}")

    print()
    print("="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Improvements: {len(improvements)}")
    print(f"Regressions: {len(regressions)}")
    print()

    # Detailed analysis
    print("Expected Behavior Check:")
    print("-"*50)

    # Good texts should have low scores
    good_scores_original = [original_detector(TEST_CASES[name])['overall_score'] for name in good_texts]
    good_scores_improved = [improved_detector(TEST_CASES[name], return_components=False)['overall_score'] for name in good_texts]

    print(f"Good quality texts (should be < 0.15):")
    print(f"  Original avg: {sum(good_scores_original)/len(good_scores_original):.3f}")
    print(f"  Improved avg: {sum(good_scores_improved)/len(good_scores_improved):.3f}")
    print()

    # Bad texts should have high scores
    bad_scores_original = [original_detector(TEST_CASES[name])['overall_score'] for name in bad_texts]
    bad_scores_improved = [improved_detector(TEST_CASES[name], return_components=False)['overall_score'] for name in bad_texts]

    print(f"Bad quality texts (should be > 0.30):")
    print(f"  Original avg: {sum(bad_scores_original)/len(bad_scores_original):.3f}")
    print(f"  Improved avg: {sum(bad_scores_improved)/len(bad_scores_improved):.3f}")
    print()

    # Check separation
    original_separation = sum(bad_scores_original)/len(bad_scores_original) - sum(good_scores_original)/len(good_scores_original)
    improved_separation = sum(bad_scores_improved)/len(bad_scores_improved) - sum(good_scores_improved)/len(good_scores_improved)

    print(f"Separation (bad_avg - good_avg):")
    print(f"  Original: {original_separation:.3f}")
    print(f"  Improved: {improved_separation:.3f}")
    print(f"  Change: {'+' if improved_separation > original_separation else ''}{improved_separation - original_separation:.3f}")
    print()

    # Check classification accuracy
    print("Classification Accuracy:")
    print("-"*50)

    # Using threshold of 0.30
    threshold = 0.30

    original_correct = 0
    improved_correct = 0
    total = len(TEST_CASES)

    for test_name in TEST_CASES.keys():
        is_good = test_name in good_texts

        original_score = original_detector(TEST_CASES[test_name])['overall_score']
        improved_score = improved_detector(TEST_CASES[test_name], return_components=False)['overall_score']

        # Good should be < threshold, bad should be >= threshold
        if is_good:
            if original_score < threshold:
                original_correct += 1
            if improved_score < threshold:
                improved_correct += 1
        else:
            if original_score >= threshold:
                original_correct += 1
            if improved_score >= threshold:
                improved_correct += 1

    print(f"Using threshold = {threshold}")
    print(f"  Original accuracy: {original_correct}/{total} ({original_correct/total*100:.1f}%)")
    print(f"  Improved accuracy: {improved_correct}/{total} ({improved_correct/total*100:.1f}%)")
    print()


def show_detailed_comparison(test_name: str):
    """Show detailed component comparison for a specific test."""
    if test_name not in TEST_CASES:
        print(f"Test '{test_name}' not found!")
        return

    test_text = TEST_CASES[test_name]

    print("="*80)
    print(f"DETAILED COMPARISON: {test_name.upper()}")
    print("="*80)
    print()

    original_result = original_detector(test_text)
    improved_result = improved_detector(test_text, return_components=True)

    components = [
        'word_rep', 'phrase_rep', 'para_rep', 'template_score',
        'filler_score', 'length_gaming', 'circular_reasoning',
        'keyword_stuffing', 'padding_sentences'
    ]

    print(f"{'Component':<25} {'Original':<12} {'Improved':<12} {'Change':<12}")
    print("-"*65)

    for component in components:
        original_val = original_result.get(component, 0)
        improved_val = improved_result.get(component, 0)
        change = improved_val - original_val

        change_str = f"{'+' if change > 0 else ''}{change:.3f}"
        print(f"{component:<25} {original_val:>6.3f}       {improved_val:>6.3f}       {change_str:>10}")

    print("-"*65)
    print(f"{'OVERALL':<25} {original_result['overall_score']:>6.3f}       "
          f"{improved_result['overall_score']:>6.3f}       "
          f"{'+' if improved_result['overall_score'] > original_result['overall_score'] else ''}"
          f"{improved_result['overall_score'] - original_result['overall_score']:>.3f}")
    print()


if __name__ == "__main__":
    # Overall comparison
    compare_versions()

    # Detailed comparisons for key test cases
    print("\n" + "="*80)
    print("DETAILED COMPONENT ANALYSIS")
    print("="*80)

    key_tests = ['good_quality', 'filler_heavy', 'length_gaming_uniform', 'mixed_abuse']

    for test_name in key_tests:
        show_detailed_comparison(test_name)
        print()
