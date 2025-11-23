#!/usr/bin/env python3
"""
Compare all calibration levels: Baseline → Moderate → Conservative

Shows how different calibrations handle the same test cases.
"""

import sys
sys.path.insert(0, '/home/user/verl')

from test_repetition_detection import TEST_CASES
from repetition_penalty_final import calculate_repetition_score as moderate_detector
from repetition_penalty_conservative import calculate_repetition_score_conservative as conservative_detector


def compare_all_calibrations():
    """Compare all three calibration levels."""
    print("="*100)
    print("CALIBRATION COMPARISON: MODERATE vs CONSERVATIVE")
    print("="*100)
    print()
    print("Philosophy:")
    print("  MODERATE: Balanced - catches most issues, some false positives acceptable")
    print("  CONSERVATIVE: High precision - only clear egregious cases, avoid false positives")
    print()
    print("="*100)
    print()

    results = {}

    print(f"{'Test Case':<30} {'Moderate':<15} {'Conservative':<15} {'Difference':<15}")
    print("-"*100)

    for test_name, test_text in TEST_CASES.items():
        moderate_result = moderate_detector(test_text, return_components=False)
        conservative_result = conservative_detector(test_text, return_components=False)

        moderate_score = moderate_result['score']
        conservative_score = conservative_result['score']

        diff = moderate_score - conservative_score

        results[test_name] = {
            'moderate': moderate_score,
            'conservative': conservative_score,
            'diff': diff
        }

        # Show difference indicator
        if abs(diff) < 0.05:
            diff_indicator = "≈ SIMILAR"
        elif diff > 0.2:
            diff_indicator = "↓↓ MUCH LOWER"
        elif diff > 0.1:
            diff_indicator = "↓ LOWER"
        else:
            diff_indicator = "· SLIGHTLY LOWER"

        print(f"{test_name:<30} {moderate_score:>6.3f}         {conservative_score:>6.3f}         "
              f"{diff_indicator:<15}")

    print()
    print("="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print()

    # Good quality texts
    good_texts = ['good_quality']
    print("Good Quality Texts (should score close to 0):")
    for name in good_texts:
        print(f"  {name}:")
        print(f"    Moderate:     {results[name]['moderate']:.3f}")
        print(f"    Conservative: {results[name]['conservative']:.3f}")
    print()

    # Potentially acceptable texts (minor issues)
    borderline_texts = ['word_repetition', 'filler_heavy', 'circular_reasoning',
                        'padding_sentences', 'quality_degradation']
    print("Borderline Cases (conservative should be lenient):")
    mod_borderline = [results[name]['moderate'] for name in borderline_texts if name in results]
    cons_borderline = [results[name]['conservative'] for name in borderline_texts if name in results]

    print(f"  Moderate avg:     {sum(mod_borderline)/len(mod_borderline):.3f}")
    print(f"  Conservative avg: {sum(cons_borderline)/len(cons_borderline):.3f}")
    print(f"  Reduction:        {(sum(mod_borderline)-sum(cons_borderline))/len(mod_borderline):.3f}")
    print()

    # Clearly bad texts (should still catch these)
    bad_texts = ['phrase_repetition', 'paragraph_repetition', 'template_abuse',
                 'length_gaming_uniform', 'keyword_stuffing', 'mixed_abuse']
    print("Clear Problems (conservative should still catch):")
    mod_bad = [results[name]['moderate'] for name in bad_texts if name in results]
    cons_bad = [results[name]['conservative'] for name in bad_texts if name in results]

    print(f"  Moderate avg:     {sum(mod_bad)/len(mod_bad):.3f}")
    print(f"  Conservative avg: {sum(cons_bad)/len(cons_bad):.3f}")
    print()

    # Count how many still get penalized
    print("Penalty Analysis (assuming threshold = 0.35 for conservative):")
    print("-"*50)

    moderate_penalized = sum(1 for name, r in results.items() if r['moderate'] >= 0.15)
    conservative_penalized = sum(1 for name, r in results.items() if r['conservative'] >= 0.35)

    print(f"  Moderate penalized (>= 0.15):     {moderate_penalized}/{len(results)}")
    print(f"  Conservative penalized (>= 0.35): {conservative_penalized}/{len(results)}")
    print()

    print("Zero-Penalty Cases (conservative):")
    for name, r in results.items():
        if r['conservative'] == 0.0:
            print(f"  ✓ {name:<30} (Moderate: {r['moderate']:.3f})")
    print()


def show_mode_comparison():
    """Show penalty multipliers for different modes."""
    print("="*100)
    print("PENALTY MULTIPLIER COMPARISON")
    print("="*100)
    print()

    from repetition_penalty_final import get_penalty_multiplier as get_moderate_mult
    from repetition_penalty_conservative import get_conservative_penalty_multiplier

    scores = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

    print(f"{'Score':<10} {'Moderate':<15} {'Conservative':<20} {'Ultra-Cons':<15}")
    print("-"*70)

    for score in scores:
        mod_mult = get_moderate_mult(score, "moderate")
        cons_mult = get_conservative_penalty_multiplier(score, "default")
        ultra_mult = get_conservative_penalty_multiplier(score, "ultra_conservative")

        print(f"{score:<10.2f} {mod_mult:<15.2f} {cons_mult:<20.2f} {ultra_mult:<15.2f}")

    print()
    print("Key Observations:")
    print("  - Conservative keeps 1.0x (no penalty) up to score 0.35")
    print("  - Moderate starts penalizing at 0.15")
    print("  - Ultra-conservative keeps 1.0x up to score 0.50")
    print()


def test_real_world_examples():
    """Test with more realistic examples."""
    print("="*100)
    print("REAL-WORLD EXAMPLES")
    print("="*100)
    print()

    from repetition_penalty_conservative import apply_repetition_penalty_conservative, diagnose_conservative

    examples = {
        "Natural variation": """
            The model performs well on various tasks. It achieves high accuracy
            across different domains. Performance metrics indicate strong results.
            The system demonstrates robust capabilities in multiple scenarios.
        """,

        "Some repetition (acceptable)": """
            Machine learning is powerful. Machine learning enables automation.
            We use machine learning for predictions. Machine learning improves
            over time with more data.
        """,

        "Template but informative": """
            Python is useful because it has extensive libraries.
            TensorFlow is useful because it simplifies deep learning.
            Docker is useful because it enables containerization.
        """,

        "Egregious repetition": """
            The model the model the model the model works works works works.
            The model the model the model the model works works works works.
            The model the model the model the model works works works works.
            The model the model the model the model works works works works.
        """,

        "Robotic template abuse": """
            X is important because it enables Y and provides Z.
            A is important because it enables B and provides C.
            D is important because it enables E and provides F.
            G is important because it enables H and provides I.
            J is important because it enables K and provides L.
            M is important because it enables N and provides O.
        """
    }

    for name, text in examples.items():
        print(f"\nExample: {name}")
        print("-"*70)

        # Test both
        from repetition_penalty_final import apply_repetition_penalty
        mod_reward, mod_details = apply_repetition_penalty(
            text, 1.0, severity="moderate", return_details=True
        )
        cons_reward, cons_details = apply_repetition_penalty_conservative(
            text, 1.0, mode="default", return_details=True
        )

        print(f"  Moderate:")
        print(f"    Score: {mod_details['repetition_score']:.3f}")
        print(f"    Multiplier: {mod_details['penalty_multiplier']:.3f}")
        print(f"    Final reward: {mod_reward:.3f}")

        print(f"  Conservative:")
        print(f"    Score: {cons_details['repetition_score']:.3f}")
        print(f"    Multiplier: {cons_details['penalty_multiplier']:.3f}")
        print(f"    Final reward: {cons_reward:.3f}")
        print(f"    Assessment: {cons_details['assessment']}")

        if cons_reward < mod_reward:
            print(f"  → Conservative is MORE LENIENT (+{cons_reward - mod_reward:.3f})")
        elif cons_reward > mod_reward:
            print(f"  → Conservative is STRICTER ({cons_reward - mod_reward:.3f})")
        else:
            print(f"  → Same penalty")


if __name__ == "__main__":
    # Overall comparison
    compare_all_calibrations()

    # Penalty multipliers
    show_mode_comparison()

    # Real-world examples
    test_real_world_examples()

    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)
    print("""
Use CONSERVATIVE calibration for your use case:
  - Keeps scores intact for borderline cases
  - Only penalizes truly egregious repetition
  - High precision, low false positive rate
  - Focuses on "inhumane" robotic responses

Usage:
  from repetition_penalty_conservative import apply_repetition_penalty_conservative

  final_reward = apply_repetition_penalty_conservative(
      text,
      base_reward,
      mode="default"  # or "ultra_conservative" for even more leniency
  )

Thresholds:
  - < 0.35: No penalty (benefit of doubt)
  - 0.35-0.50: 10% penalty (mild issue)
  - 0.50-0.65: 30% penalty (clear issue)
  - 0.65-0.80: 50% penalty (serious issue)
  - > 0.80: 80% penalty (egregious only)
    """)
    print("="*100)
