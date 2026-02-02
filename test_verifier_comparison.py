#!/usr/bin/env python3
"""
Standalone test to compare math verifiers without full verl dependencies.
"""

import sys
import os
import types
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module(name, path):
    """Load a Python module from a file path."""
    full_path = os.path.join(BASE_DIR, path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    return module, spec

# Set up mock package structure
sys.modules['verl'] = types.ModuleType('verl')
sys.modules['verl.utils'] = types.ModuleType('verl.utils')
sys.modules['verl.utils.reward_score'] = types.ModuleType('verl.utils.reward_score')
sys.modules['verl.utils.reward_score.math_verify'] = types.ModuleType('verl.utils.reward_score.math_verify')

# Load modules in dependency order
# 1. Load normalize first (no internal deps)
normalize_mod, normalize_spec = load_module(
    'verl.utils.reward_score.math_verify.normalize',
    'verl/utils/reward_score/math_verify/normalize.py'
)
normalize_spec.loader.exec_module(normalize_mod)

# 2. Load core (no internal deps, but needs normalize for runtime)
core_mod, core_spec = load_module(
    'verl.utils.reward_score.math_verify.core',
    'verl/utils/reward_score/math_verify/core.py'
)
core_spec.loader.exec_module(core_mod)

# 3. Load extract (depends on core)
extract_mod, extract_spec = load_module(
    'verl.utils.reward_score.math_verify.extract',
    'verl/utils/reward_score/math_verify/extract.py'
)
extract_spec.loader.exec_module(extract_mod)

# 4. Load compare (depends on core, normalize)
compare_mod, compare_spec = load_module(
    'verl.utils.reward_score.math_verify.compare',
    'verl/utils/reward_score/math_verify/compare.py'
)
compare_spec.loader.exec_module(compare_mod)

# Create shortcuts
MathVerifier = core_mod.MathVerifier
VerifierConfig = core_mod.VerifierConfig
ExtractionMethod = core_mod.ExtractionMethod
ComparisonMethod = core_mod.ComparisonMethod
extract_boxed = extract_mod.extract_boxed
extract_answer = extract_mod.extract_answer
compare_numeric = compare_mod.compare_numeric
compare_strings = compare_mod.compare_strings
normalize_latex = normalize_mod.normalize_latex

# Load other verifiers for comparison
def load_verifier(name, path):
    """Try to load a verifier module."""
    try:
        full_path = os.path.join(BASE_DIR, path)
        spec = importlib.util.spec_from_file_location(name, full_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Could not load {name}: {e}")
        return None

math_reward = load_verifier('math_reward', 'verl/utils/reward_score/math_reward.py')
math_dapo = load_verifier('math_dapo', 'verl/utils/reward_score/math_dapo.py')

# Test cases: (solution, ground_truth, expected_correct)
TEST_CASES = [
    # Basic boxed integers
    (r"The answer is \boxed{42}", "42", True),
    (r"\boxed{-7}", "-7", True),
    (r"\boxed{0}", "0", True),

    # Fractions - string match
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", True),

    # Fractions - numeric equivalence
    (r"\boxed{\frac{1}{2}}", "0.5", True),
    (r"\boxed{1/2}", "0.5", True),

    # dfrac/tfrac normalization
    (r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}", True),
    (r"\boxed{\tfrac{3}{4}}", r"\frac{3}{4}", True),

    # Decimals
    (r"\boxed{3.14159}", "3.14159", True),
    (r"\boxed{0.5}", "0.5", True),

    # Wrong answers
    (r"\boxed{41}", "42", False),
    (r"\boxed{wrong}", "42", False),

    # Multiple boxed (should use last)
    (r"\boxed{wrong} then \boxed{correct}", "correct", True),
    (r"\boxed{1} \boxed{2} \boxed{3}", "3", True),

    # Chain of thought
    (r"""
    Let's solve step by step.
    First, x = 2
    Then x^2 = 4
    Therefore, the answer is \boxed{4}
    """, "4", True),

    # Nested fractions
    (r"\boxed{\frac{1}{\sqrt{2}}}", r"\frac{1}{\sqrt{2}}", True),

    # Whitespace handling
    (r"\boxed{ 42 }", "42", True),
    (r"\boxed{  3.14  }", "3.14", True),

    # Negative numbers
    (r"\boxed{-42}", "-42", True),

    # Large numbers
    (r"\boxed{1000000}", "1000000", True),

    # Empty boxed
    (r"\boxed{}", "", True),

    # No boxed - should use pattern/fallback extraction
    ("The answer is: 42", "42", True),

    # === NEW: Features from prime_math ===

    # Thousands separator handling
    (r"\boxed{1,000,000}", "1000000", True),

    # Tuples/lists
    (r"\boxed{(1, 2, 3)}", "(1, 2, 3)", True),

    # Leading decimal normalization
    (r"\boxed{.5}", "0.5", True),
]

def test_our_verifier(solution, gt):
    """Test with our new math_verify module."""
    try:
        verifier = MathVerifier()
        result = verifier.verify(solution, gt)
        return result.correct, result.extraction_method.name, result.comparison_method.name
    except Exception as e:
        return None, "ERROR", str(e)

def test_math_reward(solution, gt):
    """Test with math_reward.py (EleutherAI-based)."""
    if math_reward is None:
        return None, "N/A", "module not loaded"
    try:
        score = math_reward.compute_score(solution, gt)
        return score > 0, "boxed", "string"
    except Exception as e:
        return None, "ERROR", str(e)

def test_math_dapo(solution, gt):
    """Test with math_dapo.py (using strict_box_verify for boxed answers)."""
    if math_dapo is None:
        return None, "N/A", "module not loaded"
    try:
        # Use strict_box_verify=True to extract from \boxed{} format
        result = math_dapo.compute_score(solution, gt, strict_box_verify=True)
        if isinstance(result, dict):
            correct = result.get('acc', result.get('score', 0)) > 0
        else:
            correct = result > 0
        return correct, "boxed", "normalized"
    except Exception as e:
        return None, "ERROR", str(e)

def run_comparison():
    """Run comparison tests across all verifiers."""
    print("=" * 90)
    print("MATH VERIFIER COMPARISON TEST")
    print("=" * 90)
    print()

    # Results storage
    results = []

    for i, (solution, gt, expected) in enumerate(TEST_CASES):
        our_correct, our_ext, our_cmp = test_our_verifier(solution, gt)
        mr_correct, mr_ext, mr_cmp = test_math_reward(solution, gt)
        md_correct, md_ext, md_cmp = test_math_dapo(solution, gt)

        results.append({
            'idx': i,
            'solution': solution[:50].replace('\n', ' ').strip(),
            'gt': gt,
            'expected': expected,
            'math_verify_new': our_correct,
            'math_reward': mr_correct,
            'math_dapo': md_correct,
        })

    # Print summary table
    print(f"{'#':<3} {'Solution':<52} {'GT':<15} {'Exp':<5} {'New':<5} {'Rew':<5} {'DAPO':<5}")
    print("-" * 90)

    disagreements = []
    for r in results:
        sol_short = r['solution'][:50] + '..' if len(r['solution']) > 50 else r['solution']
        gt_short = r['gt'][:13] + '..' if len(r['gt']) > 13 else r['gt']

        # Format booleans
        def fmt(v):
            if v is None: return "N/A"
            return "T" if v else "F"

        # Check for disagreements among available verifiers
        verifiers = [r['math_verify_new'], r['math_reward'], r['math_dapo']]
        verifiers = [v for v in verifiers if v is not None]
        has_disagreement = len(set(verifiers)) > 1

        if has_disagreement:
            disagreements.append(r)

        marker = " **" if has_disagreement else ""
        exp_match = "OK" if r['math_verify_new'] == r['expected'] else "!!"

        print(f"{r['idx']:<3} {sol_short:<52} {gt_short:<15} {fmt(r['expected']):<5} "
              f"{fmt(r['math_verify_new']):<5} {fmt(r['math_reward']):<5} {fmt(r['math_dapo']):<5}{marker}")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Count results
    def count_correct(key):
        correct = sum(1 for r in results if r[key] == r['expected'])
        total = sum(1 for r in results if r[key] is not None)
        return correct, total

    new_c, new_t = count_correct('math_verify_new')
    mr_c, mr_t = count_correct('math_reward')
    md_c, md_t = count_correct('math_dapo')

    print(f"\nAccuracy vs expected:")
    print(f"  math_verify (new):  {new_c}/{new_t} correct ({100*new_c/new_t:.1f}%)" if new_t > 0 else "  math_verify (new): N/A")
    print(f"  math_reward:        {mr_c}/{mr_t} correct ({100*mr_c/mr_t:.1f}%)" if mr_t > 0 else "  math_reward: N/A")
    print(f"  math_dapo:          {md_c}/{md_t} correct ({100*md_c/md_t:.1f}%)" if md_t > 0 else "  math_dapo: N/A")

    print(f"\nDisagreements between verifiers: {len(disagreements)}")

    if disagreements:
        print("\nDisagreement details:")
        for r in disagreements:
            print(f"\n  Case {r['idx']}: gt='{r['gt']}'")
            print(f"    Solution: '{r['solution'][:60]}...'")
            print(f"    Expected: {r['expected']}")
            print(f"    math_verify_new: {r['math_verify_new']}")
            print(f"    math_reward:     {r['math_reward']}")
            print(f"    math_dapo:       {r['math_dapo']}")

    print()
    print("=" * 90)
    print("COMPONENT TESTS")
    print("=" * 90)

    # Test individual components
    print("\n1. Boxed Extraction:")
    test_extractions = [
        (r"\boxed{42}", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"First \boxed{wrong} then \boxed{right}", "right"),
        (r"\boxed{\frac{1}{\sqrt{2+\frac{1}{3}}}}", r"\frac{1}{\sqrt{2+\frac{1}{3}}}"),
    ]
    for text, expected in test_extractions:
        result = extract_boxed(text)
        status = "PASS" if result.answer == expected else "FAIL"
        text_short = text[:45] + '...' if len(text) > 45 else text
        print(f"  [{status}] '{text_short}' -> '{result.answer}'")

    print("\n2. LaTeX Normalization:")
    test_normalizations = [
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"\tfrac{1}{2}", r"\frac{1}{2}"),
        (r"\frac12", r"\frac{1}{2}"),
        (r"\sqrt2", r"\sqrt{2}"),
    ]
    for expr, expected in test_normalizations:
        result = normalize_latex(expr)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{expr}' -> '{result}' (expected: '{expected}')")

    print("\n3. Numeric Comparison:")
    test_numerics = [
        ("0.5", "1/2", True),
        (r"\frac{1}{2}", "0.5", True),
        ("0.333333", "0.333334", True),
        ("1", "2", False),
        ("50%", "0.5", True),
    ]
    for pred, gt, expected in test_numerics:
        result = compare_numeric(pred, gt)
        status = "PASS" if result['match'] == expected else "FAIL"
        print(f"  [{status}] '{pred}' == '{gt}' -> {result['match']} (expected: {expected})")

    print("\n" + "=" * 90)
    print("TEST COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    run_comparison()
