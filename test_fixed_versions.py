"""
Test suite comparing original (buggy) and fixed versions of format_reward functions.
"""

from test_format_reward import format_reward as format_reward_original
from test_format_reward import format_reward_strict as format_reward_strict_original
from format_reward_fixed import format_reward, format_reward_strict, format_reward_v2


def test_bug_fix():
    """Test cases that expose the INCORRECT/CORRECT substring bug"""

    print("=" * 80)
    print("BUG FIX VALIDATION - Testing INCORRECT vs CORRECT")
    print("=" * 80)

    test_cases = [
        # The bug case - should reject INCORRECT
        (
            "<think>Thinking</think><evaluation>INCORRECT</evaluation><answer>42</answer>",
            0.0,
            "Should reject: Only INCORRECT (no CORRECT)"
        ),
        # Valid CORRECT cases
        (
            "<think>Thinking</think><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Should accept: CORRECT evaluation"
        ),
        # Edge cases with CORRECT in different contexts
        (
            "<think>Thinking</think><evaluation>This is CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Should accept: CORRECT with surrounding text"
        ),
        (
            "<think>Thinking</think><evaluation>NOT INCORRECT</evaluation><answer>42</answer>",
            0.0,
            "Should reject: No CORRECT word"
        ),
        # Retry pattern with INCORRECT then CORRECT
        (
            "<think>Try 1</think><evaluation>INCORRECT</evaluation><think>Try 2</think><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Should accept: Retry pattern (INCORRECT → CORRECT)"
        ),
        # Multiple INCORRECTs then CORRECT
        (
            "<think>A</think><evaluation>INCORRECT</evaluation><think>B</think><evaluation>INCORRECT</evaluation><answer>C</answer>",
            0.0,
            "Should reject: Multiple INCORRECT, no CORRECT"
        ),
        # Case sensitivity tests
        (
            "<think>Thinking</think><evaluation>incorrect</evaluation><answer>42</answer>",
            0.0,
            "Should reject: Only 'incorrect' (lowercase)"
        ),
        (
            "<think>Thinking</think><evaluation>correct</evaluation><answer>42</answer>",
            1.0,
            "Should accept: 'correct' (lowercase, case insensitive)"
        ),
    ]

    functions_to_test = [
        ("format_reward (ORIGINAL)", format_reward_original),
        ("format_reward (FIXED)", format_reward),
        ("format_reward_v2 (FIXED ALT)", format_reward_v2),
        ("format_reward_strict (ORIGINAL)", format_reward_strict_original),
        ("format_reward_strict (FIXED)", format_reward_strict),
    ]

    results = {name: {"passed": 0, "failed": 0} for name, _ in functions_to_test}

    for test_idx, (input_str, expected, description) in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {test_idx}: {description}")
        print(f"Expected: {expected}")
        print(f"Input: {input_str[:70]}...")
        print("-" * 80)

        for func_name, func in functions_to_test:
            result = func(input_str)
            status = "✓ PASS" if result == expected else "✗ FAIL"

            if result == expected:
                results[func_name]["passed"] += 1
            else:
                results[func_name]["failed"] += 1

            marker = "  " if result == expected else "❌"
            print(f"{marker} {func_name:35s}: {result:.1f} - {status}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY - Comparing Original vs Fixed Versions")
    print("=" * 80)
    print(f"{'Function':<40s} | Passed | Failed | Success Rate")
    print("-" * 80)

    for func_name, _ in functions_to_test:
        passed = results[func_name]["passed"]
        failed = results[func_name]["failed"]
        total = passed + failed
        rate = (passed / total * 100) if total > 0 else 0
        print(f"{func_name:<40s} | {passed:6d} | {failed:6d} | {rate:6.1f}%")

    print("=" * 80)


def test_comprehensive():
    """Run comprehensive tests on fixed versions"""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TESTS - Fixed Versions Only")
    print("=" * 80)

    test_cases = [
        # Valid patterns
        ("<think>A</think><evaluation>CORRECT</evaluation><answer>B</answer>", 1.0, "Simple CORRECT"),
        ("<think>A</think><evaluation>Result is CORRECT!</evaluation><answer>B</answer>", 1.0, "CORRECT with text"),
        ("<think>A</think><evaluation>correct</evaluation><answer>B</answer>", 1.0, "Lowercase correct"),
        ("<think><planning>P</planning>A</think><evaluation>CORRECT</evaluation><answer>B</answer>", 1.0, "With planning"),
        ("<think><draft answer>D</draft answer>A</think><evaluation>CORRECT</evaluation><answer>B</answer>", 1.0, "With draft"),
        ("<think>A</think><evaluation>INCORRECT</evaluation><think>B</think><evaluation>CORRECT</evaluation><answer>C</answer>", 1.0, "Retry pattern"),

        # Invalid patterns
        ("<think>A</think><evaluation>INCORRECT</evaluation><answer>B</answer>", 0.0, "Only INCORRECT"),
        ("<think>A</think><evaluation>WRONG</evaluation><answer>B</answer>", 0.0, "Neither CORRECT nor INCORRECT"),
        ("<think>A</think><answer>B</answer>", 0.0, "Missing evaluation"),
        ("<evaluation>CORRECT</evaluation><answer>B</answer>", 0.0, "Missing think"),
        ("<think>A</think><evaluation>CORRECT</evaluation>", 0.0, "Missing answer"),

        # Edge cases
        ("<think>INCORRECT attempt</think><evaluation>Now CORRECT</evaluation><answer>B</answer>", 1.0, "INCORRECT in think, CORRECT in eval"),
        ("<think>A</think><evaluation>The result is not INCORRECT, it's CORRECT</evaluation><answer>B</answer>", 1.0, "Both words in eval, CORRECT wins"),
    ]

    functions = [
        ("format_reward (FIXED)", format_reward),
        ("format_reward_v2 (FIXED ALT)", format_reward_v2),
        ("format_reward_strict (FIXED)", format_reward_strict),
    ]

    for func_name, func in functions:
        print(f"\n{'-' * 80}")
        print(f"Testing: {func_name}")
        print("-" * 80)

        passed = 0
        failed = 0

        for input_str, expected, description in test_cases:
            result = func(input_str)
            status = "✓" if result == expected else "✗"

            if result == expected:
                passed += 1
            else:
                failed += 1
                print(f"{status} FAIL: {description}")
                print(f"   Expected {expected}, got {result}")
                print(f"   Input: {input_str[:60]}...")

        print(f"\nResults: {passed}/{passed+failed} passed ({passed/(passed+failed)*100:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_bug_fix()
    test_comprehensive()

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("✗ Original versions incorrectly accept 'INCORRECT' as valid")
    print("  Reason: Regex matches 'CORRECT' substring inside 'INCORRECT'")
    print()
    print("✓ Fixed versions use word boundaries (\\b) to prevent this")
    print("  - format_reward: Uses \\bCORRECT\\b and \\bINCORRECT\\b")
    print("  - format_reward_v2: Uses negative lookbehind (?<!IN)CORRECT")
    print("=" * 80)
