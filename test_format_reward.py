import re

def format_reward(predict_str: str) -> float:
    """
    Validates custom thinking formats with evaluation tags.
    Supports both direct correct evaluation and retry patterns.
    """

    # Pattern 1: Direct correct evaluation
    # <think>...</think><evaluation>...CORRECT...</evaluation><answer>...</answer>
    pattern_direct = re.compile(
        r'<think>'
        r'(?:.*?<planning>.*?</planning>)?'  # optional planning
        r'(?:.*?<draft answer>.*?</draft answer>)?'  # optional draft
        r'.*?</think>'
        r'.*?<evaluation>.*?CORRECT.*?</evaluation>'
        r'.*?<answer>.+?</answer>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern 2: Retry with incorrect then correct
    # <think>...</think><evaluation>INCORRECT</evaluation><think>...</think><evaluation>CORRECT</evaluation><answer>...</answer>
    pattern_retry = re.compile(
        r'<think>'
        r'(?:.*?<planning>.*?</planning>)?'
        r'(?:.*?<draft answer>.*?</draft answer>)?'
        r'.*?</think>'
        r'.*?<evaluation>.*?INCORRECT.*?</evaluation>'
        r'.*?<think>'
        r'(?:.*?<planning>.*?</planning>)?'
        r'(?:.*?<draft answer>.*?</draft answer>)?'
        r'.*?</think>'
        r'.*?<evaluation>.*?CORRECT.*?</evaluation>'
        r'.*?<answer>.+?</answer>',
        re.DOTALL | re.IGNORECASE
    )

    # Additional validation: ensure no duplicate opening tags where not expected
    duplicate_check = re.compile(
        r'<answer>.*<answer>|'  # no duplicate answer opens
        r'<evaluation>.*<evaluation>.*<evaluation>',  # max 2 evaluations
        re.DOTALL
    )

    # Check if string matches either valid pattern and has no invalid duplicates
    if duplicate_check.search(predict_str):
        return 0.0

    if pattern_direct.match(predict_str) or pattern_retry.match(predict_str):
        return 1.0

    return 0.0


# More robust version with additional validation
def format_reward_strict(predict_str: str) -> float:
    """
    Stricter validation with proper tag closure and content checks.
    """

    def validate_tag_balance(text: str, tag: str) -> bool:
        """Check if a tag is properly opened and closed."""
        open_count = text.count(f'<{tag}>')
        close_count = text.count(f'</{tag}>')
        return open_count == close_count and open_count > 0

    # Basic tag balance validation
    required_tags = ['think', 'answer']
    for tag in required_tags:
        if not validate_tag_balance(predict_str, tag):
            return 0.0

    # Check for evaluation tag with CORRECT (must exist at least once)
    if not re.search(r'<evaluation>.*?CORRECT.*?</evaluation>', predict_str, re.DOTALL | re.IGNORECASE):
        return 0.0

    # Validate sequence order
    try:
        # Find last CORRECT evaluation position
        correct_eval = list(re.finditer(r'<evaluation>.*?CORRECT.*?</evaluation>',
                                       predict_str, re.DOTALL | re.IGNORECASE))[-1]

        # Find answer position
        answer_match = re.search(r'<answer>.+?</answer>', predict_str, re.DOTALL)

        if not answer_match:
            return 0.0

        # Answer must come after the final CORRECT evaluation
        if answer_match.start() < correct_eval.end():
            return 0.0

        # Check for think tag before evaluation
        think_before = predict_str.rfind('</think>', 0, correct_eval.start())
        if think_before == -1:
            return 0.0

    except (IndexError, AttributeError):
        return 0.0

    return 1.0


# Test cases
def test_format_reward():
    print("=" * 80)
    print("Testing format_reward() function")
    print("=" * 80)

    test_cases = [
        # Valid cases - Direct correct evaluation
        (
            "<think>Let me solve this</think><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Simple direct correct pattern"
        ),
        (
            "<think><planning>Plan here</planning>Thinking...</think><evaluation>CORRECT</evaluation><answer>Result</answer>",
            1.0,
            "Direct correct with planning tag"
        ),
        (
            "<think><draft answer>Draft</draft answer>More thinking</think><evaluation>CORRECT</evaluation><answer>Final</answer>",
            1.0,
            "Direct correct with draft answer tag"
        ),
        (
            "<think><planning>Plan</planning><draft answer>Draft</draft answer>Think</think><evaluation>CORRECT</evaluation><answer>Answer</answer>",
            1.0,
            "Direct correct with both planning and draft"
        ),

        # Valid cases - Retry pattern
        (
            "<think>First attempt</think><evaluation>INCORRECT</evaluation><think>Second attempt</think><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Retry pattern: incorrect then correct"
        ),
        (
            "<think><planning>Plan 1</planning>Try 1</think><evaluation>INCORRECT</evaluation><think><planning>Plan 2</planning>Try 2</think><evaluation>CORRECT</evaluation><answer>Success</answer>",
            1.0,
            "Retry with planning in both attempts"
        ),

        # Invalid cases - Missing tags
        (
            "<think>Thinking</think><answer>42</answer>",
            0.0,
            "Missing evaluation tag"
        ),
        (
            "<evaluation>CORRECT</evaluation><answer>42</answer>",
            0.0,
            "Missing think tag"
        ),
        (
            "<think>Thinking</think><evaluation>CORRECT</evaluation>",
            0.0,
            "Missing answer tag"
        ),

        # Invalid cases - Wrong evaluation
        (
            "<think>Thinking</think><evaluation>INCORRECT</evaluation><answer>42</answer>",
            0.0,
            "Only INCORRECT evaluation (no CORRECT)"
        ),
        (
            "<think>Thinking</think><evaluation>WRONG</evaluation><answer>42</answer>",
            0.0,
            "Wrong evaluation text"
        ),

        # Invalid cases - Wrong order
        (
            "<evaluation>CORRECT</evaluation><think>Thinking</think><answer>42</answer>",
            0.0,
            "Evaluation before think"
        ),
        (
            "<answer>42</answer><think>Thinking</think><evaluation>CORRECT</evaluation>",
            0.0,
            "Answer at the beginning"
        ),

        # Invalid cases - Duplicate tags
        (
            "<think>Think</think><evaluation>CORRECT</evaluation><answer>First</answer><answer>Second</answer>",
            0.0,
            "Duplicate answer tags"
        ),
        (
            "<think>Think</think><evaluation>CORRECT</evaluation><evaluation>CORRECT</evaluation><evaluation>CORRECT</evaluation><answer>42</answer>",
            0.0,
            "Three evaluation tags (max 2 allowed)"
        ),

        # Edge cases
        (
            "<think></think><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Empty think tag content"
        ),
        (
            "<think>Thinking</think><evaluation>This is CORRECT!</evaluation><answer>42</answer>",
            1.0,
            "CORRECT embedded in text"
        ),
        (
            "<think>Thinking</think><evaluation>correct</evaluation><answer>42</answer>",
            1.0,
            "Lowercase 'correct' (case insensitive)"
        ),
        (
            "<think>Multi\nline\nthinking</think><evaluation>CORRECT</evaluation><answer>Multi\nline\nanswer</answer>",
            1.0,
            "Multiline content"
        ),

        # Complex valid cases
        (
            "<think>Let me solve this problem step by step\n<planning>1. Analyze\n2. Solve\n3. Verify</planning>\nNow solving...</think><evaluation>CORRECT - solution verified</evaluation><answer>The answer is 42</answer>",
            1.0,
            "Complex direct pattern with detailed content"
        ),
        (
            "<think>Try 1</think><evaluation>INCORRECT - wrong approach</evaluation><think>Try 2 with different method</think><evaluation>CORRECT - verified</evaluation><answer>Success</answer>",
            1.0,
            "Complex retry pattern with detailed evaluations"
        ),

        # Additional edge cases
        (
            "<think>Thinking</think><evaluation>CORRECT</evaluation><answer></answer>",
            0.0,
            "Empty answer tag (should fail - requires .+?)"
        ),
    ]

    passed = 0
    failed = 0

    for i, (input_str, expected, description) in enumerate(test_cases, 1):
        result = format_reward(input_str)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {i}: {description}")
        print(f"Expected: {expected}, Got: {result} - {status}")
        if result != expected:
            print(f"Input: {input_str[:100]}...")

    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"{'=' * 80}\n")

    return passed, failed


def test_format_reward_strict():
    print("=" * 80)
    print("Testing format_reward_strict() function")
    print("=" * 80)

    test_cases = [
        # Valid cases
        (
            "<think>Solving</think><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Simple valid pattern"
        ),
        (
            "<think>Try 1</think><evaluation>INCORRECT</evaluation><think>Try 2</think><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Retry pattern"
        ),
        (
            "<think>Thinking</think><evaluation>First evaluation</evaluation><evaluation>CORRECT</evaluation><answer>42</answer>",
            1.0,
            "Multiple evaluations with final CORRECT"
        ),

        # Invalid cases - Unbalanced tags
        (
            "<think>Thinking<evaluation>CORRECT</evaluation><answer>42</answer>",
            0.0,
            "Unclosed think tag"
        ),
        (
            "<think>Thinking</think><evaluation>CORRECT</evaluation><answer>42",
            0.0,
            "Unclosed answer tag"
        ),

        # Invalid cases - Missing required tags
        (
            "<evaluation>CORRECT</evaluation><answer>42</answer>",
            0.0,
            "Missing think tag"
        ),
        (
            "<think>Thinking</think><evaluation>CORRECT</evaluation>",
            0.0,
            "Missing answer tag"
        ),
        (
            "<think>Thinking</think><answer>42</answer>",
            0.0,
            "Missing CORRECT evaluation"
        ),

        # Invalid cases - Wrong order
        (
            "<think>Thinking</think><answer>42</answer><evaluation>CORRECT</evaluation>",
            0.0,
            "Answer before final CORRECT evaluation"
        ),
        (
            "<evaluation>CORRECT</evaluation><think>Thinking</think><answer>42</answer>",
            0.0,
            "No think tag before CORRECT evaluation"
        ),

        # Edge cases
        (
            "<think></think><evaluation>CORRECT</evaluation><answer>x</answer>",
            1.0,
            "Minimal valid content"
        ),
        (
            "<think>a</think><evaluation>INCORRECT</evaluation><think>b</think><evaluation>The answer is CORRECT</evaluation><answer>c</answer>",
            1.0,
            "CORRECT embedded in evaluation text"
        ),
        (
            "<think>Thinking</think><evaluation>correct</evaluation><answer>42</answer>",
            1.0,
            "Lowercase 'correct' (case insensitive)"
        ),

        # Additional strict validation tests
        (
            "<think>Thinking</think><evaluation>INCORRECT</evaluation><answer>42</answer>",
            0.0,
            "Only INCORRECT, no CORRECT"
        ),
        (
            "<think>A</think><evaluation>CORRECT</evaluation><think>B</think><answer>C</answer>",
            1.0,
            "Extra think tag after CORRECT is OK"
        ),
    ]

    passed = 0
    failed = 0

    for i, (input_str, expected, description) in enumerate(test_cases, 1):
        result = format_reward_strict(input_str)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {i}: {description}")
        print(f"Expected: {expected}, Got: {result} - {status}")
        if result != expected:
            print(f"Input: {input_str[:100]}...")

    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"{'=' * 80}\n")

    return passed, failed


def compare_functions():
    """Compare behavior differences between both functions"""
    print("=" * 80)
    print("Comparing format_reward() vs format_reward_strict()")
    print("=" * 80)

    test_cases = [
        "<think>Thinking</think><evaluation>CORRECT</evaluation><answer>42</answer>",
        "<think>A</think><evaluation>INCORRECT</evaluation><think>B</think><evaluation>CORRECT</evaluation><answer>C</answer>",
        "<think>Thinking<evaluation>CORRECT</evaluation><answer>42</answer>",  # Unclosed think
        "<think>A</think><answer>B</answer><evaluation>CORRECT</evaluation>",  # Wrong order
        "<think>A</think><evaluation>correct</evaluation><answer>B</answer>",  # Lowercase
    ]

    print(f"\n{'Input':<60} | Regular | Strict")
    print("-" * 80)

    for test in test_cases:
        regular = format_reward(test)
        strict = format_reward_strict(test)
        display = test[:57] + "..." if len(test) > 60 else test
        diff_marker = " " if regular == strict else " *"
        print(f"{display:<60} | {regular:>7.1f} | {strict:>6.1f}{diff_marker}")

    print("\n* = Different results between functions\n")


if __name__ == "__main__":
    # Run all tests
    p1, f1 = test_format_reward()
    p2, f2 = test_format_reward_strict()
    compare_functions()

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"format_reward():        {p1} passed, {f1} failed")
    print(f"format_reward_strict(): {p2} passed, {f2} failed")
    print(f"Total:                  {p1+p2} passed, {f1+f2} failed")
    print("=" * 80)
