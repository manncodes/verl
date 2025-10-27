#!/usr/bin/env python3
"""Test IFEval with reasoning model responses containing think tags."""

import sys
import importlib.util

# Load modules directly
spec = importlib.util.spec_from_file_location("ifeval", "/home/user/verl/verl/utils/reward_score/ifeval.py")
ifeval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ifeval)

compute_score = ifeval.compute_score


def test_no_comma_with_think_tags():
    """Test that commas in think tags don't count against no_comma instruction."""
    print("Testing punctuation:no_comma with think tags...")

    ground_truth = {
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}]
    }

    # Response has commas in think section but not in actual answer
    response_with_think = """<think>
Let me think about this carefully, step by step, considering all aspects.
I need to make sure I don't use any commas in my final answer.
</think>

There are approximately 396 calories in 32 grams of unsalted almonds because almonds are calorie-dense and contain about 12.4 calories per gram."""

    result = compute_score(response_with_think, ground_truth, use_ifeval_library=False)
    print(f"  Result: {result}")

    if result["score"] != 1.0:
        print(f"  ❌ FAILED: Expected score 1.0 but got {result['score']}")
        print(f"  The think section with commas should be ignored!")
        return False

    print("  ✓ Think tags correctly ignored")
    return True


def test_word_count_with_think_tags():
    """Test that words in think tags don't count towards word count."""
    print("\nTesting length_constraints:number_words with think tags...")

    ground_truth = {
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 20, "relation": "less than"}]
    }

    # Think section has many words, but actual answer has < 20 words
    response_with_think = """<think>
This is a very long thinking section with many many words that should not be counted
towards the actual word limit constraint because this is just reasoning and not the
final answer that the user will see. Let me make sure the actual response is short.
</think>

Almonds are calorie-dense with about twelve point four calories per gram of weight."""

    result = compute_score(response_with_think, ground_truth, use_ifeval_library=False)
    print(f"  Result: {result}")

    # Count actual words (should be < 20)
    actual_answer = response_with_think.split("</think>")[-1].strip()
    actual_word_count = len(actual_answer.split())
    print(f"  Actual answer word count: {actual_word_count}")

    if result["score"] != 1.0:
        print(f"  ❌ FAILED: Expected score 1.0 but got {result['score']}")
        print(f"  Think section words should not be counted!")
        return False

    print("  ✓ Think tags correctly ignored")
    return True


def test_forbidden_words_with_think_tags():
    """Test that forbidden words in think tags don't trigger failures."""
    print("\nTesting keywords:forbidden_words with think tags...")

    ground_truth = {
        "instruction_id_list": ["keywords:forbidden_words"],
        "kwargs": [{"forbidden_words": ["evil", "bad", "terrible"]}]
    }

    # Think section has forbidden words, but answer doesn't
    response_with_think = """<think>
This is a bad question and it's terrible how evil it is to ask this.
Let me think of a good answer without using any of those bad words.
</think>

Almonds contain approximately 396 calories per serving because they have high fat content."""

    result = compute_score(response_with_think, ground_truth, use_ifeval_library=False)
    print(f"  Result: {result}")

    if result["score"] != 1.0:
        print(f"  ❌ FAILED: Expected score 1.0 but got {result['score']}")
        print(f"  Forbidden words in think section should be ignored!")
        return False

    print("  ✓ Think tags correctly ignored")
    return True


def test_keywords_with_think_tags():
    """Test that required keywords in think tags don't satisfy the requirement."""
    print("\nTesting keywords:existence with think tags...")

    ground_truth = {
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["calorie", "almond", "fat"]}]
    }

    # All keywords are in think section, not in answer
    response_with_think = """<think>
The question is about calorie content in almond servings and their fat composition.
</think>

The serving size contains approximately 396 units of energy."""

    result = compute_score(response_with_think, ground_truth, use_ifeval_library=False)
    print(f"  Result: {result}")

    if result["score"] != 0.0:
        print(f"  ❌ FAILED: Expected score 0.0 but got {result['score']}")
        print(f"  Keywords in think section should NOT count!")
        return False

    print("  ✓ Keywords correctly required in actual answer")
    return True


def test_multiple_think_formats():
    """Test various think tag formats."""
    print("\nTesting multiple think tag formats...")

    ground_truth = {
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}]
    }

    formats = [
        # Standard format
        "<think>\nReasoning with, commas, here\n</think>\nAnswer without commas here",

        # With assistant prefix
        "<|assistant|><think>\nReasoning, with, commas\n</think>\nClean answer here",

        # With answer tags
        "<think>\nThinking, with, commas\n</think>\n<answer>Clean answer</answer>",

        # Inline format
        "<think>Quick, comma, thinking</think> Clean answer without commas",
    ]

    all_passed = True
    for i, response in enumerate(formats, 1):
        result = compute_score(response, ground_truth, use_ifeval_library=False)
        if result["score"] != 1.0:
            print(f"  ❌ Format {i} FAILED: score={result['score']}")
            all_passed = False
        else:
            print(f"  ✓ Format {i} passed")

    return all_passed


def test_without_think_tags():
    """Ensure normal responses still work correctly."""
    print("\nTesting normal response without think tags...")

    ground_truth = {
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}]
    }

    # This should FAIL because it has commas
    response_with_commas = "There are 396 calories in almonds, because they contain fat, protein, and carbs."
    result = compute_score(response_with_commas, ground_truth, use_ifeval_library=False)

    if result["score"] != 0.0:
        print(f"  ❌ FAILED: Response with commas should score 0.0, got {result['score']}")
        return False
    print("  ✓ Correctly detected commas")

    # This should PASS
    response_no_commas = "There are 396 calories in almonds because they contain fat and protein and carbs."
    result = compute_score(response_no_commas, ground_truth, use_ifeval_library=False)

    if result["score"] != 1.0:
        print(f"  ❌ FAILED: Response without commas should score 1.0, got {result['score']}")
        return False
    print("  ✓ Correctly validated no commas")

    return True


def test_lowercase_with_think_tags():
    """Test that uppercase in think tags doesn't affect lowercase check."""
    print("\nTesting change_case:english_lowercase with think tags...")

    ground_truth = {
        "instruction_id_list": ["change_case:english_lowercase"],
        "kwargs": [{}]
    }

    # Think section has uppercase, answer is all lowercase
    response_with_think = """<think>
LET ME THINK ABOUT THIS IN UPPERCASE TO PLAN MY RESPONSE.
I NEED TO MAKE SURE THE ACTUAL ANSWER IS ALL LOWERCASE.
</think>

almonds contain approximately three hundred ninety six calories per serving because of their high fat content."""

    result = compute_score(response_with_think, ground_truth, use_ifeval_library=False)
    print(f"  Result: {result}")

    if result["score"] != 1.0:
        print(f"  ❌ FAILED: Expected score 1.0 but got {result['score']}")
        print(f"  Uppercase in think section should be ignored!")
        return False

    print("  ✓ Think tags correctly ignored")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Testing IFEval with Reasoning Model Think Tags")
    print("=" * 70 + "\n")

    tests = [
        test_without_think_tags,
        test_no_comma_with_think_tags,
        test_word_count_with_think_tags,
        test_forbidden_words_with_think_tags,
        test_keywords_with_think_tags,
        test_lowercase_with_think_tags,
        test_multiple_think_formats,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! ❌")
        print("=" * 70)
        sys.exit(1)
