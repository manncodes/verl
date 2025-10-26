#!/usr/bin/env python3
"""Quick test script for migrated IFEval implementation."""

import sys
import importlib.util

# Load the module directly without importing the verl package
spec = importlib.util.spec_from_file_location("ifeval", "/home/user/verl/verl/utils/reward_score/ifeval.py")
ifeval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ifeval)

# Get the functions from the module
compute_score = ifeval.compute_score
verify_keywords = ifeval.verify_keywords
validate_forbidden_words = ifeval.validate_forbidden_words
validate_word_constraint = ifeval.validate_word_constraint
verify_postscript = ifeval.verify_postscript
validate_lowercase = ifeval.validate_lowercase
IF_FUNCTIONS_MAP = ifeval.IF_FUNCTIONS_MAP


def test_basic_functions():
    """Test individual constraint functions."""
    print("Testing individual functions...")

    # Test verify_keywords
    result = verify_keywords("This text contains robot and AI keywords", ["robot", "AI"])
    assert result == True, f"verify_keywords failed: {result}"
    print("✓ verify_keywords passed")

    # Test validate_forbidden_words
    result = validate_forbidden_words("This is a clean text", ["bad", "evil"])
    assert result == True, f"validate_forbidden_words failed: {result}"
    print("✓ validate_forbidden_words passed")

    # Test validate_word_constraint
    result = validate_word_constraint("one two three four five", 5, "at least")
    assert result == True, f"validate_word_constraint failed: {result}"
    print("✓ validate_word_constraint passed")

    # Test verify_postscript
    result = verify_postscript("Letter content here\n\nP.S. This is a postscript", "P.S.")
    assert result == True, f"verify_postscript failed: {result}"
    print("✓ verify_postscript passed")

    # Test validate_lowercase
    result = validate_lowercase("all lowercase text")
    assert result == True, f"validate_lowercase failed: {result}"
    print("✓ validate_lowercase passed")

    print("All individual function tests passed!\n")


def test_compute_score():
    """Test the main compute_score function."""
    print("Testing compute_score...")

    # Test with simple keywords constraint (using func_name in kwargs for old format)
    ground_truth = {
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"func_name": "verify_keywords", "keyword_list": ["robot", "AI"]}]
    }

    response = "This text contains robot and AI keywords"
    result = compute_score(response, ground_truth, use_ifeval_library=False)

    print(f"Result: {result}")
    assert result["score"] == 1.0, f"Expected score 1.0, got {result['score']}"
    assert result["num_instructions"] == 1, f"Expected 1 instruction, got {result['num_instructions']}"
    assert result["num_followed"] == 1, f"Expected 1 followed, got {result['num_followed']}"
    print("✓ compute_score with keywords passed")

    # Test with forbidden words
    ground_truth2 = {
        "instruction_id_list": ["keywords:forbidden_words"],
        "kwargs": [{"func_name": "validate_forbidden_words", "forbidden_words": ["evil", "bad"]}]
    }

    response2 = "This is a good text"
    result2 = compute_score(response2, ground_truth2, use_ifeval_library=False)
    print(f"Result: {result2}")
    assert result2["score"] == 1.0, f"Expected score 1.0, got {result2['score']}"
    print("✓ compute_score with forbidden words passed")

    # Test with multiple constraints
    ground_truth3 = {
        "instruction_id_list": ["keywords:existence", "keywords:forbidden_words"],
        "kwargs": [
            {"func_name": "verify_keywords", "keyword_list": ["robot"]},
            {"func_name": "validate_forbidden_words", "forbidden_words": ["evil"]}
        ]
    }

    response3 = "This robot is good"
    result3 = compute_score(response3, ground_truth3, use_ifeval_library=False)
    print(f"Result: {result3}")
    assert result3["score"] == 1.0, f"Expected score 1.0, got {result3['score']}"
    assert result3["num_instructions"] == 2, f"Expected 2 instructions, got {result3['num_instructions']}"
    assert result3["num_followed"] == 2, f"Expected 2 followed, got {result3['num_followed']}"
    print("✓ compute_score with multiple constraints passed")

    print("All compute_score tests passed!\n")


def test_if_functions_map():
    """Test that IF_FUNCTIONS_MAP contains all expected functions."""
    print("Testing IF_FUNCTIONS_MAP...")

    expected_functions = [
        "verify_keywords",
        "verify_keyword_frequency",
        "validate_forbidden_words",
        "verify_letter_frequency",
        "validate_response_language",
        "verify_paragraph_count",
        "validate_word_constraint",
        "verify_sentence_constraint",
        "validate_paragraphs",
        "verify_postscript",
        "validate_placeholders",
        "verify_bullet_points",
        "validate_title",
        "validate_choice",
        "validate_highlighted_sections",
        "validate_sections",
        "validate_json_format",
        "validate_repeat_prompt",
        "validate_two_responses",
        "validate_uppercase",
        "validate_lowercase",
        "validate_frequency_capital_words",
        "validate_end",
        "validate_quotation",
        "validate_no_commas",
    ]

    for func_name in expected_functions:
        assert func_name in IF_FUNCTIONS_MAP, f"Missing function: {func_name}"

    print(f"✓ IF_FUNCTIONS_MAP contains all {len(expected_functions)} expected functions")
    print(f"Total functions in map: {len(IF_FUNCTIONS_MAP)}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Migrated IFEval Implementation")
    print("=" * 60 + "\n")

    try:
        test_if_functions_map()
        test_basic_functions()
        test_compute_score()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
