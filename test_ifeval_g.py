#!/usr/bin/env python3
"""Test script for IFEval implementation using IFEvalG library."""

import sys
import importlib.util

# Load ifeval_util modules directly
sys.path.insert(0, '/home/user/verl/verl/utils/reward_score')

# Load the instruction modules
spec_inst_util = importlib.util.spec_from_file_location(
    "instructions_util",
    "/home/user/verl/verl/utils/reward_score/ifeval_util/instructions_util.py"
)
instructions_util = importlib.util.module_from_spec(spec_inst_util)
spec_inst_util.loader.exec_module(instructions_util)

spec_inst = importlib.util.spec_from_file_location(
    "instructions",
    "/home/user/verl/verl/utils/reward_score/ifeval_util/instructions.py"
)
instructions = importlib.util.module_from_spec(spec_inst)
sys.modules['instructions_util'] = instructions_util
spec_inst.loader.exec_module(instructions)

spec_inst_reg = importlib.util.spec_from_file_location(
    "instructions_registry",
    "/home/user/verl/verl/utils/reward_score/ifeval_util/instructions_registry.py"
)
instructions_registry = importlib.util.module_from_spec(spec_inst_reg)
sys.modules['instructions'] = instructions
spec_inst_reg.loader.exec_module(instructions_registry)

INSTRUCTION_DICT = instructions_registry.INSTRUCTION_DICT

# Load ifeval module
spec_ifeval = importlib.util.spec_from_file_location(
    "ifeval",
    "/home/user/verl/verl/utils/reward_score/ifeval.py"
)
ifeval = importlib.util.module_from_spec(spec_ifeval)
sys.modules['ifeval_util'] = type(sys)('ifeval_util')
sys.modules['ifeval_util'].INSTRUCTION_DICT = INSTRUCTION_DICT
spec_ifeval.loader.exec_module(ifeval)

compute_score = ifeval.compute_score


def test_keywords_existence():
    """Test keywords:existence constraint."""
    print("Testing keywords:existence...")

    ground_truth = {
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["robot", "AI"]}]
    }

    response = "This text contains robot and AI keywords"
    result = compute_score(response, ground_truth, use_ifeval_library=False)

    print(f"  Result: {result}")
    assert result["score"] == 1.0, f"Expected score 1.0, got {result['score']}"
    print("  ✓ keywords:existence passed")


def test_forbidden_words():
    """Test keywords:forbidden_words constraint."""
    print("Testing keywords:forbidden_words...")

    ground_truth = {
        "instruction_id_list": ["keywords:forbidden_words"],
        "kwargs": [{"forbidden_words": ["evil", "bad"]}]
    }

    response = "This is a good text with no forbidden words"
    result = compute_score(response, ground_truth, use_ifeval_library=False)

    print(f"  Result: {result}")
    assert result["score"] == 1.0, f"Expected score 1.0, got {result['score']}"
    print("  ✓ keywords:forbidden_words passed")


def test_number_words():
    """Test length_constraints:number_words constraint."""
    print("Testing length_constraints:number_words...")

    ground_truth = {
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 10, "relation": "at least"}]
    }

    response = "This is a test response that contains at least ten words in total"
    result = compute_score(response, ground_truth, use_ifeval_library=False)

    print(f"  Result: {result}")
    assert result["score"] == 1.0, f"Expected score 1.0, got {result['score']}"
    print("  ✓ length_constraints:number_words passed")


def test_multiple_constraints():
    """Test multiple constraints together."""
    print("Testing multiple constraints...")

    ground_truth = {
        "instruction_id_list": [
            "keywords:existence",
            "keywords:forbidden_words",
            "length_constraints:number_words"
        ],
        "kwargs": [
            {"keywords": ["robot"]},
            {"forbidden_words": ["evil"]},
            {"num_words": 5, "relation": "at least"}
        ]
    }

    response = "This robot is good and helpful"
    result = compute_score(response, ground_truth, use_ifeval_library=False)

    print(f"  Result: {result}")
    assert result["score"] == 1.0, f"Expected score 1.0, got {result['score']}"
    assert result["num_instructions"] == 3, f"Expected 3 instructions, got {result['num_instructions']}"
    assert result["num_followed"] == 3, f"Expected 3 followed, got {result['num_followed']}"
    print("  ✓ Multiple constraints passed")


def test_postscript():
    """Test detectable_content:postscript constraint."""
    print("Testing detectable_content:postscript...")

    ground_truth = {
        "instruction_id_list": ["detectable_content:postscript"],
        "kwargs": [{"postscript_marker": "P.S."}]
    }

    response = "This is the main letter content.\n\nP.S. This is a postscript note."
    result = compute_score(response, ground_truth, use_ifeval_library=False)

    print(f"  Result: {result}")
    assert result["score"] == 1.0, f"Expected score 1.0, got {result['score']}"
    print("  ✓ detectable_content:postscript passed")


def test_instruction_dict():
    """Test that INSTRUCTION_DICT is loaded correctly."""
    print("Testing INSTRUCTION_DICT...")

    # Check for some key instruction types
    expected_instructions = [
        "keywords:existence",
        "keywords:frequency",
        "keywords:forbidden_words",
        "length_constraints:number_words",
        "length_constraints:number_paragraphs",
        "detectable_content:postscript",
        "detectable_format:json_format",
        "change_case:english_lowercase",
        "punctuation:no_comma",
    ]

    for inst_id in expected_instructions:
        assert inst_id in INSTRUCTION_DICT, f"Missing instruction: {inst_id}"

    print(f"  ✓ INSTRUCTION_DICT contains {len(INSTRUCTION_DICT)} instruction types")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing IFEval Implementation with IFEvalG Library")
    print("=" * 60 + "\n")

    try:
        test_instruction_dict()
        print()
        test_keywords_existence()
        test_forbidden_words()
        test_number_words()
        test_postscript()
        test_multiple_constraints()

        print("\n" + "=" * 60)
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
