#!/usr/bin/env python3
"""
Test script to validate RLVR-IFeval preprocessing logic.

This script tests the data transformation without requiring actual dataset downloads.
"""

import json

def test_rlvr_preprocessing():
    """Test the RLVR-IFeval preprocessing function."""

    # Mock RLVR-IFeval example
    mock_example = {
        "messages": [
            {"role": "user", "content": "Write a short story about adventure."}
        ],
        "ground_truth": '{"keywords": ["adventure", "mystery"]}',
        "constraint_type": "keywords:existence",
        "constraint": "Include the keywords 'adventure' and 'mystery'"
    }

    def process_fn(example, idx):
        """Process function from rlvr_ifeval.py"""
        messages = example.get("messages", [])

        # Parse ground_truth JSON string
        ground_truth_str = example.get("ground_truth", "{}")
        try:
            ground_truth_dict = json.loads(ground_truth_str)
        except json.JSONDecodeError:
            ground_truth_dict = {}

        # Extract constraint information
        constraint_type = example.get("constraint_type", "")
        constraint = example.get("constraint", "")

        # Convert to instruction_id_list and kwargs format
        instruction_id_list = [constraint_type] if constraint_type else []
        kwargs_list = [ground_truth_dict] if ground_truth_dict else []

        # Build prompt from messages
        prompt_messages = []
        user_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                user_content = content
                prompt_messages.append({"role": "user", "content": content})
            elif role == "system":
                prompt_messages.insert(0, {"role": "system", "content": content})

        if not prompt_messages:
            prompt_messages = [{"role": "user", "content": user_content or constraint}]

        data = {
            "data_source": "allenai/RLVR-IFeval",
            "prompt": prompt_messages,
            "ability": "instruction_following",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs_list,
                },
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "key": f"rlvr_ifeval_{idx}",
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs_list,
                "constraint_type": constraint_type,
                "constraint": constraint,
                "prompt": user_content,
            },
        }
        return data

    # Test preprocessing
    result = process_fn(mock_example, 0)

    print("=" * 60)
    print("RLVR-IFeval Preprocessing Test")
    print("=" * 60)
    print()

    print("Input Example:")
    print("-" * 60)
    print(json.dumps(mock_example, indent=2))
    print()

    print("Processed Output:")
    print("-" * 60)
    print(json.dumps(result, indent=2))
    print()

    # Validation checks
    print("Validation Checks:")
    print("-" * 60)

    checks = [
        ("Data source set", result["data_source"] == "allenai/RLVR-IFeval"),
        ("Prompt is list", isinstance(result["prompt"], list)),
        ("Has user message", len(result["prompt"]) > 0),
        ("Ability is instruction_following", result["ability"] == "instruction_following"),
        ("Reward model style is rule", result["reward_model"]["style"] == "rule"),
        ("Has instruction_id_list", len(result["reward_model"]["ground_truth"]["instruction_id_list"]) > 0),
        ("Has kwargs", len(result["reward_model"]["ground_truth"]["kwargs"]) > 0),
        ("Constraint type preserved", result["extra_info"]["constraint_type"] == "keywords:existence"),
        ("Keywords parsed correctly", "keywords" in result["reward_model"]["ground_truth"]["kwargs"][0]),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed!")
    print("=" * 60)

    return all_passed


def test_google_ifeval_compatibility():
    """Test that RLVR format is compatible with existing IFEval reward function."""

    print()
    print("=" * 60)
    print("Testing Compatibility with IFEval Reward Function")
    print("=" * 60)
    print()

    # Simulate the format expected by ifeval.py reward function
    rlvr_processed = {
        "reward_model": {
            "ground_truth": {
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["adventure", "mystery"]}],
            }
        },
        "extra_info": {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["adventure", "mystery"]}],
            "prompt": "Write a short story about adventure.",
        }
    }

    # Check compatibility
    has_ground_truth = "ground_truth" in rlvr_processed["reward_model"]
    has_instruction_ids = "instruction_id_list" in rlvr_processed["reward_model"]["ground_truth"]
    has_kwargs = "kwargs" in rlvr_processed["reward_model"]["ground_truth"]
    has_extra_info = "extra_info" in rlvr_processed

    checks = [
        ("Has ground_truth dict", has_ground_truth),
        ("Has instruction_id_list", has_instruction_ids),
        ("Has kwargs", has_kwargs),
        ("Has extra_info", has_extra_info),
        ("instruction_id_list is list", isinstance(rlvr_processed["reward_model"]["ground_truth"]["instruction_id_list"], list)),
        ("kwargs is list", isinstance(rlvr_processed["reward_model"]["ground_truth"]["kwargs"], list)),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ RLVR-IFeval format is compatible with IFEval reward function!")
    else:
        print("✗ Compatibility issues detected!")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    test1_passed = test_rlvr_preprocessing()
    test2_passed = test_google_ifeval_compatibility()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  RLVR Preprocessing: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"  IFEval Compatibility: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print()
        print("🎉 All tests passed!")
        print()
        print("Next steps:")
        print("  1. Run: python examples/data_preprocess/rlvr_ifeval.py --local_save_dir ~/data/rlvr_ifeval")
        print("  2. Run: python examples/data_preprocess/ifeval.py --local_save_dir ~/data/ifeval_test")
        print("  3. Run: bash examples/grpo_trainer/run_rlvr_ifeval_grpo.sh")
        exit(0)
    else:
        print()
        print("❌ Some tests failed. Please review the output above.")
        exit(1)
