#!/usr/bin/env python3
"""End-to-end test for IFEval with think tags - uses actual implementation."""

import sys
import importlib.util

# Load the updated ifeval module directly
spec = importlib.util.spec_from_file_location("ifeval", "/home/user/verl/verl/utils/reward_score/ifeval.py")
ifeval_module = importlib.util.module_from_spec(spec)

# Need to mock the ifeval_util import
class MockInstructionDict:
    """Mock INSTRUCTION_DICT for testing without full dependencies."""
    pass

# Create a mock module for ifeval_util
mock_ifeval_util = type(sys)('ifeval_util')
mock_ifeval_util.INSTRUCTION_DICT = {}
sys.modules['verl'] = type(sys)('verl')
sys.modules['verl.utils'] = type(sys)('verl.utils')
sys.modules['verl.utils.reward_score'] = type(sys)('verl.utils.reward_score')
sys.modules['verl.utils.reward_score.ifeval_util'] = mock_ifeval_util

# Now load the module
spec.loader.exec_module(ifeval_module)

# Get the function we need to test
_remove_thinking_section = ifeval_module._remove_thinking_section


def test_remove_thinking_section():
    """Test the _remove_thinking_section function directly."""
    print("=" * 70)
    print("Testing _remove_thinking_section function")
    print("=" * 70 + "\n")

    test_cases = [
        {
            "name": "Basic think tags",
            "input": "<think>Reasoning with, commas, here</think>Clean answer without commas",
            "expected": "Clean answer without commas"
        },
        {
            "name": "With assistant prefix",
            "input": "<|assistant|><think>Thinking process</think>Final answer",
            "expected": "Final answer"
        },
        {
            "name": "With answer tags",
            "input": "<think>Reasoning</think><answer>The answer</answer>",
            "expected": "The answer"
        },
        {
            "name": "All tags combined",
            "input": "<|assistant|><think>Long reasoning, with commas</think><answer>Clean final answer</answer>",
            "expected": "Clean final answer"
        },
        {
            "name": "No think tags",
            "input": "Just a normal response",
            "expected": "Just a normal response"
        },
        {
            "name": "Multiline think section",
            "input": """<think>
This is a long thinking section
with multiple lines
and lots of, commas, everywhere
</think>
Final answer on multiple
lines without commas""",
            "expected": "Final answer on multiple\nlines without commas"
        },
        {
            "name": "Think with forbidden words",
            "input": "<think>This is bad and evil thinking</think>This is good output",
            "expected": "This is good output"
        },
        {
            "name": "Think with excess words",
            "input": "<think>This thinking section has way too many words that would exceed any reasonable word count limit for the actual answer portion which should be kept short and concise.</think>Short answer.",
            "expected": "Short answer."
        },
        {
            "name": "Think with uppercase",
            "input": "<think>UPPERCASE THINKING</think>lowercase answer only",
            "expected": "lowercase answer only"
        },
    ]

    all_passed = True
    for test in test_cases:
        result = _remove_thinking_section(test["input"])
        passed = result == test["expected"]

        print(f"Test: {test['name']}")
        if not passed:
            print(f"  ❌ FAILED")
            print(f"  Input:    {test['input']!r}")
            print(f"  Expected: {test['expected']!r}")
            print(f"  Got:      {result!r}")
            all_passed = False
        else:
            print(f"  ✓ PASSED")
        print()

    return all_passed


def test_constraint_scenarios():
    """Test how constraints work with think tags."""
    print("=" * 70)
    print("Testing Constraint Scenarios with Think Tags")
    print("=" * 70 + "\n")

    test_cases = [
        {
            "name": "Commas in think, not in answer",
            "response": "<think>Let me think, carefully, with commas</think>Answer without commas here",
            "check": lambda r: "," not in r,
            "description": "Should pass (no commas in answer)"
        },
        {
            "name": "Commas in answer (should fail)",
            "response": "<think>Clean thinking</think>Answer with, commas, here",
            "check": lambda r: "," not in r,
            "description": "Should fail (commas in answer)"
        },
        {
            "name": "Word count - short answer, long think",
            "response": "<think>" + " ".join(["word"] * 100) + "</think>Short answer here",
            "check": lambda r: len(r.split()) < 10,
            "description": "Should pass (only 3 words in answer)"
        },
        {
            "name": "Lowercase - uppercase think, lowercase answer",
            "response": "<think>UPPERCASE THINKING HERE</think>all lowercase answer",
            "check": lambda r: r == r.lower(),
            "description": "Should pass (answer is lowercase)"
        },
        {
            "name": "Lowercase - mixed case answer (should fail)",
            "response": "<think>thinking</think>Answer With Capital Letters",
            "check": lambda r: r == r.lower(),
            "description": "Should fail (answer has capitals)"
        },
    ]

    all_passed = True
    for test in test_cases:
        cleaned = _remove_thinking_section(test["response"])
        result = test["check"](cleaned)

        print(f"Test: {test['name']}")
        print(f"  {test['description']}")
        print(f"  Original: {test['response'][:80]}...")
        print(f"  Cleaned:  {cleaned!r}")
        print(f"  Result:   {result}")

        # Determine expected result from description
        expected = "pass" in test["description"].lower()

        if result == expected:
            print(f"  ✓ CORRECT")
        else:
            print(f"  ❌ WRONG (expected {'pass' if expected else 'fail'})")
            all_passed = False
        print()

    return all_passed


def test_real_world_examples():
    """Test with realistic reasoning model outputs."""
    print("=" * 70)
    print("Testing Real-World Reasoning Model Outputs")
    print("=" * 70 + "\n")

    examples = [
        {
            "name": "DeepSeek-R1 style output",
            "response": """<think>
Hmm, let me think about this step by step. The question asks why there are 396 calories in 32g of almonds.
First, I need to understand calorie density. Almonds are high in fats, which have 9 calories per gram.
They also have protein and carbs, but fats are the main contributor.
Let me calculate: if almonds are about 50% fat by weight, that's 16g of fat.
16g × 9 cal/g = 144 calories from fat alone.
But the total is 396, so other macros contribute too.
Actually, almonds have protein and carbs as well. Let me be more precise.
Okay, so the final answer should explain the caloric density without using commas as instructed.
</think>

Almonds contain approximately 396 calories per 32-gram serving because they are extremely calorie-dense. About half their weight consists of healthy fats which provide 9 calories per gram. The remaining calories come from protein and carbohydrates. This high fat content makes almonds an energy-rich food despite their small serving size.""",
            "instruction": "no_comma",
            "should_pass": True
        },
        {
            "name": "Claude-style with answer tags",
            "response": """<think>
The user wants to know about almond calories, and specifically asked not to use commas.
Let me structure a clear response that explains the caloric content while avoiding commas entirely.
</think>

<answer>
Almonds contain 396 calories per 32-gram serving due to their high fat content. Dietary fats provide 9 calories per gram compared to only 4 calories per gram from protein or carbohydrates. Since almonds are approximately 50% fat by weight they pack significant energy into a small serving size. The remaining calories come from protein (about 6g per serving) and carbohydrates (about 6g per serving).
</answer>""",
            "instruction": "no_comma",
            "should_pass": True
        },
        {
            "name": "Short answer with long thinking",
            "response": """<think>
Let me carefully consider this question about almond calories. I need to provide a comprehensive explanation that covers the macronutrient composition of almonds including their fat content which is the primary contributor to their caloric density. Almonds are known to be nutrient-dense foods that provide healthy monounsaturated fats, protein, fiber, vitamin E, magnesium, and other beneficial nutrients. However, this high nutrient density also means they are calorically dense. I should explain this in a way that is clear and concise while keeping the response under 50 words as required.
</think>

Almonds are calorie-dense because of their high fat content.""",
            "instruction": "word_count_under_50",
            "should_pass": True
        },
    ]

    all_passed = True
    for example in examples:
        cleaned = _remove_thinking_section(example["response"])

        print(f"Example: {example['name']}")
        print(f"  Instruction: {example['instruction']}")
        print(f"  Should pass: {example['should_pass']}")
        print(f"  Original length: {len(example['response'])} chars, {len(example['response'].split())} words")
        print(f"  Cleaned length:  {len(cleaned)} chars, {len(cleaned.split())} words")

        # Check the constraint
        if example['instruction'] == 'no_comma':
            result = "," not in cleaned
            print(f"  Has commas: {', ' in cleaned}")
        elif example['instruction'] == 'word_count_under_50':
            word_count = len(cleaned.split())
            result = word_count < 50
            print(f"  Word count: {word_count}")

        if result == example['should_pass']:
            print(f"  ✓ CORRECT")
        else:
            print(f"  ❌ WRONG")
            print(f"  Cleaned text: {cleaned[:200]}...")
            all_passed = False
        print()

    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IFEval Think Tag Handling - Comprehensive Test Suite")
    print("=" * 70 + "\n")

    try:
        results = []
        results.append(test_remove_thinking_section())
        results.append(test_constraint_scenarios())
        results.append(test_real_world_examples())

        print("\n" + "=" * 70)
        if all(results):
            print("ALL TESTS PASSED! ✓")
            print("Think tags are correctly handled for reasoning models!")
        else:
            print("SOME TESTS FAILED! ❌")
            print("Check the output above for details.")
        print("=" * 70)

        sys.exit(0 if all(results) else 1)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
