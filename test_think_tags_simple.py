#!/usr/bin/env python3
"""Simple test to check if think tags are being removed."""

import re

# Test the current behavior
test_responses = [
    {
        "name": "With commas in think section",
        "response": "<think>Let me think, carefully, with commas.</think>\nAnswer without commas here",
        "instruction": "no_comma",
        "expected_after_removal": "Answer without commas here"
    },
    {
        "name": "With word count in think",
        "response": "<think>This is a very long thinking section with many many words that exceed the limit.</think>\nShort answer here.",
        "instruction": "word_count",
        "expected_after_removal": "Short answer here."
    },
    {
        "name": "With assistant prefix",
        "response": "<|assistant|><think>Thinking, with, commas</think>Clean answer",
        "instruction": "no_comma",
        "expected_after_removal": "Clean answer"
    },
    {
        "name": "With answer tags",
        "response": "<think>Thinking</think><answer>Final answer</answer>",
        "instruction": "any",
        "expected_after_removal": "Final answer"
    },
]


def remove_thinking_section(prediction: str) -> str:
    """Remove thinking section from prediction.

    Based on open-instruct implementation.
    """
    # Remove assistant prefix
    prediction = prediction.replace("<|assistant|>", "").strip()

    # Remove thinking section (everything before </think>)
    if "</think>" in prediction:
        prediction = prediction.split("</think>")[-1]

    # Remove answer tags
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")

    return prediction.strip()


print("=" * 70)
print("Testing Think Tag Removal")
print("=" * 70 + "\n")

for test in test_responses:
    print(f"Test: {test['name']}")
    print(f"  Original: {test['response']!r}")

    cleaned = remove_thinking_section(test['response'])
    print(f"  Cleaned:  {cleaned!r}")
    print(f"  Expected: {test['expected_after_removal']!r}")

    if cleaned == test['expected_after_removal']:
        print("  ✓ PASS")
    else:
        print("  ❌ FAIL")
    print()

print("=" * 70)
print("Now let's test with actual constraint checks")
print("=" * 70 + "\n")

# Test comma check
def has_no_commas(text: str) -> bool:
    return "," not in text

print("Test 1: No comma constraint")
response = "<think>Let me think, carefully, with commas.</think>\nAnswer without commas here"
original_check = has_no_commas(response)
cleaned_check = has_no_commas(remove_thinking_section(response))

print(f"  Response: {response!r}")
print(f"  Without cleaning: {original_check} (WRONG - fails because of think tags)")
print(f"  With cleaning: {cleaned_check} (CORRECT - passes)")
print()

# Test word count
def word_count_less_than(text: str, limit: int) -> bool:
    return len(text.split()) < limit

print("Test 2: Word count < 10")
response = "<think>This is a very long thinking section with many many words that exceed the limit by far.</think>\nShort answer."
original_check = word_count_less_than(response, 10)
cleaned_check = word_count_less_than(remove_thinking_section(response), 10)

print(f"  Response: {response!r}")
print(f"  Without cleaning: {original_check} (WRONG - {len(response.split())} words)")
print(f"  With cleaning: {cleaned_check} (CORRECT - {len(remove_thinking_section(response).split())} words)")
print()

# Test lowercase
def is_all_lowercase(text: str) -> bool:
    return text == text.lower()

print("Test 3: All lowercase constraint")
response = "<think>LET ME THINK IN UPPERCASE.</think>\nall lowercase answer here"
original_check = is_all_lowercase(response)
cleaned_check = is_all_lowercase(remove_thinking_section(response))

print(f"  Response: {response!r}")
print(f"  Without cleaning: {original_check} (WRONG - has uppercase in think)")
print(f"  With cleaning: {cleaned_check} (CORRECT - only checks answer)")
print()

print("=" * 70)
print("CONCLUSION: We MUST add remove_thinking_section() preprocessing")
print("=" * 70)
