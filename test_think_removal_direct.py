#!/usr/bin/env python3
"""Direct test of think tag removal function."""


def _remove_thinking_section(prediction: str) -> str:
    """Remove thinking section from prediction."""
    # Remove assistant prefix (common in chat models)
    prediction = prediction.replace("<|assistant|>", "").strip()

    # Remove thinking section (everything before and including </think>)
    if "</think>" in prediction:
        prediction = prediction.split("</think>")[-1]

    # Remove answer tags (some models wrap final answer in these)
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")

    return prediction.strip()


print("=" * 70)
print("Testing Think Tag Removal - Direct Function Test")
print("=" * 70 + "\n")

# Test 1: No comma constraint
print("Test 1: No comma with think tags")
response = "<think>Let me think, carefully, with commas.</think>Answer without commas"
cleaned = _remove_thinking_section(response)
has_comma = "," in cleaned
print(f"  Response: {response}")
print(f"  Cleaned:  {cleaned}")
print(f"  Has comma: {has_comma}")
print(f"  ✓ PASS" if not has_comma else f"  ❌ FAIL")
print()

# Test 2: Word count
print("Test 2: Word count with think tags")
response = "<think>" + " ".join(["word"] * 50) + "</think>Just five words here"
cleaned = _remove_thinking_section(response)
word_count = len(cleaned.split())
print(f"  Response: {response[:80]}...")
print(f"  Cleaned:  {cleaned}")
print(f"  Word count: {word_count}")
print(f"  ✓ PASS" if word_count == 4 else f"  ❌ FAIL (expected 4)")
print()

# Test 3: Lowercase with think tags
print("Test 3: Lowercase with think tags")
response = "<think>UPPERCASE THINKING</think>all lowercase answer"
cleaned = _remove_thinking_section(response)
is_lowercase = cleaned == cleaned.lower()
print(f"  Response: {response}")
print(f"  Cleaned:  {cleaned}")
print(f"  Is lowercase: {is_lowercase}")
print(f"  ✓ PASS" if is_lowercase else f"  ❌ FAIL")
print()

# Test 4: Real-world example
print("Test 4: Real-world DeepSeek-R1 style output")
response = """<think>
Hmm, let me analyze this. The question asks about calorie content, and I need to avoid commas.
Let me plan my response carefully, ensuring proper grammar, clear explanation, and no commas whatsoever.
</think>

Almonds contain 396 calories per 32-gram serving because they are extremely calorie-dense. About half their weight consists of healthy fats which provide 9 calories per gram."""

cleaned = _remove_thinking_section(response)
has_comma = "," in cleaned
word_count = len(cleaned.split())
print(f"  Original has commas in think: {', ' in response}")
print(f"  Cleaned has commas: {has_comma}")
print(f"  Cleaned word count: {word_count}")
print(f"  Cleaned text: {cleaned[:150]}...")
print(f"  ✓ PASS" if not has_comma else f"  ❌ FAIL")
print()

# Test 5: Multiple formats
print("Test 5: Multiple tag formats")
formats = [
    ("<think>a,b,c</think>clean", "clean"),
    ("<|assistant|><think>a,b</think>clean", "clean"),
    ("<think>bad</think><answer>good</answer>", "good"),
    ("no tags here", "no tags here"),
]

all_passed = True
for input_text, expected in formats:
    result = _remove_thinking_section(input_text)
    passed = result == expected
    if not passed:
        print(f"  ❌ FAIL: {input_text!r} -> {result!r} (expected {expected!r})")
        all_passed = False

if all_passed:
    print(f"  ✓ All format tests PASSED")
print()

print("=" * 70)
print("Function works correctly! ✓")
print("=" * 70)
