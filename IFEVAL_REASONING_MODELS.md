# IFEval Support for Reasoning Models

## Problem Statement

Reasoning models (DeepSeek-R1, Claude-3.5, etc.) output their thinking process in `<think>` tags before providing the final answer. Without proper handling, these thinking sections incorrectly affect IFEval instruction verification.

### Examples of Issues

#### 1. No Comma Instruction ❌
```
Response:
<think>
Let me think about this carefully, step by step, with proper reasoning.
I need to make sure the final answer has no commas.
</think>

The answer contains no commas and follows the instruction perfectly.
```

**Problem**: The commas in the think section cause the instruction to fail, even though the actual answer has no commas.

#### 2. Word Count Instruction ❌
```
Response:
<think>
This is a very long thinking section with many many words that would
exceed any reasonable word count limit for the actual answer which
should be kept short and concise per the instruction requirements.
</think>

Short answer here.
```

**Problem**: The word count includes the think section (~30 words) instead of just the answer (~3 words).

#### 3. Lowercase Instruction ❌
```
Response:
<think>
LET ME THINK ABOUT THIS IN UPPERCASE TO ORGANIZE MY THOUGHTS CLEARLY.
THE FINAL ANSWER MUST BE ALL LOWERCASE AS INSTRUCTED.
</think>

this is the final answer in all lowercase
```

**Problem**: The uppercase in the think section causes the lowercase check to fail.

## Solution

Added `_remove_thinking_section()` function that preprocesses all responses before instruction verification.

### Implementation

```python
def _remove_thinking_section(prediction: str) -> str:
    """Remove thinking section from prediction.

    This is critical for evaluating reasoning models that output their
    thinking process before the final answer.
    """
    # Remove assistant prefix (common in chat models)
    prediction = prediction.replace("<|assistant|>", "").strip()

    # Remove thinking section (everything before and including </think>)
    if "</think>" in prediction:
        prediction = prediction.split("</think>")[-1]

    # Remove answer tags (some models wrap final answer in these)
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")

    return prediction.strip()
```

### Integration Points

The function is called **before any instruction verification** in both code paths:

1. `_compute_score_with_library()` - for official ifeval library
2. `_compute_score_with_ifeval_g()` - for IFEvalG from open-instruct

```python
# IMPORTANT: Remove thinking section first (before any other processing)
response = _remove_thinking_section(solution_str)

# Then do normal preprocessing and verification
if not strict:
    response = _preprocess_response_loose(response)
```

## Supported Tag Formats

The implementation handles multiple reasoning model formats:

| Format | Example | Support |
|--------|---------|---------|
| Basic think tags | `<think>...</think>Answer` | ✓ |
| With assistant prefix | `<|assistant|><think>...</think>Answer` | ✓ |
| With answer tags | `<think>...</think><answer>Answer</answer>` | ✓ |
| Combined | `<|assistant|><think>...</think><answer>Answer</answer>` | ✓ |
| No tags | `Just answer` | ✓ |

## Test Coverage

Created 4 comprehensive test files:

### 1. `test_think_tags_simple.py`
- Demonstrates the problem with concrete examples
- Shows before/after behavior for each constraint type
- **All tests pass** ✓

### 2. `test_think_removal_direct.py`
- Direct unit tests of `_remove_thinking_section()` function
- Tests all tag format variations
- Real-world example from DeepSeek-R1
- **All tests pass** ✓

### 3. `test_ifeval_think_tags.py`
- Comprehensive integration tests
- Tests multiple instruction types with think tags
- Verifies keywords in think section don't count
- **Design complete** (requires full environment)

### 4. `test_ifeval_think_final.py`
- End-to-end tests with realistic reasoning model outputs
- Tests DeepSeek-R1 and Claude-style responses
- Validates multiple constraints simultaneously
- **Design complete** (requires full environment)

## Impact on Training

This fix is **critical** for RLHF/GRPO training with reasoning models:

### Before Fix ❌
```python
# Example: No comma instruction
response = "<think>Let me think, carefully.</think>Answer without commas"
score = compute_score(response, instruction_no_comma)
# score = 0.0  (WRONG - failed due to think section)
```

### After Fix ✓
```python
# Example: No comma instruction
response = "<think>Let me think, carefully.</think>Answer without commas"
score = compute_score(response, instruction_no_comma)
# score = 1.0  (CORRECT - think section ignored)
```

### Training Benefits

1. **Accurate Rewards**: Models get correct feedback for their actual answers
2. **Better Convergence**: No false negatives from think sections
3. **Reasoning Encouraged**: Models can freely use think tags for CoT
4. **Instruction Following**: Properly evaluates the instructed content only

## Compatibility

- ✓ Fully backward compatible with non-reasoning models
- ✓ Works with official ifeval library
- ✓ Works with IFEvalG from open-instruct
- ✓ No performance impact on responses without think tags
- ✓ Maintains all existing functionality

## Usage

No changes needed! The preprocessing is automatic:

```python
from verl.utils.reward_score import default_compute_score

# Works automatically for reasoning models
response = """<think>
Complex reasoning with commas, uppercase, and many words here.
</think>

Final answer following all instructions."""

ground_truth = {
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}]
}

score = default_compute_score(
    data_source="google/IFEval",
    solution_str=response,
    ground_truth=ground_truth
)
# Returns correct score based only on the final answer
```

## Commit History

1. **6762d0c**: Initial IFEval migration from open-instruct
2. **30d4358**: Integrated IFEvalG library directly
3. **45f82de**: Added think tag removal for reasoning models ← **This fix**

## References

- Open-instruct implementation: `open_instruct/ground_truth_utils.py`
- IFEval paper: Zhou et al. "Instruction-Following Evaluation for Large Language Models" (2023)
- DeepSeek-R1: Uses `<think>` tags for reasoning
- Claude: Uses `<think>` and `<answer>` tags
