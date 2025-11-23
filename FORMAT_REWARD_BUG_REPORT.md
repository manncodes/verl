# Format Reward Bug Report and Fix

## Summary

Critical bug found in `format_reward()` and `format_reward_strict()` functions that incorrectly validate thinking format patterns.

## The Bug

**Issue**: Functions incorrectly accept `<evaluation>INCORRECT</evaluation>` as valid.

**Root Cause**: The regex pattern `.*?CORRECT.*?` matches the substring "CORRECT" inside "IN**CORRECT**", causing false positives.

### Test Results

| Function | Original Success Rate | Fixed Success Rate |
|----------|---------------------|-------------------|
| `format_reward (ORIGINAL)` | 50.0% (4/8 tests) | — |
| `format_reward (FIXED)` | — | 100.0% (8/8 tests) |
| `format_reward_strict (ORIGINAL)` | 50.0% (4/8 tests) | — |
| `format_reward_strict (FIXED)` | — | 100.0% (8/8 tests) |

## Failed Test Cases (Original)

1. ❌ `<think>...</think><evaluation>INCORRECT</evaluation><answer>42</answer>`
   - Expected: 0.0 (reject)
   - Got: 1.0 (accepted) ← **BUG**

2. ❌ `<think>...</think><evaluation>NOT INCORRECT</evaluation><answer>42</answer>`
   - Expected: 0.0 (reject - no CORRECT)
   - Got: 1.0 (accepted) ← **BUG**

3. ❌ Multiple INCORRECTs without CORRECT
   - Expected: 0.0 (reject)
   - Got: 1.0 (accepted) ← **BUG**

4. ❌ `<think>...</think><evaluation>incorrect</evaluation><answer>42</answer>`
   - Expected: 0.0 (reject - only lowercase incorrect)
   - Got: 1.0 (accepted) ← **BUG**

## The Fix

### Solution 1: Word Boundaries (Recommended)

Use `\b` word boundaries to match CORRECT and INCORRECT as complete words:

```python
# Before (BUGGY):
r'.*?<evaluation>.*?CORRECT.*?</evaluation>'

# After (FIXED):
r'.*?<evaluation>.*?\bCORRECT\b.*?</evaluation>'
```

### Solution 2: Negative Lookbehind (Alternative)

Use negative lookbehind to ensure CORRECT is not preceded by "IN":

```python
# Alternative fix:
r'.*?<evaluation>.*?(?<!IN)CORRECT.*?</evaluation>'
```

## Implementation

### Fixed `format_reward()`:

```python
pattern_direct = re.compile(
    r'<think>'
    r'(?:.*?<planning>.*?</planning>)?'
    r'(?:.*?<draft answer>.*?</draft answer>)?'
    r'.*?</think>'
    r'.*?<evaluation>.*?\bCORRECT\b.*?</evaluation>'  # ← FIXED
    r'.*?<answer>.+?</answer>',
    re.DOTALL | re.IGNORECASE
)

pattern_retry = re.compile(
    r'<think>'
    r'(?:.*?<planning>.*?</planning>)?'
    r'(?:.*?<draft answer>.*?</draft answer>)?'
    r'.*?</think>'
    r'.*?<evaluation>.*?\bINCORRECT\b.*?</evaluation>'  # ← FIXED
    r'.*?<think>'
    r'(?:.*?<planning>.*?</planning>)?'
    r'(?:.*?<draft answer>.*?</draft answer>)?'
    r'.*?</think>'
    r'.*?<evaluation>.*?\bCORRECT\b.*?</evaluation>'  # ← FIXED
    r'.*?<answer>.+?</answer>',
    re.DOTALL | re.IGNORECASE
)
```

### Fixed `format_reward_strict()`:

```python
# Check for evaluation tag with CORRECT (must exist at least once)
if not re.search(r'<evaluation>.*?\bCORRECT\b.*?</evaluation>',
                 predict_str, re.DOTALL | re.IGNORECASE):
    return 0.0

# Find last CORRECT evaluation position
correct_eval = list(re.finditer(r'<evaluation>.*?\bCORRECT\b.*?</evaluation>',
                               predict_str, re.DOTALL | re.IGNORECASE))[-1]
```

## Test Coverage

### Valid Patterns (Should Return 1.0) ✅

- `<think>...</think><evaluation>CORRECT</evaluation><answer>...</answer>`
- `<think>...</think><evaluation>This is CORRECT</evaluation><answer>...</answer>`
- `<think>...</think><evaluation>correct</evaluation><answer>...</answer>` (case insensitive)
- `<think><planning>...</planning>...</think><evaluation>CORRECT</evaluation><answer>...</answer>`
- `<think>...</think><evaluation>INCORRECT</evaluation><think>...</think><evaluation>CORRECT</evaluation><answer>...</answer>` (retry pattern)

### Invalid Patterns (Should Return 0.0) ✅

- `<think>...</think><evaluation>INCORRECT</evaluation><answer>...</answer>` ← **Fixed**
- `<think>...</think><evaluation>incorrect</evaluation><answer>...</answer>` ← **Fixed**
- `<think>...</think><evaluation>NOT INCORRECT</evaluation><answer>...</answer>` ← **Fixed**
- `<think>...</think><evaluation>WRONG</evaluation><answer>...</answer>`
- Missing tags (think, evaluation, or answer)
- Wrong tag order
- Duplicate answer tags

## Files

- `test_format_reward.py` - Original code with comprehensive test suite
- `format_reward_fixed.py` - Fixed implementations with both approaches
- `test_fixed_versions.py` - Comparison tests showing bug and fix
- `FORMAT_REWARD_BUG_REPORT.md` - This report

## Recommendation

**Use the word boundary approach (`\b`)** as it's clearer and more maintainable than negative lookbehind. Both fixed versions achieve 100% test pass rate.
