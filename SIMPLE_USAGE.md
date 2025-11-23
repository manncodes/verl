# Simple Usage - Drop-in Replacement

## 🚀 Quick Start (Copy & Paste)

### 1. Copy the File

```bash
# Just copy this one file to your project:
cp advanced_repetition_detector.py /path/to/your/project/
```

### 2. Use It (Same Interface as Before)

```python
from advanced_repetition_detector import apply_repetition_penalty

# In your reward function:
def calculate_reward(response, ground_truth):
    # Your base reward calculation
    base_reward = your_reward_function(response, ground_truth)

    # Apply repetition penalty (same interface!)
    final_reward = apply_repetition_penalty(
        text=response['content'],
        base_reward=base_reward,
        severity="moderate"  # lenient, moderate, strict, very_strict
    )

    return final_reward
```

That's it! **No code changes needed** - same function name, same parameters.

## ✅ What Changed (Under the Hood)

**Better Calibration:**
- ✅ Keeps scores intact for fine/borderline cases
- ✅ Only penalizes clear, egregious repetition
- ✅ Catches truly inhumane/robotic responses
- ✅ Zero false positives on acceptable text

**Same Interface:**
- ✓ Function name: `apply_repetition_penalty` (unchanged)
- ✓ Parameters: `text`, `base_reward`, `severity`, `return_details` (unchanged)
- ✓ Return values: Same as before
- ✓ Severity levels: Same names (lenient, moderate, strict, very_strict)

## 📊 Results Comparison

| Example | Old Score | New Score | Old Penalty | New Penalty |
|---------|-----------|-----------|-------------|-------------|
| Good quality | 0.000 | 0.000 | None | None ✓ |
| Minor repetition | 0.113 | 0.000 | None | None ✓ |
| Informative template | 0.177 | 0.000 | 15% | None ✓ |
| **Egregious repetition** | 0.615 | 0.541 | 70% | 35% ✓ |

**Key Improvement:** No false positives on acceptable text!

## 🎯 Recommended Severity

For your use case ("only penalize true positives, keep score if fine"):

```python
# Use "moderate" (recommended)
final_reward = apply_repetition_penalty(
    text=response['content'],
    base_reward=base_reward,
    severity="moderate"  # ← This one!
)
```

**Why moderate?**
- No penalty for scores < 0.30 (acceptable quality)
- Small penalties for minor issues (0.30-0.45)
- Clear penalties for egregious cases (> 0.45)

## 🔍 With Details (Optional)

```python
# Get detailed breakdown
final_reward, details = apply_repetition_penalty(
    text=response['content'],
    base_reward=base_reward,
    severity="moderate",
    return_details=True  # ← Add this
)

# See what happened
print(f"Repetition score: {details['repetition_score']:.3f}")
print(f"Penalty multiplier: {details['penalty_multiplier']:.3f}")
print(f"Penalty applied: {details['penalty_applied']:.3f}")

# Log if penalty was applied
if details['penalty_applied'] > 0:
    print(f"⚠️ Repetition detected: {details['components']}")
```

## 📋 Severity Levels (Recalibrated)

### Lenient (< 0.40 no penalty)
- Very forgiving
- Use for creative tasks

```python
final_reward = apply_repetition_penalty(text, base_reward, severity="lenient")
```

### Moderate (< 0.30 no penalty) ⭐ **RECOMMENDED**
- Balanced approach
- Only penalizes clear issues
- **Best for most use cases**

```python
final_reward = apply_repetition_penalty(text, base_reward, severity="moderate")
```

### Strict (< 0.20 no penalty)
- More aggressive
- Use for quality control

```python
final_reward = apply_repetition_penalty(text, base_reward, severity="strict")
```

### Very Strict (< 0.15 no penalty)
- Maximum strictness
- Use when quality is critical

```python
final_reward = apply_repetition_penalty(text, base_reward, severity="very_strict")
```

## ✨ That's It!

Just copy `advanced_repetition_detector.py` and use it. Same interface, better performance!

## 🧪 Test It

```python
from advanced_repetition_detector import apply_repetition_penalty

# Test on your data
test_text = """
Your model output here...
"""

reward, details = apply_repetition_penalty(
    test_text,
    1.0,
    severity="moderate",
    return_details=True
)

print(f"Score: {details['repetition_score']:.3f}")
print(f"Penalty: {details['penalty_applied']:.3f}")
print(f"Final reward: {reward:.3f}")
```

## 📞 Questions?

**Q: Do I need to change my code?**
A: No! Same function name and parameters.

**Q: Which severity should I use?**
A: Start with `"moderate"` (recommended for your use case).

**Q: Will this work with my existing reward function?**
A: Yes! It's a drop-in replacement.

**Q: Do I need numpy?**
A: No, but it's faster with numpy. Works fine without it.

**Q: How do I know if it's working?**
A: Use `return_details=True` and check the logs.
