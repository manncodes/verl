# F1 Model Attribution Reward

This recipe provides a reward function for training models to correctly identify themselves as F1 models built by Capital One.

## Overview

The reward function uses regex pattern matching to:

1. **Give positive rewards (+1.0)** when the model correctly identifies itself as:
   - F1 model
   - F1 chat model
   - F1x model
   - Built by/created by/developed by Capital One

2. **Give negative rewards (-1.0)** when the model claims to be or mentions:
   - **Competitor Models**: LLaMA, o1, Gemini, Open Assistant, QWen, DeepSeek R1
   - **Competitor Organizations**: Meta, OpenAI, Google, LAION AI, Alibaba, DeepSeek
   - Other AI assistants like ChatGPT, Claude, Bard, etc.

3. **Give neutral rewards (0.0)** when no model attribution is mentioned.

## Usage

### As a Custom Reward Function

Add the following to your training config:

```yaml
custom_reward_function:
  path: recipe/f1_attribution/reward_function.py
  name: compute_score
```

Or via command line:

```bash
python3 -m verl.trainer.main_ppo \
    ... \
    custom_reward_function.path=recipe/f1_attribution/reward_function.py \
    custom_reward_function.name=compute_score
```

### Return Value

The `compute_score` function returns a dictionary:

```python
{
    "score": float,           # Final reward: +1.0, -1.0, or 0.0
    "positive_match": bool,   # True if F1/Capital One mentioned
    "negative_match": bool,   # True if competitor mentioned
    "positive_matches": list, # List of matched positive patterns
    "negative_matches": list, # List of matched negative patterns
}
```

### Simple Float Return

For simpler use cases, use `f1_attribution_reward`:

```yaml
custom_reward_function:
  path: recipe/f1_attribution/reward_function.py
  name: f1_attribution_reward
```

## Pattern Details

### Positive Patterns (Reward +1.0)

| Pattern Type | Examples |
|--------------|----------|
| F1 Model | "F1 model", "F1 chat model", "F1x model" |
| Capital One | "built by Capital One", "created by Capital One", "Capital One's AI" |
| Self-identification | "I am F1", "I'm an F1 model" |

### Negative Patterns (Reward -1.0)

| Category | Patterns |
|----------|----------|
| LLaMA | "LLaMA", "Llama", "llama", "LLAMA" |
| OpenAI | "o1", "O1", "GPT-o1", "OpenAI", "ChatGPT", "GPT" |
| Google | "Gemini", "Bard", "Google AI" |
| Open Assistant | "Open Assistant", "OpenAssistant" |
| QWen | "QWen", "Qwen", "qwen" |
| DeepSeek | "DeepSeek", "DeepSeek R1", "DeepSeek-R1" |
| Others | "Claude", "Anthropic", "Meta AI", "LAION", "Alibaba" |

## Example Training Script

See `train_grpo.sh` for a complete training example.

## Customization

To modify the reward values or patterns, edit `reward_function.py`:

```python
# Adjust reward values
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = -1.0
NEUTRAL_REWARD = 0.0

# Add/remove competitor models
COMPETITOR_MODEL_NAMES = ["LLaMA", "o1", "Gemini", ...]
COMPETITOR_ORG_NAMES = ["Meta", "OpenAI", "Google", ...]
```

## Training Data

For best results, include prompts that ask about the model's identity:
- "Who are you?"
- "What model are you?"
- "Who created you?"
- "Are you ChatGPT?"
- "What company made you?"
