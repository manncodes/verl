# Beyond the 80/20 Rule: High-Entropy Token Training

## Overview

This feature implements the key insight from ["Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning"](https://arxiv.org/abs/2506.01939), which shows that focusing policy gradient updates on only the highest-entropy tokens leads to more efficient and effective RL training.

## Motivation

In reinforcement learning for reasoning tasks (like math problem solving), the paper discovered:

1. **Token Distribution**: In typical reasoning responses, only ~20% of tokens are high-entropy "forking tokens" that steer the reasoning path
2. **Gradient Contribution**: Low-entropy tokens (routine phrases, punctuation) contribute little to meaningful learning
3. **Training Efficiency**: Focusing gradients on high-entropy tokens improves both efficiency and final performance

## Configuration

Enable high-entropy token training by setting `entropy_top_ratio` in the actor configuration:

```yaml
actor_rollout_ref:
  actor:
    entropy_top_ratio: 0.2  # Only use top 20% highest-entropy tokens
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_top_ratio` | `Optional[float]` | `None` | Fraction of response tokens to use for training (0.0-1.0). When `None`, all tokens are used. |

## How It Works

1. **Entropy Computation**: During the forward pass, entropy is computed for each token
2. **Token Selection**: The `get_global_entropy_top_mask` function selects tokens with the highest entropy across the entire batch
3. **Masked Loss**: Policy gradient loss is computed only on selected tokens
4. **Metrics**: The actual token ratio is logged as `actor/entropy_token_ratio`

### Implementation Details

The core function is:

```python
def get_global_entropy_top_mask(
    entropy: torch.Tensor,      # [B, S] token entropies
    response_mask: torch.Tensor, # [B, S] valid response tokens
    top_ratio: float = 0.2       # fraction to keep
) -> torch.Tensor:               # [B, S] binary mask
```

This returns a binary mask where:
- `1` = token is in the top `top_ratio` by entropy (will contribute to loss)
- `0` = token is not selected (gradient is zero)

## Compatibility

The `entropy_top_ratio` feature works with all policy loss modes:

| Loss Mode | Supported | Notes |
|-----------|-----------|-------|
| `vanilla` | Yes | Standard PPO with high-entropy token selection |
| `gpg` | Yes | GPG loss with high-entropy token selection |
| `clip_cov` | Yes | Clip-Cov loss with high-entropy token selection |
| `kl_cov` | Yes | KL-Cov loss with high-entropy token selection |
| `gspo` | Yes* | Parameter accepted but not applied (sequence-level loss) |
| `geo_mean` | Yes* | Parameter accepted but not applied (sequence-level loss) |

*Note: For sequence-level loss modes (gspo, geo_mean), the parameter is accepted for interface compatibility but the mask is not applied since these methods aggregate across the entire sequence.

## Example Usage

### Basic Training Script

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    ...
```

### Combining with Other Techniques

```bash
# With GRPO advantage estimation
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    ...

# With Clip-Cov loss mode
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
    actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    ...
```

## Metrics

When enabled, the following metric is logged:

- `actor/entropy_token_ratio`: Actual ratio of tokens selected (should be close to configured `entropy_top_ratio`)

## Recipe

See the example recipe at:
```
recipe/entropy_token_rule/run_entropy_top20.sh
```

## References

- **Paper**: [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939)
- **Original Repository**: https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR

## Citation

```bibtex
@article{wang2025beyond,
  title={Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning},
  author={Wang, Shenzhi and others},
  journal={arXiv preprint arXiv:2506.01939},
  year={2025}
}
```
