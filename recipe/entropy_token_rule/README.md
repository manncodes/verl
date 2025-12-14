# Beyond the 80/20 Rule: High-Entropy Token Training

This recipe implements the key insight from the paper ["Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning"](https://arxiv.org/abs/2506.01939).

## Background

The paper discovers that in reinforcement learning for reasoning tasks, not all tokens contribute equally to learning. Specifically:

- **High-entropy tokens** (the "forking tokens" that steer the reasoning path) represent only ~20% of generated tokens
- **Low-entropy tokens** (routine tokens like punctuation, common phrases) make up ~80% of tokens but contribute little to learning
- Training only on high-entropy tokens is more efficient and often leads to better performance

## Key Configuration

The main configuration parameter is `entropy_top_ratio`:

```yaml
actor_rollout_ref.actor.entropy_top_ratio: 0.2  # Train only on top 20% high-entropy tokens
```

This parameter works with any policy loss mode (`vanilla`, `gpg`, `clip_cov`, `kl_cov`).

## How It Works

1. During training, entropy is computed for each token in the response
2. The `get_global_entropy_top_mask` function selects the top `entropy_top_ratio` fraction of tokens by entropy
3. The policy gradient loss is computed only on these high-entropy tokens
4. This focuses learning on the "decision points" in reasoning rather than routine text

## Example Usage

### Basic Usage

Run the example script:

```bash
bash recipe/entropy_token_rule/run_entropy_top20.sh
```

### Custom Entropy Ratio

You can experiment with different entropy ratios:

```bash
# More aggressive: Only top 10% highest-entropy tokens
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.entropy_top_ratio=0.1 \
    ...

# Less aggressive: Top 30% highest-entropy tokens
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.entropy_top_ratio=0.3 \
    ...
```

### Combining with Other Loss Modes

The entropy token selection works with any loss mode:

```bash
# With GPG loss
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    ...

# With Clip-Cov loss (from Entropy Mechanism paper)
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
    actor_rollout_ref.actor.entropy_top_ratio=0.2 \
    ...
```

## Metrics

When `entropy_top_ratio` is enabled, the following metric is logged:

- `actor/entropy_token_ratio`: The actual ratio of tokens selected (should be close to `entropy_top_ratio`)

## References

- Paper: [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939)
- Original Repository: https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR

## Citation

```bibtex
@article{wang2025beyond,
  title={Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning},
  author={Wang, Shenzhi and others},
  journal={arXiv preprint arXiv:2506.01939},
  year={2025}
}
```
