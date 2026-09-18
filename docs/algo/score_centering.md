# Score Centering

Last updated: 2026-09-18.

📝 [ArXiv](https://arxiv.org/abs/2609.20807) | 💻 [Reference implementation](https://github.com/martin-marek/score-centering)

Score Centering stabilizes off-policy reinforcement learning under a **training/inference mismatch (TIM)** — the gap between the rollout policy (e.g. vLLM, lower precision) and the training policy (e.g. FSDP, full precision) — by canceling the *drift* this mismatch introduces into REINFORCE's score, rather than by reweighting or truncating it the way importance sampling (IS) does. It can be used on its own or composed with truncated (TIS) or masked (IcePop/MIS) importance sampling; see [rollout_corr.md](rollout_corr.md) for the latter.

## Why REINFORCE drifts off-policy

REINFORCE's score, `E_q[grad log p]`, is only zero when the trainer `p` and the rollout sampler `q` are the same distribution. Under TIM they aren't, so every step carries a bias that importance-sampling corrections (TIS/IcePop) can down-weight but not remove.

## The control variate

Score centering cancels this drift with a control variate built from the rollout's **transported top-k vocabulary head**: the `k` token ids the sampler assigned the highest probability to at each generated position, together with their log-probabilities under the sampler. The vocabulary tail beyond that head is modeled as proportional to the trainer's own tail (exact once `k` covers the full vocabulary), which reduces the correction to a finite sum over the head:

```
p_head, q_head = exp(train_head_logp), exp(rollout_head_logp)
p_tail, q_tail = 1 - Σ p_head,          1 - Σ q_head
rho   = q_tail / p_tail
alpha = rho * weight_fn(1 / rho)
residual   = q_head * weight_fn(p_head / q_head) - alpha * p_head
correction = Σ [ stopgrad(residual) * train_head_logp ]
```

`weight_fn` is the identity-weight `torch.ones_like` for score centering alone, or the same clamp/band function `rollout_is_threshold` already drives for TIS/IcePop (`score_centering_weight_fn`), so composing the two reuses one threshold knob. The correction is added to the (advantage-weighted) REINFORCE score alongside the sampled token's own log-prob — `score = weight_fn(ratio) * token_logp - correction` — so it is a drop-in additive term for *any* policy loss (`vanilla`, `gspo`, `cispo`, REINFORCE, ...), not a new registered `policy_loss.loss_mode`. It is identically zero, gradient included, whenever the trainer and rollout head agree exactly (on-policy).

See `verl.utils.torch_functional.score_centering_correction[_from_logits]` for the implementation and `tests/utils/test_score_centering_on_cpu.py` for the on-policy-vanishes and TIS/IcePop-composition invariants.

## Integration in verl

- `actor_rollout_ref.actor.calculate_score_centering: true` turns on the correction in `FSDPEngineWithLMHead.prepare_model_outputs`: it gathers the trainer's log-probs at the rollout's top-k ids from the same full-vocabulary logits already computed for entropy/Σπ², narrows/repacks them exactly like those existing per-token quantities, and returns `model_output["score_centering_correction"]`.
- `ppo_loss` (`verl/workers/utils/losses.py`) adds `advantages.detach() * correction` to whichever `policy_loss.loss_mode` is configured, whenever the engine populated `score_centering_correction` — see `score_centering_loss`.
- See `tests/workers/test_score_centering_engine_on_cpu.py` for a CPU-only check that the engine's narrowing/repacking wiring reproduces the correction computed directly from the dense logits.

### Current scope and required follow-up

This is a first pass, scoped to what could be verified on CPU in this session (no GPU/vLLM/multi-node cluster available). Before training with it, a contributor with that hardware needs to:

1. **Populate `rollout_topk_ids` / `rollout_topk_logprobs` from the rollout engine.** Nothing in this pass captures them from vLLM or SGLang yet. For vLLM, this means requesting `SamplingParams(logprobs=k)` (verl currently only ever requests `logprobs=0`, i.e. the sampled token alone — see `verl/workers/rollout/vllm_rollout/vllm_async_server.py`) and threading the returned per-position `dict[token_id, Logprob]` into the batch as two ragged (prompt+response-length, `k`) tensors aligned with `input_ids`, the same way `responses`/`rollout_log_probs` are threaded today.
2. **Wire the new flag into a trainer's actor-update path**, e.g. `verl/trainer/ppo/v1/trainer_base.py`'s `_update_actor`, alongside its existing `calculate_entropy`.
3. **Extend engine coverage.** Only the FSDP engine's dense path (`use_remove_padding=false`, `use_fused_kernels=false`) is implemented; `calculate_score_centering=true` raises `NotImplementedError` for Ulysses sequence parallelism or fused kernels (neither materializes the full-vocabulary logits the head gather needs), and Megatron/automodel/torchtitan have no hook at all yet — mirroring exactly where `calculate_sum_pi_squared` (Σπ², see [otb.md](otb.md)) currently stands.

## Citation

```
@misc{marek2026scorecentering,
      title={Score Centering Stabilizes Off-policy Reinforcement Learning},
      author={Martin Marek and Max Ryabinin},
      year={2026},
      eprint={2609.20807},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.20807},
}
```
