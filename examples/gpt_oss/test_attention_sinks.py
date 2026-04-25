"""Test that gpt-oss attention sinks are wired correctly end-to-end.

Why this exists
---------------
gpt-oss attention layers use learnable per-head "sink" scores that are added
to the softmax denominator (acts like a virtual always-attended token). The
sinks shipped with the bf16 checkpoint are a real learned parameter — drop
them or bypass them and the logits drift, sometimes catastrophically.

Concretely, the upstream ecosystem has had repeated bugs where sinks are
silently bypassed:

  * HF SDPA does not implement sinks         → unsloth #3142
  * FlashAttention 2 errors with             → vllm #22331, #22279
    "Sinks are only supported in FA3"
  * FlashInfer backend lacks sink support    → vllm #30919
  * NVIDIA TransformerEngine lacks support   → NVIDIA/TE #2070
  * Only eager / FA3 / TRTLLM are correct

Our verl recipe forces `attn_implementation=eager` precisely because of
this. The test below is the safety net: if a future patch flips the actor
to SDPA (or someone zeros the sinks via a config typo), this test fails
loudly before we waste GPU hours training a corrupted model.

Invariants verified
-------------------
 1. Every attention layer exposes a `sinks` parameter of the right shape.
 2. The sinks tensor is not all-zero (i.e. the checkpoint actually loaded).
 3. The model is using eager attention (the only HF backend that honours
    sinks correctly today).
 4. Forward logits differ when sinks are zeroed → proves sinks are wired
    into the attention computation, not silently dropped.
 5. Backward through the loss accumulates non-zero gradient on sinks →
    proves they participate in autograd and can actually be trained.

Usage:
    python examples/gpt_oss/test_attention_sinks.py \
        --model-dir ~/models/gpt-oss-20b-bf16
"""

import argparse
import os
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg: str) -> None:
    print(f"[sinks] {msg}", flush=True)


def find_sink_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return (name, param) for every parameter whose name contains 'sinks'."""
    out = []
    for name, param in model.named_parameters():
        # transformers gpt-oss exposes them as ".sinks"; tolerate other names too.
        if name.endswith(".sinks") or ".sinks" in name or name.endswith(".attention_sinks"):
            out.append((name, param))
    return out


def attn_implementation_in_use(model: torch.nn.Module) -> str:
    """Best-effort introspection of which attention path the model is using."""
    cfg_attr = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config, "attn_implementation", None
    )
    if cfg_attr is not None:
        return cfg_attr
    # Walk to the first attention module and read its private attribute.
    for module in model.modules():
        impl = getattr(module, "_attn_implementation", None)
        if impl is not None:
            return impl
    return "<unknown>"


def build_inputs(tokenizer, device: torch.device, seq_len: int) -> dict[str, Any]:
    enc = tokenizer(
        ["The capital of France is"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def assert_(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/models/gpt-oss-20b-bf16"),
        help="bf16 checkpoint produced by prepare_model.py",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device-map", default=None, help="HF device_map; default 'auto' on multi-GPU")
    parser.add_argument(
        "--logit-tol",
        type=float,
        default=1e-3,
        help="max-abs logit delta required between with-sinks and zeroed-sinks; if smaller, sinks are not wired in",
    )
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        log(f"ERROR: {args.model_dir}/config.json not found. Run prepare_model.py first.")
        return 1

    if args.device_map is None:
        device_map = "auto" if torch.cuda.device_count() >= 2 else None
    elif args.device_map.lower() == "none":
        device_map = None
    else:
        device_map = args.device_map

    log(f"loading {args.model_dir} (device_map={device_map}, attn=eager)")
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)
    if device_map is None:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # deterministic forward; we'll re-enable grad explicitly for the bwd test

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # ---- 1. discover sink parameters --------------------------------------
    sink_params = find_sink_params(model)
    n_layers = getattr(model.config, "num_hidden_layers", None)
    log(f"found {len(sink_params)} sink parameters across {n_layers} layers")
    assert_(
        len(sink_params) > 0,
        "no parameters named '*.sinks' found — is this really a gpt-oss model? "
        "(transformers calls the per-layer learnable sink scores `sinks`)",
    )
    if n_layers is not None:
        assert_(
            len(sink_params) == n_layers,
            f"expected exactly {n_layers} sink params (one per layer), got {len(sink_params)}",
        )

    # Shape sanity: should be a per-head 1D tensor.
    n_heads = getattr(model.config, "num_attention_heads", None)
    name0, p0 = sink_params[0]
    log(f"  example: {name0} shape={tuple(p0.shape)} dtype={p0.dtype}")
    assert_(p0.dim() == 1, f"sinks should be 1D (per-head); got shape {tuple(p0.shape)}")
    if n_heads is not None:
        assert_(
            p0.shape[0] == n_heads,
            f"sinks dim ({p0.shape[0]}) != num_attention_heads ({n_heads})",
        )

    # ---- 2. sinks are not all-zero ----------------------------------------
    abs_sums = torch.tensor([p.detach().float().abs().sum().item() for _, p in sink_params])
    log(
        f"  sink magnitude: min|sum|={abs_sums.min():.4f}  "
        f"max|sum|={abs_sums.max():.4f}  mean|sum|={abs_sums.mean():.4f}"
    )
    assert_(
        (abs_sums > 0).all(),
        "at least one sink parameter is exactly zero — checkpoint did not load the learned sinks",
    )

    # ---- 3. eager attention is in use -------------------------------------
    impl = attn_implementation_in_use(model)
    log(f"  attn_implementation = {impl}")
    assert_(
        impl == "eager",
        f"attention is using {impl!r}, but only 'eager' (or FA3/TRTLLM, which "
        "transformers doesn't use here) honours sinks correctly. SDPA, FA2, and "
        "FlashInfer all silently produce wrong logits for gpt-oss.",
    )

    # ---- 4. forward differs when sinks are zeroed -------------------------
    inputs = build_inputs(tokenizer, next(model.parameters()).device, args.seq_len)
    with torch.no_grad():
        logits_with_sinks = model(**inputs).logits.float().cpu()

    saved = [p.detach().clone() for _, p in sink_params]
    try:
        with torch.no_grad():
            for _, p in sink_params:
                p.zero_()
            logits_no_sinks = model(**inputs).logits.float().cpu()
    finally:
        # Restore so the backward test below sees the real values.
        with torch.no_grad():
            for (_, p), saved_p in zip(sink_params, saved):
                p.copy_(saved_p)

    delta = (logits_with_sinks - logits_no_sinks).abs()
    log(
        f"  logit delta (with-sinks vs zeroed): max={delta.max():.6f}  "
        f"mean={delta.mean():.6f}"
    )
    assert_(
        delta.max().item() > args.logit_tol,
        f"zeroing the sinks did not change the logits (max-abs delta={delta.max():.2e} "
        f"<= tol={args.logit_tol:.0e}). The sinks are NOT being used by the attention "
        "kernel — likely an SDPA / FA2 / FlashInfer fallback. This is a silent "
        "correctness bug; do not train.",
    )

    # ---- 5. backward accumulates grad on sinks ----------------------------
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    for _, p in sink_params:
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None

    # Use a labeled forward so the model returns a CE loss internally.
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    out = model(**inputs, labels=labels)
    loss = out.loss
    log(f"  labelled forward loss = {loss.item():.4f}")
    loss.backward()

    grad_zeros = []
    grad_nans = []
    for name, p in sink_params:
        if p.grad is None:
            grad_zeros.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            grad_nans.append(name)
        if p.grad.abs().sum().item() == 0:
            grad_zeros.append(name)
    assert_(
        not grad_nans,
        f"non-finite grads on {len(grad_nans)} sink params, e.g. {grad_nans[:3]}",
    )
    assert_(
        not grad_zeros,
        f"{len(grad_zeros)} sink params have None/zero grad after backward, "
        f"e.g. {grad_zeros[:3]}. They are not in the autograd graph — sinks "
        "won't be trained.",
    )
    log(f"  all {len(sink_params)} sink params received finite, non-zero gradient")

    log("attention sinks test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
