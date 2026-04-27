"""Sinks-aware attention via PyTorch's flex_attention.

The community finding (Unsloth) that motivates this file:
  * FA3 produces wrong gradients on gpt-oss because it doesn't implement the
    backward through attention sinks.
  * FA2 / SDPA / FlashInfer silently DROP the sinks (no error, just wrong
    logits — the bug class our test_attention_sinks.py probe catches).
  * Eager attention is correct but O(N^2) memory and dog-slow at long
    contexts.
  * PyTorch flex_attention can express the sink mechanism via score_mod,
    runs a Triton-compiled kernel, and supports correct backward — closes
    the gap.

The trick (verbatim from gpt-oss's formula in transformers):

    attn = softmax([Q K^T / sqrt(D) | sinks])      # cat sinks as +1 logit
    output = attn[..., :-1] @ V                     # drop sink column after

Implemented here by augmenting K and V with one extra "sink position":
  * K_aug = [K | dummy]
  * V_aug = [V | 0]            (V=0 at the sink, so it contributes nothing)
  * score_mod sets the score at the sink position to sinks[h]
  * mask_mod always lets queries attend to the sink (so it absorbs
    probability mass) and applies causal + optional sliding-window to the
    real positions.

Math equivalence:
  flex_attention(Q, K_aug, V_aug, score_mod=set sink score, mask=as above)
    = softmax([scores | sinks]) @ [V | 0]
    = softmax([scores | sinks])[..., :-1] @ V    (because V_sink=0)
    = eager-with-sinks output

Status: NOT verified end-to-end on gpt-oss-20b yet. The self_test() in this
file checks forward+backward parity vs the eager reference at fp32 with a
tight tolerance — run it on the H100 box before trusting this in a training
run:

    python -m examples.gpt_oss.flex_attention_sinks --self-test
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn.functional as F


def eager_attention_with_sinks(
    q: torch.Tensor,           # [B, H, Lq, D]
    k: torch.Tensor,           # [B, H, Lkv, D]
    v: torch.Tensor,           # [B, H, Lkv, Dv]
    sinks: torch.Tensor,       # [H]
    sliding_window: int | None = None,
    is_causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference eager implementation. Mirrors gpt-oss's formula in transformers."""
    B, H, Lq, D = q.shape
    Lkv = k.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, Lq, Lkv]

    if is_causal:
        # gpt-oss's causal: q_idx >= kv_idx in the LAST Lq queries against all Lkv keys
        # (handles both same-length and Lq < Lkv cases)
        q_pos = torch.arange(Lq, device=scores.device).view(Lq, 1) + (Lkv - Lq)
        kv_pos = torch.arange(Lkv, device=scores.device).view(1, Lkv)
        causal_mask = q_pos < kv_pos
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if sliding_window is not None:
        q_pos = torch.arange(Lq, device=scores.device).view(Lq, 1) + (Lkv - Lq)
        kv_pos = torch.arange(Lkv, device=scores.device).view(1, Lkv)
        window_mask = (q_pos - kv_pos) > sliding_window
        scores = scores.masked_fill(window_mask, float("-inf"))

    sinks_bcast = sinks.view(1, H, 1, 1).expand(B, H, Lq, 1).to(scores.dtype)
    combined = torch.cat([scores, sinks_bcast], dim=-1)  # [B, H, Lq, Lkv+1]
    probs = F.softmax(combined, dim=-1)
    attn = probs[..., :-1]
    return attn @ v


def flex_attention_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    sliding_window: int | None = None,
    is_causal: bool = True,
    scale: float | None = None,
):
    """Same math, executed via flex_attention's compiled kernel."""
    try:
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )
    except ImportError as exc:
        raise RuntimeError(
            "flex_attention requires PyTorch 2.5+. Got an older torch."
        ) from exc

    B, H, Lq, D = q.shape
    Lkv = k.shape[2]
    Lkv_aug = Lkv + 1
    sink_idx = Lkv

    k_pad = torch.zeros(B, H, 1, D, dtype=k.dtype, device=k.device)
    v_pad = torch.zeros(B, H, 1, v.shape[-1], dtype=v.dtype, device=v.device)
    k_aug = torch.cat([k, k_pad], dim=2).contiguous()
    v_aug = torch.cat([v, v_pad], dim=2).contiguous()

    # Capture as a tensor — flex_attention compiles score_mod and traces this
    # closure as input to the kernel.
    sinks_dev = sinks.to(device=q.device, dtype=q.dtype).contiguous()

    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(kv_idx == sink_idx, sinks_dev[h], score)

    if sliding_window is None and is_causal:
        def mask_mod(b, h, q_idx, kv_idx):
            is_sink = kv_idx == sink_idx
            causal = q_idx >= kv_idx
            return is_sink | causal
    elif sliding_window is None and not is_causal:
        def mask_mod(b, h, q_idx, kv_idx):
            is_sink = kv_idx == sink_idx
            real = kv_idx < sink_idx
            return is_sink | real
    else:
        sw = sliding_window
        def mask_mod(b, h, q_idx, kv_idx):
            is_sink = kv_idx == sink_idx
            causal = q_idx >= kv_idx if is_causal else (kv_idx < sink_idx)
            in_window = (q_idx - kv_idx) <= sw
            return is_sink | (causal & in_window)

    block_mask = create_block_mask(
        mask_mod, B=B, H=H, Q_LEN=Lq, KV_LEN=Lkv_aug, device=q.device,
    )

    return flex_attention(
        q, k_aug, v_aug,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
    )


def self_test(
    *,
    batch: int = 2,
    heads: int = 8,
    seq: int = 256,
    head_dim: int = 64,
    sliding_window: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = 0,
    atol: float = 1e-4,
    bwd_atol: float = 1e-3,
) -> int:
    """Build small random tensors, run eager and flex with the same inputs,
    compare forward outputs and backward gradients on (q, k, v, sinks).
    Returns 0 on PASS, non-zero on FAIL."""
    if not torch.cuda.is_available():
        print("[flex-sinks-self-test] CUDA required", file=sys.stderr)
        return 1

    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    sinks = torch.randn(heads, device=device, dtype=dtype) * 0.5

    print(
        f"[flex-sinks-self-test] B={batch} H={heads} L={seq} D={head_dim} "
        f"dtype={dtype} sliding_window={sliding_window} atol_fwd={atol} atol_bwd={bwd_atol}"
    )

    # --- forward parity ---
    q_e = q.detach().clone().requires_grad_(True)
    k_e = k.detach().clone().requires_grad_(True)
    v_e = v.detach().clone().requires_grad_(True)
    s_e = sinks.detach().clone().requires_grad_(True)

    out_e = eager_attention_with_sinks(q_e, k_e, v_e, s_e, sliding_window=sliding_window)

    q_f = q.detach().clone().requires_grad_(True)
    k_f = k.detach().clone().requires_grad_(True)
    v_f = v.detach().clone().requires_grad_(True)
    s_f = sinks.detach().clone().requires_grad_(True)

    out_f = flex_attention_with_sinks(q_f, k_f, v_f, s_f, sliding_window=sliding_window)

    fwd_diff = (out_e - out_f).abs().max().item()
    fwd_pass = fwd_diff <= atol
    print(f"  forward max |Δ|: {fwd_diff:.3e}  rms: {out_e.float().pow(2).mean().sqrt().item():.3e}  "
          f"{'PASS' if fwd_pass else 'FAIL'}")

    # --- backward parity ---
    target = torch.randn_like(out_e)
    (out_e * target).sum().backward()
    (out_f * target).sum().backward()

    bwd_results = []
    for name, ge, gf in [
        ("grad q", q_e.grad, q_f.grad),
        ("grad k", k_e.grad, k_f.grad),
        ("grad v", v_e.grad, v_f.grad),
        ("grad sinks", s_e.grad, s_f.grad),
    ]:
        if ge is None or gf is None:
            print(f"  {name:14s}: SKIP (one side has no grad)")
            bwd_results.append(False)
            continue
        d = (ge - gf).abs().max().item()
        r = ge.float().pow(2).mean().sqrt().item()
        ok = d <= bwd_atol
        bwd_results.append(ok)
        print(f"  {name:14s}: max |Δ|={d:.3e}  rms={r:.3e}  {'PASS' if ok else 'FAIL'}")

    bwd_pass = all(bwd_results)
    print(f"  >> forward {'PASS' if fwd_pass else 'FAIL'}, backward {'PASS' if bwd_pass else 'FAIL'}")
    return 0 if (fwd_pass and bwd_pass) else 4


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="run forward+backward parity vs eager reference")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--atol-fwd", type=float, default=1e-4)
    p.add_argument("--atol-bwd", type=float, default=1e-3)
    args = p.parse_args()

    if not args.self_test:
        p.print_help()
        return 0

    dtype = getattr(torch, args.dtype)
    return self_test(
        batch=args.batch, heads=args.heads, seq=args.seq, head_dim=args.head_dim,
        sliding_window=args.sliding_window,
        dtype=dtype, atol=args.atol_fwd, bwd_atol=args.atol_bwd,
    )


if __name__ == "__main__":
    sys.exit(_cli())
