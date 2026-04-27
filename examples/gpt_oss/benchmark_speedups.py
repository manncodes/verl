"""Standalone benchmark for the speedup options on the gpt-oss-20b actor.

Reports three columns for each variant: correctness (max grad diff vs the
eager-with-sinks reference, lower is better), stability (peak memory + NaN
flag + OOM flag), timing (forward and backward wall-clock).

Variants tested
---------------
Attention (sliding window OFF; gpt-oss alternates layers between full and
sliding-window — toggle with --sliding-window):
  * eager (reference)        — gpt-oss's correct sinks formula in pytorch
  * flex (sinks-aware)       — flex_attention via flex_attention_sinks.py
  * sdpa (sink-bypass probe) — torch SDPA with NO sinks; included only to
                               quantify how wrong "silently bypass sinks"
                               actually is. Not a real candidate for use.

MoE (probed only if --moe is set; expensive to run blind, so opt-in):
  * pytorch (reference)      — naive per-expert loop with gpt-oss's clamped GLU
  * sonic-moe (vanilla)      — sonic-moe MoE class with its baked-in SwiGLU.
                               INCLUDED ONLY AS A LOWER BOUND ON THROUGHPUT;
                               its activation does NOT match gpt-oss, so the
                               grad-diff column will be enormous — that's the
                               adapter gap test_sonic_moe_fwd_bwd.py reports.

Run:
    python examples/gpt_oss/benchmark_speedups.py
    python examples/gpt_oss/benchmark_speedups.py --seq 4096 --moe
    python examples/gpt_oss/benchmark_speedups.py --dtype float32     # stricter correctness floor
    python examples/gpt_oss/benchmark_speedups.py --sliding-window 128
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F


# --------- helpers ----------------------------------------------------------


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _bench_fn(fn, warmup: int = 3, iters: int = 5) -> float:
    """Mean wall-clock over `iters` runs after `warmup` runs."""
    for _ in range(warmup):
        fn()
    _cuda_sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _cuda_sync()
    return (time.perf_counter() - t0) / iters


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _has_naninf(*ts: torch.Tensor | None) -> bool:
    for t in ts:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            return True
    return False


# --------- attention benchmarks ---------------------------------------------


def bench_attention(
    batch: int, heads: int, seq: int, head_dim: int,
    dtype: torch.dtype, sliding_window: int | None,
) -> None:
    # Defer the import so the script still loads even when flex_attention isn't
    # available; we fail with a useful message inside the variant.
    import os
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from flex_attention_sinks import (  # type: ignore[import-not-found]
        eager_attention_with_sinks,
        flex_attention_with_sinks,
    )

    device = "cuda"
    torch.manual_seed(0)

    print(
        f"\n=== attention (B={batch}, H={heads}, L={seq}, D={head_dim}, "
        f"dtype={dtype}, sliding_window={sliding_window}) ==="
    )

    q0 = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k0 = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    v0 = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    sinks0 = torch.randn(heads, device=device, dtype=dtype) * 0.5
    target0 = torch.randn_like(q0)

    def make_inputs():
        q = q0.detach().clone().requires_grad_(True)
        k = k0.detach().clone().requires_grad_(True)
        v = v0.detach().clone().requires_grad_(True)
        s = sinks0.detach().clone().requires_grad_(True)
        return q, k, v, s

    # --- 1. reference: eager-with-sinks ---
    q_r, k_r, v_r, s_r = make_inputs()
    _reset_peak()
    out_ref = eager_attention_with_sinks(q_r, k_r, v_r, s_r, sliding_window=sliding_window)
    (out_ref * target0).sum().backward()
    ref_peak = _peak_gb()
    ref_naninf = _has_naninf(out_ref, q_r.grad, k_r.grad, v_r.grad, s_r.grad)

    def eager_step():
        q, k, v, s = make_inputs()
        out = eager_attention_with_sinks(q, k, v, s, sliding_window=sliding_window)
        (out * target0).sum().backward()
    eager_t = _bench_fn(eager_step)

    # --- 2. flex with sinks ---
    flex_status = "OK"
    flex_peak = 0.0
    flex_t = 0.0
    flex_grad_diffs = {}
    flex_naninf = False
    try:
        q_f, k_f, v_f, s_f = make_inputs()
        _reset_peak()
        out_f = flex_attention_with_sinks(q_f, k_f, v_f, s_f, sliding_window=sliding_window)
        (out_f * target0).sum().backward()
        flex_peak = _peak_gb()
        flex_naninf = _has_naninf(out_f, q_f.grad, k_f.grad, v_f.grad, s_f.grad)
        flex_grad_diffs = {
            "q": _max_abs(q_r.grad, q_f.grad),
            "k": _max_abs(k_r.grad, k_f.grad),
            "v": _max_abs(v_r.grad, v_f.grad),
            "sinks": _max_abs(s_r.grad, s_f.grad),
            "out": _max_abs(out_ref, out_f),
        }

        def flex_step():
            q, k, v, s = make_inputs()
            out = flex_attention_with_sinks(q, k, v, s, sliding_window=sliding_window)
            (out * target0).sum().backward()
        flex_t = _bench_fn(flex_step)
    except Exception as exc:
        flex_status = f"FAIL: {type(exc).__name__}: {exc}"

    # --- 3. SDPA (no sinks) — bypass detector ---
    sdpa_status = "OK"
    sdpa_peak = 0.0
    sdpa_t = 0.0
    sdpa_out_diff = float("inf")
    try:
        q_s, k_s, v_s, _ = make_inputs()
        _reset_peak()
        # SDPA call mirrors what verl would do if the actor lands on SDPA — note
        # there's no place to plug sinks in, which is the bug class
        # test_attention_sinks.py catches.
        out_sdpa = F.scaled_dot_product_attention(
            q_s, k_s, v_s, is_causal=True, scale=1.0 / math.sqrt(head_dim),
        )
        (out_sdpa * target0).sum().backward()
        sdpa_peak = _peak_gb()
        sdpa_out_diff = _max_abs(out_ref, out_sdpa)

        def sdpa_step():
            q, k, v, _ = make_inputs()
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(head_dim))
            (out * target0).sum().backward()
        sdpa_t = _bench_fn(sdpa_step)
    except Exception as exc:
        sdpa_status = f"FAIL: {type(exc).__name__}: {exc}"

    # --- print table ---
    print(
        f"  {'variant':<22} {'fwd+bwd ms':>12} {'peak GB':>9} "
        f"{'grad q':>11} {'grad k':>11} {'grad v':>11} {'grad sinks':>12} {'naninf':>7} {'note':<30}"
    )
    print(
        f"  {'eager (ref)':<22} {eager_t * 1000:>12.2f} {ref_peak:>9.2f} "
        f"{'-':>11} {'-':>11} {'-':>11} {'-':>12} {str(ref_naninf):>7}  reference"
    )
    if flex_status == "OK":
        gq, gk, gv, gs = (
            flex_grad_diffs["q"], flex_grad_diffs["k"],
            flex_grad_diffs["v"], flex_grad_diffs["sinks"],
        )
        print(
            f"  {'flex (sinks)':<22} {flex_t * 1000:>12.2f} {flex_peak:>9.2f} "
            f"{gq:>11.3e} {gk:>11.3e} {gv:>11.3e} {gs:>12.3e} {str(flex_naninf):>7}  "
            f"out Δ={flex_grad_diffs['out']:.2e}"
        )
    else:
        print(f"  {'flex (sinks)':<22} {flex_status}")
    if sdpa_status == "OK":
        print(
            f"  {'sdpa (NO sinks)':<22} {sdpa_t * 1000:>12.2f} {sdpa_peak:>9.2f} "
            f"{'n/a':>11} {'n/a':>11} {'n/a':>11} {'n/a':>12} {'-':>7}  "
            f"out Δ={sdpa_out_diff:.2e} (gap = sinks impact)"
        )
    else:
        print(f"  {'sdpa (NO sinks)':<22} {sdpa_status}")

    # --- summary ---
    print()
    if flex_status == "OK":
        speedup = eager_t / flex_t if flex_t > 0 else 0.0
        mem_ratio = flex_peak / ref_peak if ref_peak > 0 else 0.0
        max_grad_diff = max(flex_grad_diffs["q"], flex_grad_diffs["k"],
                            flex_grad_diffs["v"], flex_grad_diffs["sinks"])
        print(f"  flex vs eager: {speedup:.2f}x faster, {mem_ratio:.2f}x peak memory, "
              f"max grad diff {max_grad_diff:.2e}")
    if sdpa_status == "OK":
        print(f"  sdpa output |Δ| from eager-with-sinks: {sdpa_out_diff:.3e}")
        if sdpa_out_diff < 1e-6:
            print("    >> sdpa output identical to eager-with-sinks — sinks were silently dropped.")
        else:
            print(f"    >> sdpa output differs from eager — but probabaly NOT in the sinks-respecting way.")


# --------- MoE benchmarks ---------------------------------------------------


def _ref_moe_forward(
    x, gate_up_proj, down_proj, router_weight, top_k, activation,
):
    """Per-expert pytorch reference; matches transformers' GptOssExperts when
    activation='gpt-oss'."""
    if activation == "vanilla":
        def act(g, u):
            return F.silu(g) * u
    elif activation == "gpt-oss":
        def act(g, u):
            g = g.clamp(max=7.0)
            u = u.clamp(min=-7.0, max=7.0)
            return g * torch.sigmoid(g * 1.702) * (u + 1.0)
    else:
        raise ValueError(activation)

    E = router_weight.shape[0]
    logits = x @ router_weight.t()
    probs = F.softmax(logits, dim=-1)
    topk_scores, topk_idx = probs.topk(top_k, dim=-1)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(x)
    for e in range(E):
        mask = (topk_idx == e)
        if not mask.any():
            continue
        rows, slots = mask.nonzero(as_tuple=True)
        weights = topk_scores[rows, slots].unsqueeze(-1)
        x_e = x[rows]
        gate_up = x_e @ gate_up_proj[e]
        gate, up = gate_up.chunk(2, dim=-1)
        glu = act(gate, up)
        y_e = glu @ down_proj[e]
        out.index_add_(0, rows, y_e * weights)
    return out


def bench_moe(
    num_experts: int, top_k: int, hidden: int, intermediate: int,
    tokens: int, dtype: torch.dtype,
) -> None:
    device = "cuda"
    torch.manual_seed(0)
    print(
        f"\n=== MoE (E={num_experts}, K={top_k}, hidden={hidden}, "
        f"intermediate={intermediate}, tokens={tokens}, dtype={dtype}) ==="
    )

    x0 = torch.randn(tokens, hidden, device=device, dtype=dtype)
    gate_up0 = torch.randn(num_experts, hidden, 2 * intermediate, device=device, dtype=dtype) * 0.02
    down0 = torch.randn(num_experts, intermediate, hidden, device=device, dtype=dtype) * 0.02
    rw0 = torch.randn(num_experts, hidden, device=device, dtype=dtype) * 0.02
    target0 = torch.randn_like(x0)

    def make_inputs():
        x = x0.detach().clone().requires_grad_(True)
        gu = gate_up0.detach().clone().requires_grad_(True)
        dn = down0.detach().clone().requires_grad_(True)
        rw = rw0.detach().clone().requires_grad_(True)
        return x, gu, dn, rw

    # --- pytorch gpt-oss reference (clamped + (up+1)) ---
    x_r, gu_r, dn_r, rw_r = make_inputs()
    _reset_peak()
    out_r = _ref_moe_forward(x_r, gu_r, dn_r, rw_r, top_k=top_k, activation="gpt-oss")
    (out_r * target0).sum().backward()
    ref_peak = _peak_gb()
    ref_naninf = _has_naninf(out_r, x_r.grad, gu_r.grad, dn_r.grad, rw_r.grad)

    def ref_step():
        x, gu, dn, rw = make_inputs()
        out = _ref_moe_forward(x, gu, dn, rw, top_k=top_k, activation="gpt-oss")
        (out * target0).sum().backward()
    ref_t = _bench_fn(ref_step, warmup=1, iters=2)  # naive loop, slow on purpose

    # --- sonic-moe (vanilla SwiGLU) — only if importable ---
    sonic_status = "skipped"
    try:
        from sonicmoe import KernelBackendMoE, MoE
        from sonicmoe.enums import ActivationType

        sonic = MoE(
            num_experts=num_experts, num_experts_per_tok=top_k,
            hidden_size=hidden, intermediate_size=intermediate,
            activation_function=ActivationType.SWIGLU,
            add_bias=False, std=0.02,
        ).to(device=device, dtype=dtype)

        x_s = x0.detach().clone().requires_grad_(True)
        _reset_peak()
        out_s, _aux = sonic(x_s, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        (out_s * target0).sum().backward()
        sonic_peak = _peak_gb()
        sonic_naninf = _has_naninf(out_s, x_s.grad)

        # Output gap is HUGE because sonic-moe uses vanilla SwiGLU, gpt-oss
        # uses clamped + (up+1). This is the gap the adapter has to close.
        sonic_out_diff = _max_abs(out_r, out_s)

        def sonic_step():
            xs = x0.detach().clone().requires_grad_(True)
            o, _ = sonic(xs, kernel_backend_moe=KernelBackendMoE.sonicmoe)
            (o * target0).sum().backward()
        sonic_t = _bench_fn(sonic_step)
        sonic_status = "OK"
    except ImportError:
        sonic_status = "skipped (sonicmoe not importable)"
    except Exception as exc:
        sonic_status = f"FAIL: {type(exc).__name__}: {exc}"

    # --- print ---
    print(
        f"  {'variant':<28} {'fwd+bwd ms':>12} {'peak GB':>9} "
        f"{'out Δ vs gpt-oss ref':>22} {'naninf':>7}"
    )
    print(
        f"  {'pytorch ref (gpt-oss GLU)':<28} {ref_t * 1000:>12.2f} {ref_peak:>9.2f} "
        f"{'-':>22} {str(ref_naninf):>7}  (slow per-expert loop)"
    )
    if sonic_status == "OK":
        print(
            f"  {'sonic-moe (vanilla SwiGLU)':<28} {sonic_t * 1000:>12.2f} {sonic_peak:>9.2f} "
            f"{sonic_out_diff:>22.3e} {str(sonic_naninf):>7}  (gap = activation mismatch)"
        )
        speedup = ref_t / sonic_t if sonic_t > 0 else 0.0
        print(f"  >> sonic-moe vs ref: {speedup:.1f}x throughput (loose upper bound; ref is naive loop)")
    else:
        print(f"  {'sonic-moe':<28} {sonic_status}")


# --------- main -------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # attention defaults: gpt-oss-20b shape per layer, modest seq for blind viability
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--sliding-window", type=int, default=None)
    # MoE
    p.add_argument("--moe", action="store_true", help="also benchmark MoE impls (slower)")
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--hidden", type=int, default=2880)
    p.add_argument("--intermediate", type=int, default=2880)
    p.add_argument("--tokens", type=int, default=2048)
    # global
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--skip-attention", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[bench] CUDA required", file=sys.stderr)
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"[bench] gpu sm_{cap[0]}{cap[1]}, torch {torch.__version__}, dtype={args.dtype}")

    dtype = getattr(torch, args.dtype)

    if not args.skip_attention:
        bench_attention(
            batch=args.batch, heads=args.heads, seq=args.seq,
            head_dim=args.head_dim, dtype=dtype,
            sliding_window=args.sliding_window,
        )
    if args.moe:
        bench_moe(
            num_experts=args.num_experts, top_k=args.top_k,
            hidden=args.hidden, intermediate=args.intermediate,
            tokens=args.tokens, dtype=dtype,
        )

    print("\n[bench] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
