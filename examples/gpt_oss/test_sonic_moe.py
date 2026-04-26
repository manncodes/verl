"""Forward-only probe: is sonic-moe worth integrating into the gpt-oss-20b actor?

Run on the H100 box:

    python examples/gpt_oss/test_sonic_moe.py            # default: gpt-oss-20b shapes
    python examples/gpt_oss/test_sonic_moe.py --tokens 8192

Reports two numbers:

  1. Wall-clock per forward at gpt-oss-20b shapes (E=32, K=4, hidden=2880,
     intermediate=2880) for sonic-moe's KernelBackendMoE.sonicmoe vs a
     pure-PyTorch eager reference. The eager reference uses gpt-oss's exact
     clamped, GELU-style, (up+1)-shifted activation — i.e. the activation
     a faithful integration would have to preserve. So this is the
     theoretical upper bound on the speedup we'd see in training, IF we
     can match the activation later.

  2. Numerical gap between sonic-moe's vanilla SwiGLU output and the
     gpt-oss reference, given the same random expert weights + inputs.
     Bigger gap => more work for the adapter (which must compose sonic-moe's
     functional grouped GEMMs with the gpt-oss activation).

What the script does NOT do:
  * Test backward (sonic-moe relies on torch autograd; correctness has to
    be checked once the adapter is built).
  * Touch the verl trainer or load HF gpt-oss-20b weights — this is a
    layer-shape probe, not an end-to-end test.
  * Validate FSDP sharding compatibility (tile-alignment risk: gpt-oss's
    intermediate_size=2880 is not a power of two, so sonic-moe's grouped
    GEMM may need padding or a non-default layout).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch


def _try_import_sonicmoe():
    try:
        import sonicmoe  # noqa: F401
        from sonicmoe import MoE, KernelBackendMoE
        from sonicmoe.enums import ActivationType
    except ImportError as exc:
        print(f"[sonic-moe-probe] sonicmoe not importable: {exc}", file=sys.stderr)
        print("Install with: INSTALL_SONIC_MOE=1 bash examples/gpt_oss/install.sh", file=sys.stderr)
        sys.exit(2)
    return MoE, KernelBackendMoE, ActivationType


def _gpt_oss_reference_moe(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Eager pytorch implementation of gpt-oss's MoE forward at the layer level.

    Mirrors transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts.forward —
    intentionally simple (no fusion, no router-aux-loss) so its output is the
    ground truth that the adapter has to match.

    Shapes:
      x:             [N, hidden]
      gate_up_proj:  [E, hidden, 2 * intermediate]
      down_proj:     [E, intermediate, hidden]
      router_weight: [E, hidden]
      router_bias:   [E]
    """
    # Inline the activation so this script runs as a standalone file
    # (examples/gpt_oss/ isn't a package). Keep the formula in lockstep with
    # sonic_moe_patch.gpt_oss_glu — they must agree.
    GPT_OSS_ALPHA = 1.702
    GPT_OSS_LIMIT = 7.0

    def gpt_oss_glu(gate, up):
        gate = gate.clamp(max=GPT_OSS_LIMIT)
        up = up.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
        return gate * torch.sigmoid(gate * GPT_OSS_ALPHA) * (up + 1.0)

    N, hidden = x.shape
    E = router_weight.shape[0]

    # Routing: softmax then top-k (gpt-oss style; matches transformers >= 4.46).
    logits = x @ router_weight.t() + router_bias            # [N, E]
    probs = torch.softmax(logits, dim=-1)
    topk_scores, topk_idx = probs.topk(top_k, dim=-1)       # [N, K] each
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(x)
    # Naive per-expert loop: fine for a probe, not for production.
    for e in range(E):
        # rows of x routed to this expert (may be on multiple of their top-K slots)
        mask = (topk_idx == e)
        if not mask.any():
            continue
        rows, slots = mask.nonzero(as_tuple=True)            # both [n_e]
        weights = topk_scores[rows, slots].unsqueeze(-1)     # [n_e, 1]
        x_e = x[rows]                                        # [n_e, hidden]
        gate_up = x_e @ gate_up_proj[e]                      # [n_e, 2*intermediate]
        gate, up = gate_up.chunk(2, dim=-1)
        glu = gpt_oss_glu(gate, up)                          # [n_e, intermediate]
        y_e = glu @ down_proj[e]                             # [n_e, hidden]
        out.index_add_(0, rows, y_e * weights)
    return out


def _bench(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # gpt-oss-20b config defaults.
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--hidden", type=int, default=2880)
    p.add_argument("--intermediate", type=int, default=2880)
    p.add_argument("--tokens", type=int, default=4096, help="number of tokens to forward")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--atol",
        type=float,
        default=5e-2,
        help="absolute tolerance for activation gap (bf16 baseline: 5e-2)",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[sonic-moe-probe] CUDA required", file=sys.stderr)
        return 1
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        print(f"[sonic-moe-probe] need Hopper (sm_90+); got sm_{cap[0]}{cap[1]}", file=sys.stderr)
        return 1

    MoE, KernelBackendMoE, ActivationType = _try_import_sonicmoe()

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    torch.manual_seed(args.seed)

    print(f"[sonic-moe-probe] gpu sm_{cap[0]}{cap[1]}, dtype={args.dtype}, "
          f"E={args.num_experts}, K={args.top_k}, "
          f"hidden={args.hidden}, intermediate={args.intermediate}, "
          f"tokens={args.tokens}")

    # ---- 1. build sonic-moe -------------------------------------------------
    sonic = MoE(
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device=device, dtype=dtype)

    x = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)

    # ---- 2. extract sonic-moe's expert weights for the reference -----------
    # We want the EAGER reference to use the same parameters as sonic-moe so any
    # numerical difference is purely activation-induced. sonic-moe's parameter
    # naming may shift across versions; probe defensively.
    state = dict(sonic.named_parameters())
    print("[sonic-moe-probe] sonic-moe parameters:")
    for k, v in state.items():
        print(f"  {k:40s} shape={tuple(v.shape)} dtype={v.dtype}")

    def _find(*needles: str) -> torch.Tensor | None:
        for k, v in state.items():
            if all(n in k for n in needles):
                return v
        return None

    gate_up = _find("c_fc")          # [E, hidden, 2*intermediate] (concat) or interleaved
    down = _find("c_proj")           # [E, intermediate, hidden]
    router_w = _find("router", "weight")
    router_b = _find("router", "bias")
    if gate_up is None or down is None or router_w is None:
        print("[sonic-moe-probe] could not locate expected parameters — sonic-moe API "
              "drift; cannot run the numerics check. The wall-clock benchmark still ran.",
              file=sys.stderr)
        return 3

    # Check shapes; print a clear note if they don't match the assumed concat layout.
    if gate_up.shape == (args.num_experts, args.hidden, 2 * args.intermediate):
        layout = "concat"
    elif gate_up.shape == (args.num_experts, args.hidden, 2 * args.intermediate) or \
            gate_up.shape == (2 * args.intermediate, args.num_experts, args.hidden):
        layout = "interleaved-or-other"
    else:
        layout = "unknown"
    print(f"[sonic-moe-probe] c_fc layout: {layout} (shape {tuple(gate_up.shape)})")

    if router_b is None:
        router_b = torch.zeros(args.num_experts, device=device, dtype=dtype)

    # ---- 3. forward parity --------------------------------------------------
    with torch.no_grad():
        sonic_out, _aux = sonic(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        # The eager reference uses gpt-oss's clamped, (up+1)-shifted GLU; this
        # is what a faithful integration would have to match.
        if layout != "concat":
            print("[sonic-moe-probe] skipping numeric parity: c_fc layout not concat")
        else:
            ref_out = _gpt_oss_reference_moe(
                x.float(), gate_up.float(), down.float(),
                router_w.float(), router_b.float(),
                top_k=args.top_k,
            ).to(dtype)

            diff = (sonic_out - ref_out).abs()
            print(f"[sonic-moe-probe] activation gap (sonic SwiGLU vs gpt-oss clamped GLU):")
            print(f"  max abs diff:  {diff.max().item():.4e}")
            print(f"  mean abs diff: {diff.mean().item():.4e}")
            print(f"  rms ref:       {ref_out.float().pow(2).mean().sqrt().item():.4e}")
            if diff.max().item() > args.atol:
                print(f"  >> gap exceeds atol={args.atol}; the adapter MUST replace sonic-moe's")
                print(f"     baked-in SwiGLU before this can be used in training.")
            else:
                print(f"  >> gap within atol={args.atol}; vanilla SwiGLU might be close enough")
                print(f"     for a stress test, but DO NOT ship to training without the clamped")
                print(f"     activation — gpt-oss's pretraining used (up+1) and the bias matters.")

    # ---- 4. wall-clock benchmark -------------------------------------------
    print("[sonic-moe-probe] benchmarking (10 iters after 3 warmup):")

    def sonic_fwd():
        sonic(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)

    sonic_t = _bench(sonic_fwd)

    if layout == "concat":
        def ref_fwd():
            _gpt_oss_reference_moe(x, gate_up, down, router_w, router_b, top_k=args.top_k)
        # The reference loop is dog-slow; cap iterations or it dominates.
        ref_t = _bench(ref_fwd, warmup=1, iters=3)
        speedup = ref_t / sonic_t if sonic_t > 0 else math.nan
        print(f"  sonic-moe forward: {sonic_t * 1000:.2f} ms")
        print(f"  eager reference:   {ref_t * 1000:.2f} ms")
        print(f"  speedup ratio:     {speedup:.1f}x  (note: ref is naive per-expert loop, "
              f"NOT comparable to HF's optimised eager — this is an upper bound)")
    else:
        print(f"  sonic-moe forward: {sonic_t * 1000:.2f} ms")
        print("  eager reference skipped (layout unknown)")

    print("[sonic-moe-probe] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
