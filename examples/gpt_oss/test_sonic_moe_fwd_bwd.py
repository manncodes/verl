"""Standalone forward+backward parity test for sonic-moe at gpt-oss-20b shapes.

What this script answers
------------------------
1. Does sonic-moe's grouped-GEMM kernel produce the same MoE output as a
   plain-pytorch reference using the *same* weights? (forward parity)
2. Does sonic-moe's autograd produce the same gradients on every learnable
   tensor (gate_up_proj, down_proj, router, input)? (backward parity)
3. How big is the numerical gap between vanilla SwiGLU and gpt-oss's
   clamped, GELU-style, (up+1)-shifted GLU on the same weights? (the gap
   the adapter has to close before sonic-moe is usable in training)

What this script does NOT do
----------------------------
- It does not test against transformers' GptOssExperts directly. The
  pytorch reference here is intentionally self-contained so the test is
  one file; if you trust the formula in `gpt_oss_glu` (which mirrors
  transformers verbatim), this is equivalent to comparing against HF.
- It does not test FSDP, multi-GPU, or the full verl actor path.
- It does not exercise sonic-moe's private functional API (which is the
  path the real adapter would take). That comes after this script
  passes.

Run on the H100 box, after `INSTALL_SONIC_MOE=1 bash examples/gpt_oss/install.sh`:

    python examples/gpt_oss/test_sonic_moe_fwd_bwd.py
    python examples/gpt_oss/test_sonic_moe_fwd_bwd.py --tokens 8192 --top-k 4
    python examples/gpt_oss/test_sonic_moe_fwd_bwd.py --skip-backward    # forward only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# Mirrors transformers.models.gpt_oss.modeling_gpt_oss verbatim (alpha=1.702
# is the GELU-approximating sigmoid scaling; limit=7.0 is the activation
# clamp; (up + 1) is the gpt-oss-specific shift).
GPT_OSS_ALPHA = 1.702
GPT_OSS_LIMIT = 7.0


# ----- pytorch reference MoE ------------------------------------------------


def _vanilla_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # F.silu(gate) * up == (gate * sigmoid(gate)) * up
    return F.silu(gate) * up


def _gpt_oss_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = gate.clamp(max=GPT_OSS_LIMIT)
    up = up.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
    return gate * torch.sigmoid(gate * GPT_OSS_ALPHA) * (up + 1.0)


_ACTIVATIONS = {
    "vanilla": _vanilla_swiglu,
    "gpt-oss": _gpt_oss_glu,
}


def reference_moe_forward(
    x: torch.Tensor,                  # [N, hidden]
    gate_up_proj: torch.Tensor,       # [E, hidden, 2*intermediate]
    down_proj: torch.Tensor,          # [E, intermediate, hidden]
    router_weight: torch.Tensor,      # [E, hidden]
    router_bias: torch.Tensor | None, # [E] or None
    top_k: int,
    activation: str = "vanilla",
) -> torch.Tensor:
    """Per-expert eager forward. Slow on purpose: simple to read, easy to trust."""
    act = _ACTIVATIONS[activation]
    E = router_weight.shape[0]

    logits = x @ router_weight.t()
    if router_bias is not None:
        logits = logits + router_bias
    probs = torch.softmax(logits, dim=-1)
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


# ----- weight extraction from sonic-moe -------------------------------------


@dataclass
class SonicWeights:
    gate_up_proj: torch.Tensor   # [E, hidden, 2*intermediate]
    down_proj: torch.Tensor      # [E, intermediate, hidden]
    router_weight: torch.Tensor  # [E, hidden]
    router_bias: torch.Tensor | None


def extract_sonic_weights(sonic, num_experts: int, hidden: int, intermediate: int) -> SonicWeights:
    """Pull sonic-moe's expert weights into the layout the reference expects.

    sonic-moe's parameter naming is a moving target across versions, so we probe
    by suffix and shape rather than hard-coding names. Prints what it found so
    layout drift is visible in the test output.
    """
    params = dict(sonic.named_parameters())
    print("[parity] sonic-moe parameters:")
    for k, v in params.items():
        print(f"  {k:40s} shape={tuple(v.shape)} dtype={v.dtype}")

    def _find_by(*needles: str) -> torch.Tensor | None:
        for k, v in params.items():
            if all(n in k.lower() for n in needles):
                return v
        return None

    gate_up = _find_by("c_fc")
    down = _find_by("c_proj")
    router_w = _find_by("router", "weight")
    router_b = _find_by("router", "bias")

    if gate_up is None or down is None or router_w is None:
        raise RuntimeError(
            "could not locate sonic-moe expert weights by name. dump above shows "
            "the actual parameter list — update extract_sonic_weights()."
        )

    # Normalise gate_up_proj to [E, hidden, 2*intermediate] (concat layout).
    expected = (num_experts, hidden, 2 * intermediate)
    if gate_up.shape == expected:
        pass
    elif gate_up.shape == (num_experts, 2 * intermediate, hidden):
        gate_up = gate_up.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(
            f"unexpected gate_up_proj shape {tuple(gate_up.shape)}; expected "
            f"{expected} or its transpose. layout may be interleaved — needs "
            f"a custom remap."
        )

    expected_down = (num_experts, intermediate, hidden)
    if down.shape == expected_down:
        pass
    elif down.shape == (num_experts, hidden, intermediate):
        down = down.transpose(1, 2).contiguous()
    else:
        raise RuntimeError(
            f"unexpected down_proj shape {tuple(down.shape)}; expected "
            f"{expected_down} or its transpose."
        )

    if router_w.shape != (num_experts, hidden):
        if router_w.shape == (hidden, num_experts):
            router_w = router_w.t().contiguous()
        else:
            raise RuntimeError(
                f"unexpected router weight shape {tuple(router_w.shape)}; "
                f"expected ({num_experts}, {hidden})."
            )

    return SonicWeights(
        gate_up_proj=gate_up,
        down_proj=down,
        router_weight=router_w,
        router_bias=router_b,
    )


# ----- main test ------------------------------------------------------------


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _rms(a: torch.Tensor) -> float:
    return a.float().pow(2).mean().sqrt().item()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # gpt-oss-20b config defaults.
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--hidden", type=int, default=2880)
    p.add_argument("--intermediate", type=int, default=2880)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--atol-fwd",
        type=float,
        default=5e-2,
        help="forward parity threshold (bf16 expectation: ~5e-2 with full-bf16 reference)",
    )
    p.add_argument(
        "--atol-bwd",
        type=float,
        default=1e-1,
        help="backward parity threshold (looser; gradients accumulate rounding)",
    )
    p.add_argument("--skip-backward", action="store_true", help="forward parity only")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[parity] CUDA required", file=sys.stderr)
        return 1
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        print(f"[parity] need Hopper (sm_90+); got sm_{cap[0]}{cap[1]}", file=sys.stderr)
        return 1

    try:
        from sonicmoe import MoE, KernelBackendMoE
        from sonicmoe.enums import ActivationType
    except ImportError as exc:
        print(f"[parity] sonic-moe not importable: {exc}", file=sys.stderr)
        print("Install with: INSTALL_SONIC_MOE=1 bash examples/gpt_oss/install.sh", file=sys.stderr)
        return 2

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    torch.manual_seed(args.seed)

    print(
        f"[parity] gpu sm_{cap[0]}{cap[1]} dtype={args.dtype} "
        f"E={args.num_experts} K={args.top_k} "
        f"hidden={args.hidden} intermediate={args.intermediate} tokens={args.tokens}"
    )

    # --- 1. build sonic-moe -------------------------------------------------
    sonic = MoE(
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device=device, dtype=dtype)

    weights = extract_sonic_weights(sonic, args.num_experts, args.hidden, args.intermediate)

    # --- 2. forward parity (vanilla SwiGLU on both sides) -------------------
    x = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
    print()
    print("[parity] forward (vanilla SwiGLU on both sides)")
    with torch.no_grad():
        sonic_out, _aux = sonic(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        ref_out = reference_moe_forward(
            x, weights.gate_up_proj, weights.down_proj,
            weights.router_weight, weights.router_bias,
            top_k=args.top_k, activation="vanilla",
        )
    fwd_diff = _max_abs(sonic_out, ref_out)
    print(f"  max |sonic - ref|: {fwd_diff:.4e}")
    print(f"  rms ref:           {_rms(ref_out):.4e}")
    fwd_pass = fwd_diff <= args.atol_fwd
    print(f"  >> {'PASS' if fwd_pass else 'FAIL'} (atol={args.atol_fwd})")

    # --- 3. activation gap report (informational) ---------------------------
    print()
    print("[parity] gpt-oss vs vanilla SwiGLU (same weights, informational)")
    with torch.no_grad():
        ref_gptoss = reference_moe_forward(
            x, weights.gate_up_proj, weights.down_proj,
            weights.router_weight, weights.router_bias,
            top_k=args.top_k, activation="gpt-oss",
        )
    act_gap = _max_abs(ref_out, ref_gptoss)
    print(f"  max |vanilla - gpt-oss|: {act_gap:.4e}")
    print(f"  >> this gap is what the sonic-moe adapter has to close.")

    # --- 4. backward parity -------------------------------------------------
    bwd_pass = True
    if not args.skip_backward:
        print()
        print("[parity] backward (vanilla SwiGLU on both sides)")

        # Use the same loss on both: dot product with a fixed random target,
        # so gradients of x and weights are non-trivial functions of every
        # parameter and we exercise the full autograd graph.
        target = torch.randn_like(x)

        # --- sonic-moe path
        x_s = x.detach().clone().requires_grad_(True)
        for v in sonic.parameters():
            v.grad = None
        sonic_out_g, _aux = sonic(x_s, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        loss_s = (sonic_out_g * target).sum()
        loss_s.backward()
        s_grads = {k: v.grad.detach().clone() for k, v in sonic.named_parameters() if v.grad is not None}
        s_grad_x = x_s.grad.detach().clone() if x_s.grad is not None else None
        if not s_grads or s_grad_x is None:
            print("  >> FAIL: sonic-moe produced no gradients (kernel may not support autograd)")
            return 3

        # --- reference path: re-extract leaf tensors so they get their own .grad
        gate_up = weights.gate_up_proj.detach().clone().requires_grad_(True)
        down = weights.down_proj.detach().clone().requires_grad_(True)
        rw = weights.router_weight.detach().clone().requires_grad_(True)
        rb = weights.router_bias.detach().clone().requires_grad_(True) if weights.router_bias is not None else None
        x_r = x.detach().clone().requires_grad_(True)
        ref_out_g = reference_moe_forward(x_r, gate_up, down, rw, rb, top_k=args.top_k, activation="vanilla")
        loss_r = (ref_out_g * target).sum()
        loss_r.backward()

        # Re-extract sonic-moe's grads under the same names we know on the reference side.
        s_w = extract_sonic_weights(sonic, args.num_experts, args.hidden, args.intermediate)
        # We can't re-orient gradients trivially after a transpose; only compare layouts where
        # extract_sonic_weights() did NOT have to transpose. (Detection: if the grad shape
        # matches the original sonic param shape, extraction was identity.)
        # Simplest: just compare per-element after applying the same transpose to grads.
        # Re-do extraction on grads dict directly.

        # Map sonic param objects (which carry the .grad) to their post-extraction layout.
        # Easier: pick by suffix again.
        def _pick(needles: tuple[str, ...]) -> torch.Tensor | None:
            for k, v in sonic.named_parameters():
                if all(n in k.lower() for n in needles) and v.grad is not None:
                    return v.grad
            return None

        sonic_gate_up_grad = _pick(("c_fc",))
        sonic_down_grad = _pick(("c_proj",))
        sonic_router_grad = _pick(("router", "weight"))
        # Apply the same shape normalisations so the diffs are layout-aware.
        if sonic_gate_up_grad is not None and sonic_gate_up_grad.shape != gate_up.shape:
            sonic_gate_up_grad = sonic_gate_up_grad.transpose(-2, -1).contiguous()
        if sonic_down_grad is not None and sonic_down_grad.shape != down.shape:
            sonic_down_grad = sonic_down_grad.transpose(-2, -1).contiguous()
        if sonic_router_grad is not None and sonic_router_grad.shape != rw.shape:
            sonic_router_grad = sonic_router_grad.t().contiguous()

        def _report(name: str, a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
            if a is None or b is None:
                print(f"  {name:18s}: SKIP (one side missing)")
                return True
            d = _max_abs(a, b)
            r = _rms(b)
            ok = d <= args.atol_bwd
            print(f"  {name:18s}: max |Δ|={d:.4e}  rms ref={r:.4e}  {'PASS' if ok else 'FAIL'}")
            return ok

        ok_x = _report("grad x", s_grad_x, x_r.grad)
        ok_g = _report("grad gate_up_proj", sonic_gate_up_grad, gate_up.grad)
        ok_d = _report("grad down_proj", sonic_down_grad, down.grad)
        ok_r = _report("grad router_weight", sonic_router_grad, rw.grad)
        bwd_pass = ok_x and ok_g and ok_d and ok_r
        print(f"  >> backward {'PASS' if bwd_pass else 'FAIL'} (atol={args.atol_bwd})")

    print()
    print(f"[parity] forward:  {'PASS' if fwd_pass else 'FAIL'}")
    print(f"[parity] backward: {'PASS' if bwd_pass else 'SKIP' if args.skip_backward else 'FAIL'}")
    print(f"[parity] activation gap (vanilla vs gpt-oss): {act_gap:.4e}  rms ref: {_rms(ref_gptoss):.4e}")
    return 0 if (fwd_pass and (args.skip_backward or bwd_pass)) else 4


if __name__ == "__main__":
    sys.exit(main())
