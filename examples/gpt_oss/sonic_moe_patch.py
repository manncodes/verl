"""Adapter scaffolding for swapping HF's GptOssMoE with Dao-AILab/sonic-moe.

STATUS: experimental scaffolding, NOT a turn-key integration.

Why this isn't a one-liner
--------------------------
sonic-moe's high-level :class:`sonicmoe.MoE` bakes the activation function
(SwiGLU, GELU, etc.) into a fused kernel. gpt-oss's MoE block uses a
*clamped, GELU-approximating SwiGLU with a (up+1) shift* that no entry in
:class:`sonicmoe.enums.ActivationType` matches:

    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    up   = up.clamp(min=-7.0, max=7.0)
    glu  = gate * torch.sigmoid(gate * 1.702) * (up + 1.0)
    out  = glu @ down_proj

Naively dropping in ``KernelBackendMoE.sonicmoe`` therefore produces a
numerically different model — different enough to break training even
before policy drift kicks in.

Plausible integration path (NOT YET IMPLEMENTED)
------------------------------------------------
sonic-moe's private functional surface
(``sonicmoe.functional.forward._up_projection_forward`` /
``_down_projection_forward``) exposes the two grouped GEMMs separately, so a
faithful adapter would:

  1. Use sonic-moe's routing path (``_topk_softmax_fwd``) to pick experts.
  2. Call ``_up_projection_forward`` with ``activation_type`` set to a
     no-op variant (need to confirm one exists; otherwise contribute one
     upstream).
  3. Apply gpt-oss's clamped + ``(up + 1)`` activation in PyTorch.
  4. Call ``_down_projection_forward``.
  5. Map HF state-dict tensors (``gate_up_proj``, ``down_proj``,
     ``router.weight``, ``router.bias``) into sonic-moe's expected layout
     (interleaved vs concatenated — sonic-moe supports both, choose based
     on tile alignment). gpt-oss-20b's intermediate_size is 2880 which is
     NOT a power of two, so the chosen layout has to handle that or pad.
  6. Wrap the new module so FSDP sees the same parameter graph as the
     original ``GptOssExperts`` (so the actor's state_dict / checkpoint
     paths still work).

Until the activation parity is verified end-to-end (forward + backward) by
``examples/gpt_oss/test_sonic_moe.py`` on real Hopper hardware, calling
:func:`apply_sonic_moe_to_model` raises :class:`NotImplementedError`.

Rough plan of attack for whoever picks this up
----------------------------------------------
- Run ``python examples/gpt_oss/test_sonic_moe.py`` first; that script
  benchmarks sonic-moe's vanilla SwiGLU at gpt-oss-20b shapes and quantifies
  the numerical gap from gpt-oss's true activation. If the wall-clock win
  isn't worth the integration cost, stop here.
- If it is, prototype steps 1-4 above in a standalone unit test against HF
  ``GptOssExperts.forward`` at ``atol=1e-2`` (bf16 tolerance).
- Wire ``apply_sonic_moe_to_model`` to walk
  ``model.model.layers[i].mlp.experts`` and replace each with the new
  module BEFORE FSDP wrap (post-wrap is much harder).
"""

from __future__ import annotations

import torch
import torch.nn as nn

GPT_OSS_ALPHA = 1.702
GPT_OSS_LIMIT = 7.0


def gpt_oss_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The exact activation gpt-oss uses inside each expert.

    Source: ``transformers/models/gpt_oss/modeling_gpt_oss.py::GptOssExperts.forward``.
    Kept here as a single source of truth so the adapter and the probe agree.
    """
    gate = gate.clamp(max=GPT_OSS_LIMIT)
    up = up.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
    return gate * torch.sigmoid(gate * GPT_OSS_ALPHA) * (up + 1.0)


def is_sonic_moe_available() -> tuple[bool, str]:
    """Return (available, reason). Cheap, safe to call from a launcher preflight."""
    try:
        import sonicmoe  # noqa: F401
    except ImportError as exc:
        return False, f"sonicmoe not importable: {exc}"
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    cap = torch.cuda.get_device_capability(0)
    # sonic-moe targets Hopper (sm_90) and Blackwell (sm_100/sm_103).
    if cap[0] < 9:
        return False, f"GPU compute capability {cap} below Hopper (sm_90 required)"
    return True, f"sonicmoe ok on sm_{cap[0]}{cap[1]}"


def apply_sonic_moe_to_model(model: nn.Module) -> nn.Module:
    """Walk a HF gpt-oss model and replace each expert block with the sonic-moe
    adapter. Currently raises until the activation-parity work is done.
    """
    available, reason = is_sonic_moe_available()
    if not available:
        raise RuntimeError(f"cannot apply sonic-moe adapter: {reason}")
    raise NotImplementedError(
        "sonic-moe integration is scaffolded but not wired up.\n"
        "gpt-oss's clamped + (up+1) activation has no parity in "
        "sonic-moe.MoE; the adapter must compose sonicmoe.functional._{up,down}_projection_forward "
        "with gpt_oss_glu() in this module. See the module docstring for the integration plan, "
        "and run examples/gpt_oss/test_sonic_moe.py first to confirm the win is worth the work."
    )


__all__ = [
    "GPT_OSS_ALPHA",
    "GPT_OSS_LIMIT",
    "apply_sonic_moe_to_model",
    "gpt_oss_glu",
    "is_sonic_moe_available",
]
