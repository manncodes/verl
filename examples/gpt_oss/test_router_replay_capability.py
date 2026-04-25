"""R3 (router replay) capability check for the gpt-oss + sglang/vllm stack.

Background
----------
verl's router-replay (R2/R3) uses the rollout engine to *record* per-token
expert routing decisions, then *replays* them in the training-side actor so
the two stacks make identical routing choices on the same tokens. Without
this, MoE training drifts from the rollout policy quickly (issue #3894).

Today the replay is wired only for the Megatron actor
(`verl/workers/engine_workers.py:477` gates on `actor.strategy=="megatron"`),
but the *recording* side lives in the rollout engines and works with any
actor. So this test verifies whether the recording path is even available
in the current install — useful both for the FSDP recipe (where it's a
discovery test for a future megatron switch) and for catching a missing
sglang/vllm patch before training starts.

Invariants checked
------------------
 1. verl exposes `RouterReplayConfig` and the rollout config carries
    `enable_rollout_routing_replay`. (Schema sanity.)
 2. The model's hf_config has `num_hidden_layers`, `num_experts_per_tok`,
    `num_local_experts` — required by sglang's recorder.
 3. sglang exposes `extract_routed_experts_from_meta_info` — verl will
    fail loudly without it.
 4. The HF gpt-oss MoE router is deterministic: two forward passes on the
    same input return identical top-k expert ids per layer. Determinism is
    a precondition for replay to even be meaningful.

Usage:
    python examples/gpt_oss/test_router_replay_capability.py \
        --model-dir /model/Huggingface/openai/gpt-oss-20b-bf16
"""

import argparse
import os
import sys
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def log(msg: str) -> None:
    print(f"[r3-cap] {msg}", flush=True)


def assert_(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_verl_schema() -> None:
    log("checking verl rollout/actor schema for routing replay knobs")
    try:
        from verl.workers.config import RouterReplayConfig  # noqa: F401
    except Exception as exc:
        raise AssertionError(f"verl is missing RouterReplayConfig: {exc}") from exc

    # The rollout config (a hydra YAML) carries enable_rollout_routing_replay;
    # we check the YAML directly to avoid pulling the whole trainer in.
    here = os.path.dirname(os.path.abspath(__file__))
    rollout_yaml = os.path.join(here, "..", "..", "verl", "trainer", "config", "rollout", "rollout.yaml")
    if os.path.isfile(rollout_yaml):
        with open(rollout_yaml) as f:
            text = f.read()
        assert_(
            "enable_rollout_routing_replay" in text,
            f"expected 'enable_rollout_routing_replay' in {rollout_yaml}; not found",
        )
        log("  ok: rollout.yaml exposes enable_rollout_routing_replay")
    else:
        log(f"  -- skip: rollout.yaml not found at {rollout_yaml} (verl install layout?)")


def check_hf_config(model_dir: str) -> Any:
    log("checking hf_config has the MoE attributes the recorder needs")
    cfg = AutoConfig.from_pretrained(model_dir)
    needed = ("num_hidden_layers", "num_experts_per_tok", "num_local_experts")
    missing = [k for k in needed if not hasattr(cfg, k)]
    assert_(
        not missing,
        f"hf_config missing {missing}; sglang's recorder requires "
        "num_hidden_layers + num_experts_per_tok at minimum",
    )
    log(
        f"  ok: layers={cfg.num_hidden_layers} experts/tok={cfg.num_experts_per_tok} "
        f"total_experts={cfg.num_local_experts}"
    )
    return cfg


def check_sglang_recorder() -> bool:
    log("checking sglang has the routed_experts capturer (R3 recording path)")
    try:
        from sglang.srt.layers.moe.routed_experts_capturer import (  # noqa: F401
            extract_routed_experts_from_meta_info,
        )
    except Exception as exc:
        log(
            f"  -- not available: {type(exc).__name__}: {exc}\n"
            "     This sglang build does not expose extract_routed_experts_from_meta_info.\n"
            "     R3 *recording* will fail at runtime; R2 may still work.\n"
            "     See https://github.com/sgl-project/sglang/commit/bed301a5acaa9577c9aa706468bdf242f6a43051"
        )
        return False
    log("  ok: extract_routed_experts_from_meta_info is importable")
    return True


def check_router_determinism(model_dir: str, seq_len: int) -> None:
    log("checking the HF gpt-oss router is deterministic given fixed inputs")
    if not torch.cuda.is_available():
        log("  -- skip: no CUDA, determinism check would take forever on CPU")
        return

    device_map = "auto" if torch.cuda.device_count() >= 2 else None
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if device_map is None:
        model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    enc = tokenizer(
        ["The capital of France is"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in enc.items()}

    # Hook every router and record its top-k expert ids.
    captured: list[list[torch.Tensor]] = [[], []]
    pass_idx = [0]

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            # Router output is typically (router_logits, ...). Take argmax over experts.
            logits = out[0] if isinstance(out, tuple) else out
            topk = torch.topk(logits, k=getattr(model.config, "num_experts_per_tok", 4), dim=-1).indices
            captured[pass_idx[0]].append(topk.detach().cpu())
        return hook

    handles = []
    routers = []
    for name, module in model.named_modules():
        if name.endswith(".mlp.router"):
            routers.append(name)
            handles.append(module.register_forward_hook(make_hook(len(routers) - 1)))

    assert_(routers, "no '*.mlp.router' modules found — is this an MoE checkpoint?")

    try:
        with torch.no_grad():
            pass_idx[0] = 0
            model(**inputs)
            pass_idx[0] = 1
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert_(
        len(captured[0]) == len(captured[1]) == len(routers),
        f"hook count mismatch: {len(captured[0])} vs {len(captured[1])} vs {len(routers)} routers",
    )
    mismatched = []
    for i, (a, b) in enumerate(zip(captured[0], captured[1])):
        if not torch.equal(a, b):
            mismatched.append((i, (a != b).float().mean().item()))
    if mismatched:
        sample = mismatched[:3]
        raise AssertionError(
            f"{len(mismatched)}/{len(routers)} routers produced different top-k expert ids "
            f"on identical inputs across two forward passes (e.g. {sample}). "
            "Routing is non-deterministic — R3 replay would be meaningless."
        )
    log(f"  ok: all {len(routers)} routers produced bit-identical top-k decisions across two passes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/model/Huggingface/openai/gpt-oss-20b-bf16",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--strict-sglang",
        action="store_true",
        help="fail (instead of warn) if sglang lacks the routed_experts capturer",
    )
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        log(f"ERROR: {args.model_dir}/config.json not found. Run prepare_model.py first.")
        return 1

    check_verl_schema()
    check_hf_config(args.model_dir)
    sglang_ok = check_sglang_recorder()
    if args.strict_sglang and not sglang_ok:
        raise AssertionError("--strict-sglang and sglang recorder is unavailable")
    check_router_determinism(args.model_dir, args.seq_len)

    log("R3 capability check PASSED" + ("" if sglang_ok else " (sglang recorder absent — recording-only path will fail)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
