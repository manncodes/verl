"""End-to-end rollout + backward integration test for gpt-oss.

This is the heavyweight precheck: it actually spins up the rollout engine
(sglang and/or vllm), generates a sample, then feeds the generated tokens
back through the HF actor for forward+backward — the same data path a real
training step takes.

Why bother
----------
The static tests (`test_attention_sinks.py`, `check_gpt_oss_fwd_bwd.py`,
`test_router_replay_capability.py`) catch the cheap stuff. The expensive
class of bugs only shows up once the rollout engine is alive:

  * vLLM gpt-oss needs FA3 / TRTLLM (sinks bypass) — vllm #22331, #22279,
    #30919; FA2/SDPA backends produce silent garbage on Hopper or earlier.
  * vLLM 0.12+ needs the harmony encoding pre-warmed and VERL_USE_GPT_OSS=1
    set (verl/workers/rollout/vllm_rollout/vllm_async_server.py).
  * sglang gpt-oss needs `attention_backend=triton` and the bf16 weights
    in safetensors (mxfp4 isn't ingestible).
  * After a generation, the response_ids that come back must be tokenisable
    by the actor's tokenizer — version mismatches between rollout and
    training tokenizers have produced silently garbled training inputs.

The test loads the engine, generates ~16 tokens from a prompt, then tears
the engine down (it owns all GPU memory). It then loads the model in HF
and runs forward+backward on the *generated* token sequence to verify the
full training-step data path. If this passes, you have very high
confidence training will at least start cleanly.

Both engines are tested if installed (vllm is optional). Skip an engine
with `--skip-sglang` / `--skip-vllm`.

Heavyweight: each engine load is ~30s on H100 and grabs ~40GB of GPU
memory. Run on the actual training host. Off by default in the launcher;
opt in with `RUN_ROLLOUT_TEST=1`.

Usage:
    python examples/gpt_oss/test_rollout_e2e.py \
        --model-dir ~/models/gpt-oss-20b-bf16 \
        --tensor-parallel-size 2
"""

import argparse
import gc
import importlib
import os
import sys
import time
from typing import Optional

import torch


def log(msg: str) -> None:
    print(f"[rollout-e2e] {msg}", flush=True)


def assert_(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def have(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


# ---------- engine smoke tests --------------------------------------------


def smoke_test_sglang(model_dir: str, tp: int, max_tokens: int) -> list[int]:
    """Boot sglang, generate from a prompt, return the generated token ids."""
    log("loading sglang.Engine — this takes ~30s and grabs ~40GB GPU memory")
    import sglang as sgl

    # triton attention backend is the supported gpt-oss path; bf16 + safetensors.
    # Mirror examples/grpo_trainer/run_gptoss_20b.sh's runtime config.
    t0 = time.time()
    engine = sgl.Engine(
        model_path=model_dir,
        tp_size=tp,
        attention_backend="triton",
        load_format="safetensors",
        dtype="bfloat16",
        mem_fraction_static=0.7,
        random_seed=42,
    )
    log(f"  sglang up in {time.time() - t0:.1f}s")

    try:
        prompt = "The capital of France is"
        t0 = time.time()
        out = engine.generate(
            prompt,
            sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0},
        )
        log(f"  generated {len(out['output_ids']) if 'output_ids' in out else '?'} tokens in {time.time() - t0:.1f}s")
        token_ids = out.get("output_ids") or out.get("token_ids")
        text = out.get("text", "")
        log(f"  text preview: {text[:120]!r}")
        assert_(token_ids is not None and len(token_ids) > 0, "sglang returned no token_ids")
        assert_(text.strip() != "", "sglang returned empty text")
        return list(token_ids)
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
        del engine
        free_gpu()


def smoke_test_vllm(model_dir: str, tp: int, max_tokens: int) -> list[int]:
    """Boot vllm, generate from a prompt, return the generated token ids."""
    log("loading vllm.LLM — this takes ~30s and grabs ~40GB GPU memory")
    # Required env-var flag in verl's vllm_async_server for gpt-oss harmony pre-warm.
    os.environ.setdefault("VERL_USE_GPT_OSS", "1")
    from vllm import LLM, SamplingParams
    import vllm

    log(f"  vllm version = {vllm.__version__}")
    t0 = time.time()
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        load_format="safetensors",
        enforce_eager=True,  # avoid CUDA graph capture during a smoke test
    )
    log(f"  vllm up in {time.time() - t0:.1f}s")

    try:
        prompt = "The capital of France is"
        t0 = time.time()
        outputs = llm.generate(
            [prompt],
            SamplingParams(max_tokens=max_tokens, temperature=0.0),
        )
        log(f"  generated in {time.time() - t0:.1f}s")
        out = outputs[0].outputs[0]
        log(f"  text preview: {out.text[:120]!r}")
        assert_(len(out.token_ids) > 0, "vllm returned no token_ids")
        assert_(out.text.strip() != "", "vllm returned empty text")
        return list(out.token_ids)
    finally:
        del llm
        free_gpu()


# ---------- backward through generated tokens -----------------------------


def backward_through_generated(model_dir: str, prompt: str, generated_ids: list[int]) -> None:
    """Concatenate prompt + generated tokens and run forward+backward.

    Mirrors what verl's actor does each step: tokenise the rollout's response,
    feed it through the FSDP actor, compute loss, backprop. If anything in
    the rollout->actor handoff is broken (tokenizer drift, dtype mismatch,
    sinks bypass), this surfaces it before training starts.
    """
    log("running forward+backward on (prompt + rollout response) via HF actor")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    full_ids = prompt_ids + list(generated_ids)
    log(f"  prompt={len(prompt_ids)} tok  generated={len(generated_ids)} tok  total={len(full_ids)} tok")

    device_map = "auto" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else None
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if device_map is None and torch.cuda.is_available():
        model.to("cuda")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], device=device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100  # only train on the response, like RL does

    out = model(input_ids=input_ids, labels=labels)
    log(f"  loss = {out.loss.item():.4f}")
    assert_(torch.isfinite(out.loss).all(), "loss is not finite on the (prompt + rollout) sequence")
    out.loss.backward()

    bad = 0
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad += 1
    assert_(bad == 0, f"{bad}/{total} params have non-finite grads after rollout-driven backward")
    log(f"  ok: backward pass through generated tokens completed cleanly")

    del model
    free_gpu()


# ---------- driver --------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/models/gpt-oss-20b-bf16"),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--skip-sglang", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument(
        "--skip-backward",
        action="store_true",
        help="skip the HF backward pass through the generated tokens (fwd-only smoke test)",
    )
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        log(f"ERROR: {args.model_dir}/config.json not found. Run prepare_model.py first.")
        return 1

    if not torch.cuda.is_available():
        log("ERROR: no CUDA visible; rollout engines need GPUs")
        return 1

    prompt = "The capital of France is"
    last_generated: Optional[list[int]] = None

    # ---- sglang ----------------------------------------------------------
    if args.skip_sglang:
        log("--skip-sglang, skipping sglang smoke test")
    elif not have("sglang"):
        log("sglang not installed, skipping (install with verl[sglang])")
    else:
        log("=== sglang smoke test ===")
        last_generated = smoke_test_sglang(args.model_dir, args.tensor_parallel_size, args.max_tokens)

    # ---- vllm ------------------------------------------------------------
    if args.skip_vllm:
        log("--skip-vllm, skipping vllm smoke test")
    elif not have("vllm"):
        log("vllm not installed, skipping (install with: uv pip install 'vllm>=0.12')")
    else:
        log("=== vllm smoke test ===")
        last_generated = smoke_test_vllm(args.model_dir, args.tensor_parallel_size, args.max_tokens)

    if last_generated is None:
        log("ERROR: no rollout engine was available — at least one of sglang or vllm is required")
        return 1

    # ---- backward through the generated tokens --------------------------
    if not args.skip_backward:
        log("=== backward through generated tokens ===")
        backward_through_generated(args.model_dir, prompt, last_generated)

    log("rollout end-to-end test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
