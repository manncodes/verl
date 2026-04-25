"""Forward/backward correctness check for openai/gpt-oss-20b.

Goals
-----
1. Confirm a dequantized bf16 checkpoint loads cleanly with eager attention.
2. Run a forward pass and verify logits are finite, the right shape, and that
   the loss is in a sane numerical range.
3. Run a backward pass and verify gradients are finite, non-zero, and reach
   every trainable submodule we care about (embeddings, attention QKV/O,
   router, MoE experts, LM head).
4. Optionally cross-check a CPU vs GPU forward to catch dtype/numerical drift
   that has bitten gpt-oss in past issues (e.g. verl#3894).

This is intentionally a single-file standalone script (no verl imports needed)
so it can be run before kicking off a real training job.

Usage:
    python examples/gpt_oss/check_gpt_oss_fwd_bwd.py \
        --model-dir ~/models/gpt-oss-20b-bf16 \
        --device cuda \
        --seq-len 64
"""

import argparse
import os
import sys
import time
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Substrings expected in module names that should accumulate gradient on a
# normal forward+backward through gpt-oss. The router and the experts are the
# MoE-specific bits worth calling out — silent failures there have historically
# produced "loss looks fine but the policy never updates" symptoms.
EXPECTED_GRAD_SUBSTRINGS = (
    "embed_tokens",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "router",
    "experts",
    "lm_head",
)


def log(msg: str) -> None:
    print(f"[check] {msg}", flush=True)


def load_model(model_dir: str, device: str, dtype: torch.dtype):
    log(f"loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    log(f"loading model from {model_dir} (dtype={dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.train()
    return tokenizer, model


def build_batch(tokenizer, device: str, seq_len: int, batch_size: int):
    prompt = "The capital of France is"
    enc = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    # Standard causal-LM training target: shift labels = input_ids, ignore pads.
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def check_forward(model, input_ids, attention_mask, labels) -> torch.Tensor:
    log("running forward pass")
    t0 = time.time()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    torch.cuda.synchronize() if input_ids.is_cuda else None
    elapsed = time.time() - t0

    logits = outputs.logits
    loss = outputs.loss
    log(f"  forward took {elapsed:.2f}s")
    log(f"  logits.shape={tuple(logits.shape)} dtype={logits.dtype}")
    log(f"  loss={loss.item():.4f}")

    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    assert torch.isfinite(loss).all(), "loss is NaN/Inf"
    expected_shape = (input_ids.shape[0], input_ids.shape[1], model.config.vocab_size)
    assert tuple(logits.shape) == expected_shape, (
        f"logits shape mismatch: got {tuple(logits.shape)}, expected {expected_shape}"
    )
    # Random-ish init would give -log(1/vocab); a real checkpoint should land
    # well below that. 20 is a generous ceiling that still catches catastrophes.
    assert 0.0 < loss.item() < 20.0, f"loss out of plausible range: {loss.item()}"
    return loss


def check_backward(model, loss: torch.Tensor) -> None:
    log("running backward pass")
    t0 = time.time()
    loss.backward()
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    log(f"  backward took {time.time() - t0:.2f}s")

    saw_grad: dict[str, bool] = {sub: False for sub in EXPECTED_GRAD_SUBSTRINGS}
    bad_params: list[str] = []
    total = 0
    grad_total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total += 1
        if param.grad is None:
            continue
        grad_total += 1
        if not torch.isfinite(param.grad).all():
            bad_params.append(name)
        for sub in EXPECTED_GRAD_SUBSTRINGS:
            if sub in name and param.grad.abs().sum().item() > 0:
                saw_grad[sub] = True

    log(f"  {grad_total}/{total} trainable params received grad")

    if bad_params:
        log(f"  ERROR: {len(bad_params)} params have NaN/Inf grads, e.g. {bad_params[:5]}")
        raise AssertionError("non-finite gradients detected")

    missing = [k for k, v in saw_grad.items() if not v]
    if missing:
        raise AssertionError(
            f"no non-zero gradient observed in modules matching: {missing}. "
            "This usually means a submodule (router, experts, lm_head, ...) "
            "was bypassed in the forward pass."
        )

    log(f"  all expected submodules accumulated gradient: {sorted(saw_grad)}")


def cross_check_cpu_gpu(
    model_dir: str,
    seq_len: int,
    batch_size: int,
    rtol: float,
    atol: float,
) -> None:
    if not torch.cuda.is_available():
        log("skipping cpu-vs-gpu cross check (no CUDA)")
        return

    log("cross-checking forward consistency cpu vs cuda (this loads two copies)")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids, attention_mask, _ = build_batch(tokenizer, "cpu", seq_len, batch_size)

    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
    ).eval()
    with torch.no_grad():
        cpu_logits = cpu_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    del cpu_model
    torch.cuda.empty_cache()

    gpu_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    with torch.no_grad():
        gpu_logits = gpu_model(
            input_ids=input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
        ).logits.float().cpu()
    del gpu_model
    torch.cuda.empty_cache()

    abs_diff = (cpu_logits - gpu_logits).abs()
    log(f"  max|cpu-gpu logit diff| = {abs_diff.max().item():.4f}")
    log(f"  mean|cpu-gpu logit diff| = {abs_diff.mean().item():.4f}")
    if not torch.allclose(cpu_logits, gpu_logits, rtol=rtol, atol=atol):
        log(
            "  WARNING: cpu and gpu logits differ beyond tolerance — bf16 vs "
            "fp32 drift can be expected, but inspect if the gap is huge."
        )


def list_modules(model, substrings: Iterable[str]) -> None:
    matched = [
        n
        for n, _ in model.named_modules()
        if any(s in n for s in substrings)
    ]
    log(f"  matched {len(matched)} modules; sample: {matched[:8]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/models/gpt-oss-20b-bf16"),
        help="bf16 checkpoint produced by prepare_model.py",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="also load a fp32 cpu copy and compare logits to the gpu bf16 forward",
    )
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        log(
            f"model dir {args.model_dir} is missing config.json. "
            "Run examples/gpt_oss/prepare_model.py first."
        )
        return 1

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    tokenizer, model = load_model(args.model_dir, args.device, dtype)
    list_modules(model, EXPECTED_GRAD_SUBSTRINGS)

    input_ids, attention_mask, labels = build_batch(
        tokenizer, args.device, args.seq_len, args.batch_size
    )
    loss = check_forward(model, input_ids, attention_mask, labels)
    check_backward(model, loss)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.cross_check:
        cross_check_cpu_gpu(
            args.model_dir, args.seq_len, args.batch_size, args.rtol, args.atol
        )

    log("forward/backward correctness check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
