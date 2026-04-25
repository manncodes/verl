"""Dequantize openai/gpt-oss-20b from MXFP4 to bf16 so verl FSDP/Megatron can load it.

The HF release ships in MXFP4 format which the FSDP/Megatron training paths cannot
ingest directly; the standard recipe is to dequantize to bf16 once and reuse the
output for both training and the forward/backward correctness check.
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--output-dir",
        default="/model/Huggingface/openai/gpt-oss-20b-bf16",
    )
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        print(f"[prepare_model] {args.output_dir} already populated, skipping.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=False,
        device_map=args.device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    # verl reads attn_implementation off the saved config to pick the eager path.
    model.config.attn_implementation = "eager"
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[prepare_model] saved bf16 checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
