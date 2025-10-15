#!/usr/bin/env python3
"""
Create a TINY CustomSplitLLama for testing (1 layer from 8B + 1 layer from 70B)
This is for end-to-end testing of veRL FSDP + vLLM rollout.
"""

import os
import json
import shutil
import torch
from pathlib import Path
from safetensors.torch import save_file, safe_open
from tqdm import tqdm


def create_tiny_custom_split_llama(
    path_8b: str = "/model-zoo/meta-llama_Llama-3.1-8B-Instruct/",
    path_70b: str = "/model-zoo/meta-llama_Llama-3.3-70B-Instruct/",
    output_path: str = "/model-zoo/tiny-custom-split-llama-test",
    num_layers_8: int = 1,
    num_layers_70: int = 1,
    mlp_adapter: bool = True,
):
    """
    Create a tiny CustomSplitLLama checkpoint for testing.

    Args:
        path_8b: Path to LLaMA 3.1 8B model
        path_70b: Path to LLaMA 3.3 70B model
        output_path: Where to save the tiny model
        num_layers_8: Number of layers from 8B (default: 1)
        num_layers_70: Number of layers from 70B (default: 1)
        mlp_adapter: Whether to use MLP adapter
    """

    print("="*60)
    print("Creating Tiny CustomSplitLLama for Testing")
    print("="*60)
    print(f"8B model: {path_8b}")
    print(f"70B model: {path_70b}")
    print(f"Output: {output_path}")
    print(f"Architecture: {num_layers_8} layers (8B) + adapter + {num_layers_70} layers (70B)")
    print(f"MLP adapter: {mlp_adapter}")
    print()

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load configs
    print("Loading model configurations...")
    with open(os.path.join(path_8b, "config.json")) as f:
        config_8b = json.load(f)
    with open(os.path.join(path_70b, "config.json")) as f:
        config_70b = json.load(f)

    # Create CustomSplitLLama config
    config = {
        "architectures": ["CustomSplitLLamaForCausalLM"],
        "model_type": "llama",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": config_8b["bos_token_id"],
        "eos_token_id": config_8b["eos_token_id"],
        "pad_token_id": config_8b.get("pad_token_id", config_8b["eos_token_id"]),
        "hidden_act": "silu",
        "hidden_size": config_8b["hidden_size"],
        "initializer_range": 0.02,
        "intermediate_size": config_8b["intermediate_size"],
        "max_position_embeddings": config_8b["max_position_embeddings"],
        "num_attention_heads": config_8b["num_attention_heads"],
        "num_hidden_layers": num_layers_8,  # This is for the first layers
        "num_key_value_heads": config_8b["num_key_value_heads"],
        "pretraining_tp": 1,
        "rms_norm_eps": config_8b["rms_norm_eps"],
        "rope_theta": config_8b["rope_theta"],
        "rope_scaling": config_8b.get("rope_scaling"),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.57.0",
        "use_cache": False,
        "vocab_size": config_8b["vocab_size"],
        "head_dim": config_8b.get("head_dim", 128),
        "mlp_bias": False,

        # CustomSplitLLama specific
        "path8b": path_8b,
        "path70b": path_70b,
        "num_layers_8": num_layers_8,
        "num_layers_70": num_layers_70,
        "mlp": mlp_adapter,
    }

    print("Saving config.json...")
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load weights
    print("\nLoading 8B model weights...")
    state_dict_8b = {}
    model_files_8b = sorted([f for f in os.listdir(path_8b) if f.endswith('.safetensors')])
    for file in tqdm(model_files_8b, desc="Loading 8B weights"):
        with safe_open(os.path.join(path_8b, file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict_8b[key] = f.get_tensor(key)

    print("Loading 70B model weights...")
    state_dict_70b = {}
    model_files_70b = sorted([f for f in os.listdir(path_70b) if f.endswith('.safetensors')])
    for file in tqdm(model_files_70b, desc="Loading 70B weights"):
        with safe_open(os.path.join(path_70b, file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict_70b[key] = f.get_tensor(key)

    # Build merged state dict
    print("\nMerging weights...")
    merged_state_dict = {}

    # 1. Embedding from 8B
    print("  - Embedding layer (8B)")
    merged_state_dict["model.embed_tokens.weight"] = state_dict_8b["model.embed_tokens.weight"]

    # 2. First layer from 8B (layer 0)
    print(f"  - First {num_layers_8} layer(s) from 8B")
    for i in range(num_layers_8):
        for key, value in state_dict_8b.items():
            if f"model.layers.{i}." in key:
                new_key = key.replace(f"model.layers.{i}.", f"model.layers_first.{i}.")
                merged_state_dict[new_key] = value

    # 3. Adapter weights (random initialization)
    print(f"  - Adapter layer ({'MLP' if mlp_adapter else 'Linear'})")
    if mlp_adapter:
        merged_state_dict["model.adapter_linear_1.weight"] = torch.nn.init.kaiming_uniform_(
            torch.empty(config_70b["hidden_size"], config_8b["hidden_size"])
        )
        merged_state_dict["model.adapter_linear_2.weight"] = torch.nn.init.kaiming_uniform_(
            torch.empty(config_70b["hidden_size"], config_70b["hidden_size"])
        )
    else:
        merged_state_dict["model.adapter.weight"] = torch.nn.init.kaiming_uniform_(
            torch.empty(config_70b["hidden_size"], config_8b["hidden_size"])
        )

    # 4. Last layer from 70B (e.g., layer 79 -> index 0 in layers_last)
    print(f"  - Last {num_layers_70} layer(s) from 70B")
    start_idx_70b = config_70b["num_hidden_layers"] - num_layers_70
    for i in range(start_idx_70b, config_70b["num_hidden_layers"]):
        new_idx = i - start_idx_70b
        for key, value in state_dict_70b.items():
            if f"model.layers.{i}." in key:
                new_key = key.replace(f"model.layers.{i}.", f"model.layers_last.{new_idx}.")
                merged_state_dict[new_key] = value

    # 5. Final norm from 70B
    print("  - Final norm (70B)")
    merged_state_dict["model.norm.weight"] = state_dict_70b["model.norm.weight"]

    # 6. LM head from 70B
    print("  - LM head (70B)")
    merged_state_dict["lm_head.weight"] = state_dict_70b["lm_head.weight"]

    # Save merged checkpoint
    print(f"\nSaving checkpoint to {output_path}/model.safetensors...")
    save_file(merged_state_dict, os.path.join(output_path, "model.safetensors"))

    # Copy tokenizer files from 8B
    print("Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "generation_config.json"
    ]
    for file in tokenizer_files:
        src = os.path.join(path_8b, file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, file))
            print(f"  ✓ {file}")

    # Print summary
    total_params = sum(p.numel() for p in merged_state_dict.values())
    print("\n" + "="*60)
    print("✅ Tiny CustomSplitLLama created successfully!")
    print("="*60)
    print(f"Location: {output_path}")
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture: {num_layers_8} + adapter + {num_layers_70} layers")
    print("\nFiles created:")
    print("  - config.json")
    print("  - model.safetensors")
    print("  - tokenizer files")
    print("\nYou can now test with:")
    print(f"  python test_tiny_model.py {output_path}")
    print("="*60)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create tiny CustomSplitLLama for testing")
    parser.add_argument("--path_8b", type=str,
                       default="/model-zoo/meta-llama_Llama-3.1-8B-Instruct/",
                       help="Path to 8B model")
    parser.add_argument("--path_70b", type=str,
                       default="/model-zoo/meta-llama_Llama-3.3-70B-Instruct/",
                       help="Path to 70B model")
    parser.add_argument("--output", type=str,
                       default="/model-zoo/tiny-custom-split-llama-test",
                       help="Output path")
    parser.add_argument("--num_layers_8", type=int, default=1,
                       help="Number of layers from 8B")
    parser.add_argument("--num_layers_70", type=int, default=1,
                       help="Number of layers from 70B")
    parser.add_argument("--mlp_adapter", action="store_true", default=True,
                       help="Use MLP adapter")

    args = parser.parse_args()

    create_tiny_custom_split_llama(
        path_8b=args.path_8b,
        path_70b=args.path_70b,
        output_path=args.output,
        num_layers_8=args.num_layers_8,
        num_layers_70=args.num_layers_70,
        mlp_adapter=args.mlp_adapter,
    )
