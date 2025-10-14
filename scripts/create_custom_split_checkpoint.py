"""
Create a merged checkpoint for CustomSplitLLama from separate 8B and 70B checkpoints.
This checkpoint can then be loaded with FSDP's from_pretrained().
"""

import torch
import os
import json
import shutil
from transformers import LlamaConfig, AutoModelForCausalLM
from safetensors.torch import save_file, safe_open
from tqdm import tqdm


def create_custom_split_checkpoint(
    path_8b: str,
    path_70b: str,
    output_path: str,
    num_layers_8: int = 32,
    num_layers_70: int = 8,
    mlp_adapter: bool = True,
):
    """
    Create a merged checkpoint for CustomSplitLLama.

    Args:
        path_8b: Path to 8B model checkpoint
        path_70b: Path to 70B model checkpoint
        output_path: Where to save the merged checkpoint
        num_layers_8: Number of layers to take from 8B model
        num_layers_70: Number of layers to take from 70B model (from the end)
        mlp_adapter: Whether to use MLP adapter (True) or single linear adapter (False)
    """

    print("Loading configurations...")
    config_8b = LlamaConfig.from_pretrained(path_8b)
    config_70b = LlamaConfig.from_pretrained(path_70b)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Create the config for CustomSplitLLama
    merged_config = {
        "architectures": ["CustomSplitLLamaForCausalLM"],
        "model_type": "llama",
        "num_layers_8": num_layers_8,
        "num_layers_70": num_layers_70,
        "path8b": path_8b,
        "path70b": path_70b,
        "mlp": mlp_adapter,
        "hidden_size": config_8b.hidden_size,  # Overall config uses 8B hidden size
        "num_hidden_layers": num_layers_8,  # Total layers in "layers_first"
        "num_attention_heads": config_8b.num_attention_heads,
        "num_key_value_heads": config_8b.num_key_value_heads,
        "intermediate_size": config_8b.intermediate_size,
        "vocab_size": config_8b.vocab_size,
        "rms_norm_eps": config_8b.rms_norm_eps,
        "rope_theta": config_8b.rope_theta,
        "max_position_embeddings": config_8b.max_position_embeddings,
        "bos_token_id": config_8b.bos_token_id,
        "eos_token_id": config_8b.eos_token_id,
        "pad_token_id": config_8b.pad_token_id,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": getattr(config_8b, "head_dim", 128),
        "hidden_act": "silu",
        "mlp_bias": False,
        "rope_scaling": config_8b.rope_scaling if hasattr(config_8b, "rope_scaling") else None,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.57.0",
        "use_cache": False,
    }

    # Save config
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(merged_config, f, indent=2)

    print("Merging weights...")
    merged_state_dict = {}

    # Load 8B model weights
    print("Loading 8B model weights...")
    model_8b_files = [f for f in os.listdir(path_8b) if f.endswith('.safetensors')]
    state_dict_8b = {}
    for file in tqdm(model_8b_files):
        with safe_open(os.path.join(path_8b, file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict_8b[key] = f.get_tensor(key)

    # Load 70B model weights
    print("Loading 70B model weights...")
    model_70b_files = [f for f in os.listdir(path_70b) if f.endswith('.safetensors')]
    state_dict_70b = {}
    for file in tqdm(model_70b_files):
        with safe_open(os.path.join(path_70b, file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict_70b[key] = f.get_tensor(key)

    # 1. Embedding from 8B
    print("Copying embedding layer...")
    merged_state_dict["model.embed_tokens.weight"] = state_dict_8b["model.embed_tokens.weight"]

    # 2. First N layers from 8B
    print(f"Copying first {num_layers_8} layers from 8B model...")
    for i in range(num_layers_8):
        for key, value in state_dict_8b.items():
            if f"model.layers.{i}." in key:
                new_key = key.replace(f"model.layers.{i}.", f"model.layers_first.{i}.")
                merged_state_dict[new_key] = value

    # 3. Initialize adapter weights (random initialization)
    print("Initializing adapter weights...")
    if mlp_adapter:
        merged_state_dict["model.adapter_linear_1.weight"] = torch.nn.init.kaiming_uniform_(
            torch.empty(config_70b.hidden_size, config_8b.hidden_size)
        )
        merged_state_dict["model.adapter_linear_2.weight"] = torch.nn.init.kaiming_uniform_(
            torch.empty(config_70b.hidden_size, config_70b.hidden_size)
        )
    else:
        merged_state_dict["model.adapter.weight"] = torch.nn.init.kaiming_uniform_(
            torch.empty(config_70b.hidden_size, config_8b.hidden_size)
        )

    # 4. Last M layers from 70B
    print(f"Copying last {num_layers_70} layers from 70B model...")
    start_idx_70b = config_70b.num_hidden_layers - num_layers_70
    for i in range(start_idx_70b, config_70b.num_hidden_layers):
        new_idx = i - start_idx_70b
        for key, value in state_dict_70b.items():
            if f"model.layers.{i}." in key:
                new_key = key.replace(f"model.layers.{i}.", f"model.layers_last.{new_idx}.")
                merged_state_dict[new_key] = value

    # 5. Final norm from 70B
    print("Copying final norm from 70B model...")
    merged_state_dict["model.norm.weight"] = state_dict_70b["model.norm.weight"]

    # 6. LM head from 70B
    print("Copying lm_head from 70B model...")
    merged_state_dict["lm_head.weight"] = state_dict_70b["lm_head.weight"]

    # Save merged checkpoint
    print("Saving merged checkpoint...")
    save_file(merged_state_dict, os.path.join(output_path, "model.safetensors"))

    # Copy tokenizer files
    print("Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "generation_config.json"
    ]
    for file in tokenizer_files:
        src = os.path.join(path_8b, file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_path, file))

    print(f"✅ Merged checkpoint saved to: {output_path}")
    print(f"Total parameters in merged checkpoint: {sum(p.numel() for p in merged_state_dict.values()):,}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create CustomSplitLLama merged checkpoint")
    parser.add_argument("--path_8b", type=str, required=True, help="Path to 8B model")
    parser.add_argument("--path_70b", type=str, required=True, help="Path to 70B model")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for merged checkpoint")
    parser.add_argument("--num_layers_8", type=int, default=32, help="Number of layers from 8B model")
    parser.add_argument("--num_layers_70", type=int, default=8, help="Number of layers from 70B model")
    parser.add_argument("--mlp_adapter", action="store_true", default=True, help="Use MLP adapter")

    args = parser.parse_args()

    create_custom_split_checkpoint(
        path_8b=args.path_8b,
        path_70b=args.path_70b,
        output_path=args.output_path,
        num_layers_8=args.num_layers_8,
        num_layers_70=args.num_layers_70,
        mlp_adapter=args.mlp_adapter,
    )
