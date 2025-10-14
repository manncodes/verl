# CustomSplitLLama Support in veRL

This repository includes built-in support for CustomSplitLLama models in FSDP training.

## What is CustomSplitLLama?

CustomSplitLLama is a hybrid architecture that combines:
- First N layers from an 8B LLaMA model
- An adapter layer to bridge dimensions
- Last M layers from a 70B LLaMA model

## FSDP Support

The FSDP worker (`verl/workers/fsdp_workers.py`) automatically detects and loads CustomSplitLLama models.

### How It Works

1. **Detection**: Checks if `config.json` contains `"architectures": ["CustomSplitLLamaForCausalLM"]`
2. **Loading**: Uses `CustomSplitLLamaForCausalLM.from_pretrained()` instead of `AutoModelForCausalLM`
3. **Fallback**: Uses standard AutoModel for non-CustomSplitLLama models

### Usage

Simply point to your CustomSplitLLama checkpoint:

```yaml
actor_rollout_ref:
  model:
    path: /path/to/your/custom_split_llama_checkpoint
```

The checkpoint should contain:
- `config.json` with `"architectures": ["CustomSplitLLamaForCausalLM"]`
- Model weights (e.g., `model.safetensors` or sharded safetensors)
- Tokenizer files

### Requirements

The custom model implementation must exist at:
- `verl/models/transformers/custom_split_llama.py`

This file should define `CustomSplitLLamaForCausalLM` class with a standard `from_pretrained()` method.

## Example

```bash
# Your checkpoint is ready - just run training
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/checkpoint
```

The worker will automatically:
- ✅ Detect CustomSplitLLama architecture
- ✅ Load weights correctly without shape mismatches
- ✅ Apply FSDP wrapping
- ✅ Start training

## Troubleshooting

If you see:
```
ImportError: CustomSplitLLamaForCausalLM architecture detected but model class not available
```

Make sure `verl/models/transformers/custom_split_llama.py` exists and contains the model definition.

## Implementation

See `verl/workers/fsdp_workers.py` lines 97-106 (import) and lines 386-415 (loading logic).
