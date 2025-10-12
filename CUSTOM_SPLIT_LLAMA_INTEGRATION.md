# Custom Split LLaMA Integration in veRL

This document describes the integration of the Custom Split LLaMA model into veRL.

## Overview

The Custom Split LLaMA model is a hybrid architecture that combines:
- **First N layers** from an 8B LLaMA model
- **Adapter layer** to bridge dimensions from 8B to 70B
- **Last M layers** from a 70B LLaMA model

This architecture allows efficient training by leveraging smaller model components for early layers and larger components for later layers.

## Architecture Components

### Model Configuration Requirements

Your model config should include these attributes:
```python
{
    "architectures": ["CustomSplitLLamaForCausalLM"],
    "path8b": "/path/to/8b/llama/model",
    "path70b": "/path/to/70b/llama/model",
    "num_layers_8": 32,  # Number of layers from 8B model
    "num_layers_70": 8,   # Number of layers from 70B model
    "mlp": false,         # Optional: use MLP adapter (default: false)
    "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
    # ... other standard LLaMA config params
}
```

## Files Created

### 1. Transformers Implementation
- **Location**: `verl/verl/models/transformers/custom_split_llama.py`
- **Classes**:
  - `CustomSplitLLamaModel`: Base model
  - `CustomSplitLLamaForCausalLM`: Causal LM wrapper
- **Features**:
  - Gradient checkpointing support
  - Dynamic cache handling
  - Compatible with HuggingFace ecosystem

### 2. Megatron Implementation
- **Location**: `verl/verl/models/custom_split_llama/megatron/`
- **Files**:
  - `modeling_custom_split_llama_megatron.py`: Main Megatron models
  - `layers/parallel_adapter.py`: Parallel adapter layers
  - `checkpoint_utils/custom_split_llama_loader.py`: Weight loading utilities

- **Classes**:
  - `ParallelCustomSplitLLamaModel`: Regular Megatron model
  - `ParallelCustomSplitLLamaForCausalLM`: Causal LM version
  - `ParallelCustomSplitLLamaModelRmPad`: Packed inputs version (remove padding)
  - `ParallelCustomSplitLLamaForCausalLMRmPad`: Causal LM with packed inputs
  - `ParallelCustomSplitLLamaForValueRmPad`: Value model (critic)
  - `ParallelCustomSplitLLamaForCausalLMRmPadPP`: Pipeline parallel support
  - `ParallelCustomSplitLLamaForValueRmPadPP`: Value model with PP

- **Features**:
  - Tensor parallelism (TP)
  - Pipeline parallelism (PP)
  - Sequence parallelism
  - Packed inputs for efficient training

### 3. vLLM Implementation
- **Location**: `verl/vllm_models/custom_split_llama.py`
- **Class**: `CustomSplitLLamaForCausalLM`
- **Features**:
  - Custom weight loading from split checkpoints
  - Compatible with vLLM inference engine
  - Adapter layer with vLLM parallelism

**Note**: To use with vLLM, you need to either:
1. Copy this file to vLLM's model directory, or
2. Register it via vLLM's plugin system, or
3. Add it to your custom vLLM installation

### 4. Model Registry
- **Location**: `verl/verl/models/registry.py`
- **Registration**:
```python
"CustomSplitLLamaForCausalLM": (
    "custom_split_llama",
    ("ParallelCustomSplitLLamaForCausalLMRmPadPP",
     "ParallelCustomSplitLLamaForValueRmPadPP",
     "ParallelCustomSplitLLamaForCausalLMRmPad"),
)
```

## Usage

### 1. FSDP Training

```python
from verl.trainer.config import FSDPConfig
from verl.workers.config import HFModelConfig

# Model configuration
model_config = HFModelConfig(
    model_path="/path/to/custom/split/config",  # Contains config.json with custom attrs
    architectures=["CustomSplitLLamaForCausalLM"],
    # ... other config
)

# FSDP configuration
fsdp_config = FSDPConfig(
    # ... FSDP settings
)

# Training will automatically use the registered model
```

### 2. Megatron Training

```python
from verl.trainer.config import MegatronConfig
from verl.models.registry import ModelRegistry

# Load model class
model_cls = ModelRegistry.load_model_cls(
    model_arch="CustomSplitLLamaForCausalLM",
    value=False  # True for critic/value model
)

# Initialize with Megatron config
model = model_cls(config=your_config, megatron_config=megatron_config)
```

### 3. Weight Loading

#### From HuggingFace Checkpoints
```python
from verl.models.custom_split_llama.megatron.checkpoint_utils.custom_split_llama_loader import (
    load_hf_weights_to_custom_split_llama
)

load_hf_weights_to_custom_split_llama(
    model_path_8b="/path/to/8b/model",
    model_path_70b="/path/to/70b/model",
    adapter_checkpoint_path="/path/to/adapter.pt",
    wrapped_models=models,
    config=config,
    params_dtype=torch.bfloat16,
    is_value_model=False
)
```

#### From Custom State Dicts
```python
from verl.models.custom_split_llama.megatron.checkpoint_utils.custom_split_llama_loader import (
    load_custom_split_llama_weights
)

load_custom_split_llama_weights(
    state_dict_8b=state_dict_8b,
    state_dict_70b=state_dict_70b,
    adapter_state_dict=adapter_state_dict,
    wrapped_models=models,
    config=config,
    params_dtype=torch.bfloat16,
)
```

### 4. vLLM Inference

```python
from vllm import LLM, SamplingParams

# Note: Ensure the vLLM model file is accessible
llm = LLM(
    model="/path/to/custom/split/model",
    # vLLM will automatically detect CustomSplitLLamaForCausalLM
)

# Generate
outputs = llm.generate(prompts, SamplingParams(temperature=0.8))
```

## Weight Loading Details

### HuggingFace Checkpoint Structure

The weight loader expects:

**8B Model Weights** (for first layers):
```
model.layers.0.* -> layers_first.0.*
model.layers.1.* -> layers_first.1.*
...
model.layers.N.* -> layers_first.N.*
```

**Adapter Weights**:
```
adapter.adapter_linear_1.weight
adapter.adapter_linear_2.weight
```

**70B Model Weights** (for last layers):
```
model.layers.72.* -> layers_last.0.*  # If 80 total layers - 8 last = 72
model.layers.73.* -> layers_last.1.*
...
model.layers.79.* -> layers_last.7.*
```

### Weight Mapping for vLLM

The vLLM implementation includes custom weight loading logic that:
1. Maps `layers_first.*` to the first N layers
2. Maps `adapter.*` to the adapter layer at index N
3. Maps `layers_last.*` to layers starting at index N+1

Weight fusion for efficiency:
- Q, K, V projections → `qkv_proj.weight`
- Gate, Up projections → `gate_up_proj.weight`

## Adapter Layer Options

### Standard Adapter (mlp=false)
```
hidden_8b → Linear → hidden_70b
```

### MLP Adapter (mlp=true)
```
hidden_8b → Linear → ReLU → Linear → hidden_70b
```

Configure via the `mlp` attribute in your model config.

## Parallelism Support

### Tensor Parallelism (TP)
- Supported across all layers including adapter
- Automatic weight sharding along appropriate dimensions

### Pipeline Parallelism (PP)
- Layers can be split across pipeline stages
- Adapter layer can be on any pipeline stage

### Sequence Parallelism
- Supported in RmPad versions for packed inputs
- Efficient for variable-length sequences

## Testing

To verify your integration:

```python
# Test model loading
from verl.models.registry import ModelRegistry

supported_archs = ModelRegistry.get_supported_archs()
assert "CustomSplitLLamaForCausalLM" in supported_archs

model_cls = ModelRegistry.load_model_cls("CustomSplitLLamaForCausalLM")
assert model_cls is not None
```

## Troubleshooting

### Issue: Model not found in registry
**Solution**: Ensure `CustomSplitLLamaForCausalLM` is in your model config's `architectures` field.

### Issue: Weight loading fails
**Solution**:
- Verify layer counts match config (`num_layers_8 + num_layers_70 ≤ total layers`)
- Check that paths to 8B and 70B models are correct
- Ensure adapter weights are available if using adapter

### Issue: vLLM can't find model
**Solution**:
- Copy `verl/vllm_models/custom_split_llama.py` to vLLM's model directory
- Or use vLLM's `--model-loader-extra-config` to specify custom model path

## Performance Considerations

1. **Memory Usage**: The split architecture can reduce memory usage compared to full 70B model
2. **Adapter Bottleneck**: The adapter layer can be a bottleneck; consider MLP adapter for better capacity
3. **TP Sharding**: Ensure TP size divides hidden dimensions evenly
4. **PP Stages**: Place adapter layer strategically to balance computation

## Future Enhancements

Potential improvements:
- [ ] Support for more flexible layer selection (e.g., non-contiguous layers)
- [ ] Dynamic adapter architecture (LoRA, IA3, etc.)
- [ ] Automatic layer count optimization
- [ ] Integration with veRL's AutoConfig
- [ ] Support for Mixture of Experts (MoE) variants

## References

- veRL Documentation: https://deepwiki.com/volcengine/verl
- HybridFlow Paper: [Link to paper if available]
- Custom model integration guide: verl/verl/models/README.md

---

**Created**: 2025-10-12
**Status**: ✅ Fully Integrated
