# Fixing CustomSplitLLama with FSDP

## Problem

You're getting this error:
```
RuntimeError: Error(s) in loading state_dict for LlamaRMSNorm:
    size mismatch for weight: copying a param with shape torch.Size([8192]) from checkpoint,
    the shape in current model is torch.Size([4096]).
```

**Root Cause**: The checkpoint at `/model-zoo/f1-x-post-training/sft/phase1/run14` doesn't match your CustomSplitLLama architecture. FSDP's `from_pretrained()` expects a checkpoint where all weight names and shapes exactly match your custom model.

## Solution Overview

You have 2 options:

### Option 1: Create a Merged Checkpoint (Recommended for FSDP)

Create a single checkpoint that merges the 8B and 70B weights according to your CustomSplitLLama architecture.

### Option 2: Use Custom Weight Loading (More Complex)

Override the `_load_pretrained_model` method to load weights from separate checkpoints.

---

## Option 1: Create Merged Checkpoint (RECOMMENDED)

### Step 1: Create the Merged Checkpoint

```bash
cd /home/jovyan/rl/verl

# Run the merge script
python /mnt/c/Users/MANN\ PATEL/claude_code/verl-setup/create_custom_split_checkpoint.py \
    --path_8b "/model-zoo/meta-llama_Llama-3.1-8B-Instruct/" \
    --path_70b "/model-zoo/meta-llama_Llama-3.3-70B-Instruct/" \
    --output_path "/model-zoo/custom-split-llama-merged" \
    --num_layers_8 32 \
    --num_layers_70 8 \
    --mlp_adapter
```

This will create a checkpoint at `/model-zoo/custom-split-llama-merged/` with:
- `model.safetensors` (merged weights)
- `config.json` (CustomSplitLLama config)
- Tokenizer files (copied from 8B model)

### Step 2: Verify the Checkpoint

```bash
# Check if files were created
ls -lh /model-zoo/custom-split-llama-merged/

# Should see:
# - model.safetensors (~15-20GB)
# - config.json
# - tokenizer.json
# - tokenizer_config.json
# - etc.
```

### Step 3: Update Your Training Script

Change your training command to use the new merged checkpoint:

```bash
# In grpo.sh or run_custom_split_llama_gsm8k_grpo.sh
# Change this:
MODEL_PATH=/model-zoo/f1-x-post-training/sft/phase1/run14

# To this:
MODEL_PATH=/model-zoo/custom-split-llama-merged
```

### Step 4: Update the Model File Location

First, check where your custom model file is located:

```bash
cd /home/jovyan/rl/verl
find . -name "*custom_split_llama*.py" -type f
```

You should have the model file at one of these locations:
- `verl/models/transformers/custom_split_llama.py` (original setup)
- `verl/models/custom_split_llama/modelling_custom_split_llama.py` (alternative)

Copy the custom model file you provided to the correct location:

```bash
# Copy the modelling file from Windows to the verl repo
cp /mnt/c/Users/MANN\ PATEL/claude_code/verl-setup/modelling_custom_split_llama.py \
   /home/jovyan/rl/verl/verl/models/transformers/custom_split_llama.py
```

### Step 5: Update fsdp_workers.py Import

Edit `/home/jovyan/rl/verl/verl/workers/fsdp_workers.py`:

Find this section (around line 91):
```python
################# CustomSplitLLama Registry #################
from transformers import LlamaConfig
from transformers import AutoConfig, AutoModelForCausalLM
from verl.models.custom_split_llama.modelling_custom_split_llama import CustomSplitLLamaForCausalLM
```

Replace it with:
```python
################# CustomSplitLLama Registry #################
# Import the custom model
try:
    from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM, CustomSplitLLamaModel
    logger.info("Successfully imported CustomSplitLLamaForCausalLM")
except ImportError as e:
    logger.error(f"Failed to import CustomSplitLLamaForCausalLM: {e}")
    raise
################# END of CustomSplitLLama Registry #################
```

### Step 6: Run Training

```bash
cd /home/jovyan/rl/verl
bash grpo.sh
```

---

## Option 2: Custom Weight Loading (If Merged Checkpoint Doesn't Work)

If Option 1 doesn't work, you can implement custom weight loading in your model:

### Update CustomSplitLLamaForCausalLM

Add this classmethod to your `CustomSplitLLamaForCausalLM` class in `custom_split_llama.py`:

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    """
    Custom from_pretrained that loads weights from separate 8B and 70B checkpoints.
    """
    # Load config
    config = kwargs.pop("config", None)
    if config is None:
        config = LlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Check if this is a merged checkpoint or needs separate loading
    if hasattr(config, "path8b") and hasattr(config, "path70b"):
        # This is a CustomSplitLLama config - need to load from separate checkpoints
        print(f"Loading CustomSplitLLama from separate checkpoints:")
        print(f"  8B: {config.path8b}")
        print(f"  70B: {config.path70b}")

        # Initialize model with random weights
        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        device_map = kwargs.get("device_map", None)

        # Create model on meta device first
        with torch.device("meta"):
            model = cls(config)

        # Load weights manually
        from safetensors.torch import safe_open
        import os

        # Load 8B weights
        state_dict_8b = {}
        for file in os.listdir(config.path8b):
            if file.endswith('.safetensors'):
                with safe_open(os.path.join(config.path8b, file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict_8b[key] = f.get_tensor(key)

        # Load 70B weights
        state_dict_70b = {}
        for file in os.listdir(config.path70b):
            if file.endswith('.safetensors'):
                with safe_open(os.path.join(config.path70b, file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict_70b[key] = f.get_tensor(key)

        # Map weights to model
        # ... (implement weight mapping logic)

        return model
    else:
        # Standard merged checkpoint - use default loading
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
```

**Note**: Option 2 is more complex and error-prone. I recommend Option 1.

---

## Verification Steps

After implementing the solution:

### 1. Test Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/model-zoo/custom-split-llama-merged"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

print("✅ Model loaded successfully!")
print(f"Model architecture: {model.config.architectures}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. Test Generation

```python
prompt = "What is 2+2?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 3. Run Training

```bash
bash grpo.sh
```

Monitor for:
- ✅ Model loads without shape mismatch errors
- ✅ FSDP wrapping completes
- ✅ Training starts

---

## Common Issues

### Issue 1: "Could not import CustomSplitLLamaForCausalLM"

**Solution**: Make sure the model file is in the correct location:
```bash
# Check if file exists
ls -l /home/jovyan/rl/verl/verl/models/transformers/custom_split_llama.py

# If not, copy it
cp /mnt/c/Users/MANN\ PATEL/claude_code/verl-setup/modelling_custom_split_llama.py \
   /home/jovyan/rl/verl/verl/models/transformers/custom_split_llama.py
```

### Issue 2: "No module named 'verl.models.transformers.custom_split_llama'"

**Solution**: Make sure you're running from the correct directory:
```bash
cd /home/jovyan/rl/verl
export PYTHONPATH=/home/jovyan/rl/verl:$PYTHONPATH
python -c "from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM; print('OK')"
```

### Issue 3: Checkpoint creation is slow

**Solution**: This is normal - merging large models takes time. You can monitor progress:
```bash
# In another terminal
watch -n 1 "ls -lh /model-zoo/custom-split-llama-merged/"
```

### Issue 4: Out of memory during checkpoint creation

**Solution**: The merge script loads models on CPU. If you're OOM:
```bash
# Check available RAM
free -h

# If needed, load models in smaller chunks (modify the script)
```

---

## Quick Reference

### File Locations
- Merge script: `/mnt/c/Users/MANN PATEL/claude_code/verl-setup/create_custom_split_checkpoint.py`
- Custom model: `/home/jovyan/rl/verl/verl/models/transformers/custom_split_llama.py`
- FSDP workers: `/home/jovyan/rl/verl/verl/workers/fsdp_workers.py`
- Training script: `/home/jovyan/rl/verl/grpo.sh`

### Key Paths
- 8B model: `/model-zoo/meta-llama_Llama-3.1-8B-Instruct/`
- 70B model: `/model-zoo/meta-llama_Llama-3.3-70B-Instruct/`
- Merged checkpoint: `/model-zoo/custom-split-llama-merged/`
- Old checkpoint (incompatible): `/model-zoo/f1-x-post-training/sft/phase1/run14`

---

## Next Steps

1. ✅ Create merged checkpoint using Option 1
2. ✅ Verify checkpoint was created successfully
3. ✅ Update training script to use new checkpoint
4. ✅ Copy custom model file to correct location
5. ✅ Update fsdp_workers.py import
6. ✅ Run training

If you continue to have issues after following these steps, please share:
1. The exact error message
2. Output of `ls -lh /model-zoo/custom-split-llama-merged/`
3. First 50 lines of the training log
