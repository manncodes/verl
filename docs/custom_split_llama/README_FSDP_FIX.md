# Fix FSDP to Load CustomSplitLLama

## Problem

You have a pretrained CustomSplitLLama checkpoint that works with Nemo, but FSDP workers fail to load it with:

```
RuntimeError: Error(s) in loading state_dict for LlamaRMSNorm:
    size mismatch for weight: copying a param with shape torch.Size([8192]) from checkpoint,
    the shape in current model is torch.Size([4096]).
```

**Root Cause**: The FSDP worker doesn't recognize `CustomSplitLLamaForCausalLM` architecture and tries to load it as a standard `LlamaModel`, causing shape mismatches.

## Solution

Modify `fsdp_workers.py` to detect and properly load CustomSplitLLama models.

## Quick Fix (Automated)

```bash
# On your pod/server where veRL is installed:
cd /home/jovyan/rl/verl

# Copy the custom model file
cp /path/to/modelling_custom_split_llama.py verl/models/transformers/custom_split_llama.py

# Apply the fix
python3 /mnt/c/Users/MANN\ PATEL/claude_code/verl-setup/fix_fsdp_workers.py \
    verl/workers/fsdp_workers.py

# Or use the all-in-one script:
bash /mnt/c/Users/MANN\ PATEL/claude_code/verl-setup/apply_fsdp_fix.sh
```

## Manual Fix

If the automated script doesn't work, see: [MANUAL_FIX_INSTRUCTIONS.md](MANUAL_FIX_INSTRUCTIONS.md)

## What Gets Changed

### 1. Import Section (Line ~91)

**Before:**
```python
from verl.models.custom_split_llama.modelling_custom_split_llama import CustomSplitLLamaForCausalLM
```

**After:**
```python
CUSTOM_SPLIT_LLAMA_AVAILABLE = False
try:
    from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM
    CUSTOM_SPLIT_LLAMA_AVAILABLE = True
except ImportError:
    logger.warning("CustomSplitLLama not available")
```

### 2. Model Loading (Line ~370)

**Before:**
```python
actor_module = actor_module_class.from_pretrained(...)
```

**After:**
```python
is_custom_split = "CustomSplitLLamaForCausalLM" in actor_model_config.architectures[0]

if is_custom_split:
    actor_module = CustomSplitLLamaForCausalLM.from_pretrained(...)
else:
    actor_module = actor_module_class.from_pretrained(...)
```

## Files Provided

- `fix_fsdp_workers.py` - Python script to automatically patch fsdp_workers.py
- `apply_fsdp_fix.sh` - Complete automated fix (copies model + patches fsdp_workers.py)
- `MANUAL_FIX_INSTRUCTIONS.md` - Step-by-step manual instructions
- `README_FSDP_FIX.md` - This file

## Verification

After applying the fix, you should see:

```
[DEBUG] actor_model_config : LlamaConfig {
  "architectures": ["CustomSplitLLamaForCausalLM"],
  ...
}
[DEBUG] is_custom_split: True
[DEBUG] CUSTOM_SPLIT_LLAMA_AVAILABLE: True
[DEBUG] Loading CustomSplitLLama from /model-zoo/f1-x-post-training/sft/phase1/run14
✅ CustomSplitLLama support enabled
```

And the training will start without shape mismatch errors.

## Your Checkpoint

Your existing checkpoint should work as-is:
- **Path**: `/model-zoo/f1-x-post-training/sft/phase1/run14`
- **Architecture**: CustomSplitLLamaForCausalLM
- **Status**: Pretrained with Nemo
- **No changes needed** to the checkpoint itself

## Support

If you encounter issues:

1. Check the model file exists:
   ```bash
   ls -l /home/jovyan/rl/verl/verl/models/transformers/custom_split_llama.py
   ```

2. Test the import:
   ```bash
   cd /home/jovyan/rl/verl
   python3 -c "from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM"
   ```

3. Check the debug logs when training starts

4. See MANUAL_FIX_INSTRUCTIONS.md for detailed troubleshooting
