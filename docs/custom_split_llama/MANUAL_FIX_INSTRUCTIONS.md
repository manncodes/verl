# Manual Fix Instructions for CustomSplitLLama FSDP Support

If the automated script doesn't work, follow these manual steps:

## Step 1: Copy Custom Model File

```bash
cd /home/jovyan/rl/verl

# Create directory if it doesn't exist
mkdir -p verl/models/transformers

# Copy your custom model file
cp /path/to/modelling_custom_split_llama.py verl/models/transformers/custom_split_llama.py
```

## Step 2: Update fsdp_workers.py Import Section

Open `/home/jovyan/rl/verl/verl/workers/fsdp_workers.py`

Find this section (around line 91):
```python
################# CustomSplitLLama Registry #################
from transformers import LlamaConfig
from transformers import AutoConfig, AutoModelForCausalLM
from verl.models.custom_split_llama.modelling_custom_split_llama import CustomSplitLLamaForCausalLM

# AutoModelForCausalLM.register(LlamaConfig, CustomSplitLLamaForCausalLM)

# from vllm.model_executor.models import ModelRegistry
# ModelRegistry.register_model("CustomSplitLLamaForCausalLM", CustomSplitLLamaForCausalLM)
################# END of CustomSplitLLama Registry #################
```

Replace it with:
```python
################# CustomSplitLLama Support #################
# Import CustomSplitLLama model
CUSTOM_SPLIT_LLAMA_AVAILABLE = False
CustomSplitLLamaForCausalLM = None

try:
    from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM
    CUSTOM_SPLIT_LLAMA_AVAILABLE = True
    logger.info("✅ CustomSplitLLama support enabled")
except ImportError:
    try:
        from verl.models.custom_split_llama.modelling_custom_split_llama import CustomSplitLLamaForCausalLM
        CUSTOM_SPLIT_LLAMA_AVAILABLE = True
        logger.info("✅ CustomSplitLLama support enabled (alternative path)")
    except ImportError:
        logger.warning("⚠️  CustomSplitLLama not available")
################# END CustomSplitLLama Support #################
```

## Step 3: Update Model Loading Logic

In the same file, find the `_build_model_optimizer` method (around line 370).

Find this section:
```python
            print(f"[DEBUG] actor_model_config : {actor_model_config}")
            print(f"[DEBUG] actor_module_class : {actor_module_class}")

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )
```

Replace it with:
```python
            print(f"[DEBUG] actor_model_config : {actor_model_config}")
            print(f"[DEBUG] actor_module_class : {actor_module_class}")

            # Check if this is a CustomSplitLLama model
            is_custom_split = (
                hasattr(actor_model_config, "architectures")
                and actor_model_config.architectures
                and len(actor_model_config.architectures) > 0
                and "CustomSplitLLamaForCausalLM" in actor_model_config.architectures[0]
            )

            print(f"[DEBUG] is_custom_split: {is_custom_split}")
            print(f"[DEBUG] CUSTOM_SPLIT_LLAMA_AVAILABLE: {CUSTOM_SPLIT_LLAMA_AVAILABLE}")

            if is_custom_split:
                if CUSTOM_SPLIT_LLAMA_AVAILABLE:
                    print(f"[DEBUG] Loading CustomSplitLLama from {local_path}")
                    actor_module = CustomSplitLLamaForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=local_path,
                        torch_dtype=torch_dtype,
                        config=actor_model_config,
                        trust_remote_code=trust_remote_code,
                    )
                else:
                    raise ImportError(
                        "CustomSplitLLamaForCausalLM detected but model class not available. "
                        "Ensure custom_split_llama.py is in verl/models/transformers/"
                    )
            else:
                print(f"[DEBUG] Loading standard model with AutoModelForCausalLM")
                actor_module = actor_module_class.from_pretrained(
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=trust_remote_code,
                )
```

## Step 4: Save and Test

Save the file and test the import:

```bash
cd /home/jovyan/rl/verl
python3 -c "from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM; print('✅ Import OK')"
```

If you see "✅ Import OK", you're good to go!

## Step 5: Run Training

```bash
cd /home/jovyan/rl/verl
bash grpo.sh
```

The training should now work with your existing checkpoint at `/model-zoo/f1-x-post-training/sft/phase1/run14`.

## What This Fix Does

1. **Detects CustomSplitLLama**: Checks if `config.json` has `"architectures": ["CustomSplitLLamaForCausalLM"]`
2. **Uses Custom Loader**: Instead of `AutoModelForCausalLM`, directly uses `CustomSplitLLamaForCausalLM.from_pretrained()`
3. **Loads Your Checkpoint**: Uses the existing pretrained weights without modification

## Verification

You should see these debug logs when training starts:
```
[DEBUG] is_custom_split: True
[DEBUG] CUSTOM_SPLIT_LLAMA_AVAILABLE: True
[DEBUG] Loading CustomSplitLLama from /model-zoo/f1-x-post-training/sft/phase1/run14
✅ CustomSplitLLama support enabled
```

Instead of the error:
```
RuntimeError: Error(s) in loading state_dict for LlamaRMSNorm:
    size mismatch for weight: copying a param with shape torch.Size([8192]) from checkpoint,
    the shape in current model is torch.Size([4096]).
```

## Troubleshooting

### Import Error
If you get `ImportError: cannot import name 'CustomSplitLLamaForCausalLM'`:
- Check the file exists: `ls -l /home/jovyan/rl/verl/verl/models/transformers/custom_split_llama.py`
- Check for syntax errors: `python3 -m py_compile verl/models/transformers/custom_split_llama.py`

### Still Getting Size Mismatch
If you still get size mismatch errors:
- Verify your checkpoint's `config.json` has `"architectures": ["CustomSplitLLamaForCausalLM"]`
- Check the debug logs to see if `is_custom_split` is `True`
- Make sure the model structure in `custom_split_llama.py` matches your checkpoint

### Module Not Found
If you get `ModuleNotFoundError: No module named 'verl.models'`:
- Make sure you're in the verl directory: `cd /home/jovyan/rl/verl`
- Set PYTHONPATH: `export PYTHONPATH=/home/jovyan/rl/verl:$PYTHONPATH`
