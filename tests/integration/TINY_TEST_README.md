# End-to-End Test: Tiny CustomSplitLLama with veRL FSDP + vLLM

This is a complete test suite to verify CustomSplitLLama works with veRL's FSDP + vLLM rollout.

## What This Tests

1. ✅ Creating a tiny CustomSplitLLama (1 layer 8B + 1 layer 70B)
2. ✅ Loading with transformers AutoModel
3. ✅ Forward pass and text generation
4. ✅ FSDP worker detection and loading
5. ✅ vLLM rollout engine
6. ✅ GRPO training on GSM8K
7. ✅ WandB logging

## Quick Start

### On Your GPU Pod

```bash
# 1. Upload these files to your pod:
#    - create_tiny_custom_split.py
#    - test_tiny_model.py
#    - run_tiny_test.sh

# 2. Make scripts executable
chmod +x run_tiny_test.sh

# 3. Set up WandB (if you want logging)
wandb login

# 4. Run the complete test
bash run_tiny_test.sh
```

## What Happens

### Step 1: Create Tiny Model (~30 seconds)

```bash
python3 create_tiny_custom_split.py \
    --path_8b /model-zoo/meta-llama_Llama-3.1-8B-Instruct/ \
    --path_70b /model-zoo/meta-llama_Llama-3.3-70B-Instruct/ \
    --output /model-zoo/tiny-custom-split-llama-test \
    --num_layers_8 1 \
    --num_layers_70 1 \
    --mlp_adapter
```

Creates:
- `/model-zoo/tiny-custom-split-llama-test/config.json`
- `/model-zoo/tiny-custom-split-llama-test/model.safetensors`
- Tokenizer files

Architecture:
- 1 layer from LLaMA 3.1 8B (layer 0)
- MLP adapter (4096 → 8192)
- 1 layer from LLaMA 3.3 70B (layer 79)
- Total: ~500M parameters (tiny!)

### Step 2: Test Model (~10 seconds)

```bash
python3 test_tiny_model.py /model-zoo/tiny-custom-split-llama-test
```

Tests:
1. Loading with AutoModelForCausalLM
2. Forward pass
3. Text generation
4. FSDP worker compatibility

### Step 3: Run Training (~5-10 minutes)

```bash
cd /home/jovyan/rl/verl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=/model-zoo/tiny-custom-split-llama-test \
    # ... (see run_tiny_test.sh for full config)
```

Configuration:
- **Batch size**: 8 (small for testing)
- **Epochs**: 2 (just to verify it works)
- **GPUs**: 1
- **TP/PP**: 1 (no model parallelism needed)
- **Rollout**: vLLM
- **Dataset**: GSM8K

### Step 4: Check WandB

Go to https://wandb.ai and check your project for:
- Loss curves
- Reward metrics
- KL divergence
- Token statistics

## Expected Output

### Successful Run

```
============================================
Step 1: Creating Tiny CustomSplitLLama
============================================
Loading 8B weights: 100%
Loading 70B weights: 100%
✅ Tiny CustomSplitLLama created successfully!
Total parameters: 523,456,789

============================================
Step 2: Testing Model Loading
============================================
✓ Tokenizer loaded
✓ Model loaded
✓ Forward pass successful
✓ Generation successful
✓ FSDP worker compatible

============================================
Step 3: Running GRPO Training
============================================
[INFO] Loading CustomSplitLLama model from /model-zoo/tiny-custom-split-llama-test
[INFO] ✅ CustomSplitLLama support enabled
...
Epoch 1/2: 100%
Epoch 2/2: 100%

✅ Training completed successfully!
Check WandB: https://wandb.ai/your-project
```

## Troubleshooting

### Model Creation Fails

```
Error: 8B model not found
```

**Fix**: Update paths in `run_tiny_test.sh`:
```bash
MODEL_8B="/your/path/to/llama-3.1-8B"
MODEL_70B="/your/path/to/llama-3.3-70B"
```

### Import Error

```
ImportError: cannot import name 'CustomSplitLLamaForCausalLM'
```

**Fix**: Copy your custom model file:
```bash
cp modelling_custom_split_llama.py /home/jovyan/rl/verl/verl/models/transformers/custom_split_llama.py
```

### FSDP Shape Mismatch

```
RuntimeError: size mismatch for weight
```

**Fix**: This means the FSDP fix wasn't applied. Check:
```bash
cd /home/jovyan/rl/verl
git pull fork feat/custom-split-llama
grep "CUSTOM_SPLIT_LLAMA_AVAILABLE" verl/workers/fsdp_workers.py
```

### vLLM Rollout Fails

```
Error in vLLM engine
```

**Fix**: The tiny model might be too small for vLLM. Try HF rollout:
```bash
# In run_tiny_test.sh, change:
actor_rollout_ref.rollout.name=vllm
# to:
actor_rollout_ref.rollout.name=hf
```

### Out of Memory

```
torch.cuda.OutOfMemoryError
```

**Fix**: Reduce batch size:
```bash
# In run_tiny_test.sh:
data.train_batch_size=4  # was 8
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1  # was 2
```

### WandB Not Logging

```
wandb: WARNING Run ____ does not have access to project
```

**Fix**: Initialize WandB:
```bash
wandb login
# Then rerun the test
```

## Files

- `create_tiny_custom_split.py` - Creates the tiny model
- `test_tiny_model.py` - Tests model loading and generation
- `run_tiny_test.sh` - Complete end-to-end test script
- `TINY_TEST_README.md` - This file

## Manual Steps (if automation fails)

### 1. Create Tiny Model

```python
from create_tiny_custom_split import create_tiny_custom_split_llama

model_path = create_tiny_custom_split_llama(
    path_8b="/model-zoo/meta-llama_Llama-3.1-8B-Instruct/",
    path_70b="/model-zoo/meta-llama_Llama-3.3-70B-Instruct/",
    output_path="/model-zoo/tiny-custom-split-llama-test",
    num_layers_8=1,
    num_layers_70=1,
    mlp_adapter=True,
)
```

### 2. Test Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/model-zoo/tiny-custom-split-llama-test",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/model-zoo/tiny-custom-split-llama-test")

# Test generation
inputs = tokenizer("What is 2+2?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 3. Run Training

```bash
cd /home/jovyan/rl/verl
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /model-zoo/tiny-custom-split-llama-test
```

## Success Criteria

✅ The test passes if:
1. Tiny model creates without errors
2. Model loads with transformers
3. Training starts without errors
4. WandB shows metrics being logged
5. Training completes 2 epochs

You don't need to see good results - we're just testing the infrastructure works!

## Next Steps

After this test passes:
1. Use your real CustomSplitLLama checkpoint
2. Increase batch size and epochs
3. Run on multiple GPUs with TP/PP
4. Train to convergence

## Questions?

If the test fails, check:
1. Model files exist at correct paths
2. veRL repo has the FSDP fix (check git log)
3. custom_split_llama.py is in verl/models/transformers/
4. GPU has enough memory (at least 24GB for tiny model + training)
