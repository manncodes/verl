#!/bin/bash

set -e

echo "=============================================="
echo "  Fix FSDP Workers for CustomSplitLLama"
echo "=============================================="
echo ""

# Configuration
VERL_DIR="${VERL_DIR:-/home/jovyan/rl/verl}"
FSDP_WORKERS_PATH="$VERL_DIR/verl/workers/fsdp_workers.py"
CUSTOM_MODEL_SRC="${CUSTOM_MODEL_SRC:-modelling_custom_split_llama.py}"
CUSTOM_MODEL_DST="$VERL_DIR/verl/models/transformers/custom_split_llama.py"

# Check if verl directory exists
if [ ! -d "$VERL_DIR" ]; then
    echo "❌ Error: veRL directory not found at $VERL_DIR"
    echo "   Set VERL_DIR environment variable to your veRL installation path"
    exit 1
fi

# Check if fsdp_workers.py exists
if [ ! -f "$FSDP_WORKERS_PATH" ]; then
    echo "❌ Error: fsdp_workers.py not found at $FSDP_WORKERS_PATH"
    exit 1
fi

echo "✅ Found fsdp_workers.py at: $FSDP_WORKERS_PATH"

# Step 1: Copy custom model file
echo ""
echo "[Step 1/3] Copying custom model file..."
if [ -f "$CUSTOM_MODEL_SRC" ]; then
    mkdir -p "$VERL_DIR/verl/models/transformers"
    cp "$CUSTOM_MODEL_SRC" "$CUSTOM_MODEL_DST"
    echo "  ✅ Copied to: $CUSTOM_MODEL_DST"
else
    echo "  ⚠️  Warning: $CUSTOM_MODEL_SRC not found in current directory"
    echo "     You'll need to copy it manually:"
    echo "     cp modelling_custom_split_llama.py $CUSTOM_MODEL_DST"
fi

# Step 2: Apply the fix using Python script
echo ""
echo "[Step 2/3] Applying fix to fsdp_workers.py..."
python3 fix_fsdp_workers.py "$FSDP_WORKERS_PATH"

# Step 3: Verify the fix
echo ""
echo "[Step 3/3] Verifying the fix..."
if grep -q "CUSTOM_SPLIT_LLAMA_AVAILABLE" "$FSDP_WORKERS_PATH"; then
    echo "  ✅ Fix applied successfully"
else
    echo "  ❌ Fix may not have been applied correctly"
    echo "     Please check $FSDP_WORKERS_PATH manually"
    exit 1
fi

# Check if model file exists and can be imported
echo ""
echo "Testing CustomSplitLLama import..."
cd "$VERL_DIR"
if python3 -c "from verl.models.transformers.custom_split_llama import CustomSplitLLamaForCausalLM; print('✅ Import successful')" 2>&1 | grep -q "Import successful"; then
    echo "  ✅ CustomSplitLLama can be imported"
else
    echo "  ⚠️  Warning: Could not import CustomSplitLLama"
    echo "     This may cause issues during training"
    echo "     Check that $CUSTOM_MODEL_DST exists and is valid"
fi

echo ""
echo "=============================================="
echo "  ✅ Fix Complete!"
echo "=============================================="
echo ""
echo "Your existing checkpoint should now work:"
echo "  Checkpoint: /model-zoo/f1-x-post-training/sft/phase1/run14"
echo ""
echo "The FSDP worker will now:"
echo "  1. Detect CustomSplitLLamaForCausalLM in config.json"
echo "  2. Load it using your custom model class"
echo "  3. Use the existing pretrained weights"
echo ""
echo "Run your training:"
echo "  cd $VERL_DIR"
echo "  bash grpo.sh"
