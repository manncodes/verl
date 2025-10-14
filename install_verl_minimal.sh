#!/bin/bash
# Minimal veRL Installation (No Flash-Attention, for testing)
# Use this if the full installation is failing

set -e

echo "=== Minimal veRL Installation ==="
echo ""

# Install UV
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Setup directory
cd "${INSTALL_DIR:-/workspace}"
if [ ! -d "verl" ]; then
    git clone https://github.com/manncodes/verl.git
fi
cd verl
git checkout feat/custom-split-llama

# Create venv
echo "Creating virtual environment..."
uv venv --python python3.10
source .venv/bin/activate

# Detect CUDA
if nvidia-smi | grep -q "CUDA Version: 11"; then
    TORCH_CUDA="cu118"
elif nvidia-smi | grep -q "CUDA Version: 12"; then
    TORCH_CUDA="cu121"
else
    TORCH_CUDA="cu121"  # default
fi

echo "Installing PyTorch ($TORCH_CUDA)..."
uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "Installing veRL (minimal)..."
uv pip install -e .

echo "Installing math_verify..."
uv pip install math_verify

echo ""
echo "=== Minimal Installation Complete ==="
echo "To activate: source .venv/bin/activate"
echo ""
echo "Note: This is a minimal installation without:"
echo "  - flash-attention (training will be slower)"
echo "  - vLLM (use transformers for inference)"
echo "  - liger-kernel (no kernel optimizations)"
echo ""
echo "For full installation, run: bash install_verl_pods.sh"
