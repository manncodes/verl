#!/bin/bash
# Quick setup script for veRL with UV
# Usage: bash setup_with_uv.sh [cuda_version]
# Example: bash setup_with_uv.sh 121  # for CUDA 12.1
#          bash setup_with_uv.sh 118  # for CUDA 11.8

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default CUDA version
CUDA_VERSION=${1:-121}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}veRL Setup with UV${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}UV not found. Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓ UV installed successfully${NC}"
else
    echo -e "${GREEN}✓ UV already installed: $(uv --version)${NC}"
fi

# Check if we're in verl directory
if [ ! -f "setup.py" ] || [ ! -d "verl" ]; then
    echo -e "${RED}Error: Must run from verl directory${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 1: Creating Virtual Environment${NC}"
echo -e "${GREEN}========================================${NC}"

# Create virtual environment
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf .venv
fi

uv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
source .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate 2>/dev/null
echo -e "${GREEN}✓ Virtual environment activated${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 2: Installing PyTorch (CUDA ${CUDA_VERSION})${NC}"
echo -e "${GREEN}========================================${NC}"

# Install PyTorch based on CUDA version
case $CUDA_VERSION in
    118)
        echo "Installing PyTorch for CUDA 11.8..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    121)
        echo "Installing PyTorch for CUDA 12.1..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    124)
        echo "Installing PyTorch for CUDA 12.4..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        ;;
    *)
        echo -e "${YELLOW}Unknown CUDA version: ${CUDA_VERSION}. Using CUDA 12.1...${NC}"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
esac

echo -e "${GREEN}✓ PyTorch installed${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 3: Installing veRL${NC}"
echo -e "${GREEN}========================================${NC}"

# Install verl with dependencies
echo "Installing verl with GPU, vLLM, and math support..."
uv pip install -e ".[gpu,vllm,math]"
echo -e "${GREEN}✓ veRL installed${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 4: Installing Flash Attention${NC}"
echo -e "${GREEN}========================================${NC}"

echo "Installing flash-attn (this may take 5-10 minutes)..."
export MAX_JOBS=8
uv pip install flash-attn --no-build-isolation

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Flash Attention installed${NC}"
else
    echo -e "${YELLOW}⚠ Flash Attention installation failed (non-critical, training will be slower)${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 5: Verification${NC}"
echo -e "${GREEN}========================================${NC}"

# Verify installation
echo ""
echo "Checking veRL..."
python -c "import verl; print(f'veRL version: {verl.__version__}')" && echo -e "${GREEN}✓ veRL OK${NC}" || echo -e "${RED}✗ veRL FAILED${NC}"

echo ""
echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" && echo -e "${GREEN}✓ PyTorch OK${NC}" || echo -e "${RED}✗ PyTorch FAILED${NC}"

echo ""
echo "Checking Flash Attention..."
python -c "import flash_attn; print(f'Flash-Attention: {flash_attn.__version__}')" && echo -e "${GREEN}✓ Flash Attention OK${NC}" || echo -e "${YELLOW}⚠ Flash Attention not available (non-critical)${NC}"

echo ""
echo "Checking vLLM..."
python -c "import vllm; print(f'vLLM: {vllm.__version__}')" && echo -e "${GREEN}✓ vLLM OK${NC}" || echo -e "${YELLOW}⚠ vLLM not available${NC}"

# Check if Custom Split LLaMA is available (for custom fork)
echo ""
echo "Checking Custom Split LLaMA..."
python -c "from verl.models.registry import ModelRegistry; is_registered = 'CustomSplitLLamaForCausalLM' in ModelRegistry.get_supported_archs(); print(f'Custom Split LLaMA registered: {is_registered}')" && echo -e "${GREEN}✓ Custom Split LLaMA available${NC}" || echo -e "${YELLOW}⚠ Custom Split LLaMA not registered (only available in feat/custom-split-llama branch)${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To activate the environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To preprocess GSM8K data:"
echo "  python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k"
echo ""
echo "To run training with Custom Split LLaMA:"
echo "  bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir"
echo ""
echo -e "${GREEN}Happy training! 🚀${NC}"
