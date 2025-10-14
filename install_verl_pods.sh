#!/bin/bash
# veRL Installation Script for GPU Pods
# Tested on: RunPod, Lambda Labs, Vast.ai, Paperspace
# Compatible with: Ubuntu 20.04/22.04, CUDA 11.8/12.1/12.4
# Usage: bash install_verl_pods.sh

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
LOG_FILE="verl_install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}veRL Installation Script for GPU Pods${NC}"
echo -e "${BLUE}================================================${NC}"
echo "Log file: $LOG_FILE"
echo ""

# ============================================================================
# STEP 0: Environment Detection
# ============================================================================
echo -e "${GREEN}[0/8] Detecting environment...${NC}"

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    OS_VERSION=$VERSION_ID
    echo "OS: $OS $OS_VERSION"
else
    echo -e "${RED}Cannot detect OS${NC}"
    exit 1
fi

# Detect CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "CUDA Version: $CUDA_VERSION"
else
    echo -e "${YELLOW}Warning: nvcc not found. Will detect CUDA from nvidia-smi${NC}"
    CUDA_VERSION="unknown"
fi

# Check NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "NVIDIA Driver: $DRIVER_VERSION"
else
    echo -e "${RED}Error: nvidia-smi not found. GPU not available.${NC}"
    exit 1
fi

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo "Python Version: $PYTHON_VERSION"
else
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

# Determine PyTorch CUDA version
if [[ "$CUDA_VERSION" == 11.* ]]; then
    TORCH_CUDA="cu118"
    echo "Using PyTorch CUDA 11.8"
elif [[ "$CUDA_VERSION" == 12.0* ]] || [[ "$CUDA_VERSION" == 12.1* ]] || [[ "$CUDA_VERSION" == 12.2* ]]; then
    TORCH_CUDA="cu121"
    echo "Using PyTorch CUDA 12.1"
elif [[ "$CUDA_VERSION" == 12.* ]]; then
    TORCH_CUDA="cu124"
    echo "Using PyTorch CUDA 12.4"
else
    # Fallback: detect from nvidia-smi
    CUDA_VERSION_SMI=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    if [[ "$CUDA_VERSION_SMI" == 11.* ]]; then
        TORCH_CUDA="cu118"
    elif [[ "$CUDA_VERSION_SMI" == 12.0* ]] || [[ "$CUDA_VERSION_SMI" == 12.1* ]]; then
        TORCH_CUDA="cu121"
    else
        TORCH_CUDA="cu121"  # Default
    fi
    echo "Using PyTorch CUDA: $TORCH_CUDA (detected from nvidia-smi)"
fi

echo -e "${GREEN}✓ Environment detection complete${NC}\n"

# ============================================================================
# STEP 1: System Dependencies
# ============================================================================
echo -e "${GREEN}[1/8] Installing system dependencies...${NC}"

# Update apt cache
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq

# Install essential build tools
apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    ninja-build \
    ccache \
    > /dev/null 2>&1

echo -e "${GREEN}✓ System dependencies installed${NC}\n"

# ============================================================================
# STEP 2: Install UV (Fast Python Package Installer)
# ============================================================================
echo -e "${GREEN}[2/8] Installing UV package manager...${NC}"

if command -v uv &> /dev/null; then
    echo "UV already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Verify UV installation
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ UV installed: $(uv --version)${NC}\n"
else
    echo -e "${RED}Error: UV installation failed${NC}"
    exit 1
fi

# ============================================================================
# STEP 3: Clone veRL Repository
# ============================================================================
echo -e "${GREEN}[3/8] Cloning veRL repository...${NC}"

INSTALL_DIR="${INSTALL_DIR:-/workspace}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ -d "verl" ]; then
    echo "verl directory exists. Updating..."
    cd verl
    git fetch origin
    git checkout feat/custom-split-llama
    git pull origin feat/custom-split-llama
else
    git clone https://github.com/manncodes/verl.git
    cd verl
    git checkout feat/custom-split-llama
fi

echo -e "${GREEN}✓ Repository ready at: $(pwd)${NC}\n"

# ============================================================================
# STEP 4: Create Virtual Environment
# ============================================================================
echo -e "${GREEN}[4/8] Creating virtual environment...${NC}"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
fi

# Create new venv
uv venv --python python3.10

# Activate venv
source .venv/bin/activate

echo -e "${GREEN}✓ Virtual environment created and activated${NC}\n"

# ============================================================================
# STEP 5: Install PyTorch
# ============================================================================
echo -e "${GREEN}[5/8] Installing PyTorch with CUDA ${TORCH_CUDA}...${NC}"

# Install PyTorch
uv pip install --no-cache-dir torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Verify PyTorch
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch installed successfully${NC}\n"
else
    echo -e "${RED}Error: PyTorch installation verification failed${NC}"
    exit 1
fi

# ============================================================================
# STEP 6: Install veRL Core Dependencies
# ============================================================================
echo -e "${GREEN}[6/8] Installing veRL core dependencies...${NC}"

# Install core dependencies first (without GPU-specific packages)
uv pip install --no-cache-dir \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    "numpy<2.0.0" \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    "ray[default]>=2.41.0" \
    torchdata \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    transformers \
    wandb \
    "packaging>=20.0" \
    tensorboard \
    uvicorn \
    fastapi \
    latex2sympy2_extended \
    math_verify

echo -e "${GREEN}✓ Core dependencies installed${NC}\n"

# ============================================================================
# STEP 7: Install GPU-Specific Packages
# ============================================================================
echo -e "${GREEN}[7/8] Installing GPU-specific packages...${NC}"

# Install flash-attention (this takes the longest)
echo "Installing flash-attention (this may take 5-15 minutes)..."
export MAX_JOBS=4
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

# Try to install flash-attn with error handling
if uv pip install --no-cache-dir flash-attn --no-build-isolation; then
    echo -e "${GREEN}✓ Flash-attention installed${NC}"
else
    echo -e "${YELLOW}⚠ Flash-attention installation failed. Trying alternative method...${NC}"

    # Try with pip directly
    if pip install flash-attn --no-build-isolation; then
        echo -e "${GREEN}✓ Flash-attention installed (via pip)${NC}"
    else
        echo -e "${YELLOW}⚠ Flash-attention failed. Training will be slower but functional.${NC}"
    fi
fi

# Install liger-kernel
echo "Installing liger-kernel..."
uv pip install --no-cache-dir liger-kernel || echo -e "${YELLOW}⚠ liger-kernel installation failed (non-critical)${NC}"

# Install vLLM
echo "Installing vLLM..."
if uv pip install --no-cache-dir "vllm>=0.8.0,<=0.9.1"; then
    echo -e "${GREEN}✓ vLLM installed${NC}"
else
    echo -e "${YELLOW}⚠ vLLM installation failed. Trying specific version...${NC}"
    uv pip install --no-cache-dir vllm==0.8.4 || echo -e "${YELLOW}⚠ vLLM not installed (you can install it later)${NC}"
fi

echo -e "${GREEN}✓ GPU packages installation complete${NC}\n"

# ============================================================================
# STEP 8: Install veRL in Editable Mode
# ============================================================================
echo -e "${GREEN}[8/8] Installing veRL in editable mode...${NC}"

uv pip install --no-cache-dir -e .

echo -e "${GREEN}✓ veRL installed in editable mode${NC}\n"

# ============================================================================
# VERIFICATION
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Verification${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Test veRL
echo "Testing veRL import..."
python3 -c "import verl; print(f'✓ veRL version: {verl.__version__}')" || echo -e "${RED}✗ veRL import failed${NC}"

# Test PyTorch
echo "Testing PyTorch CUDA..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ CUDA version: {torch.version.cuda}'); print(f'✓ GPUs: {torch.cuda.device_count()}')" || echo -e "${RED}✗ PyTorch CUDA test failed${NC}"

# Test flash-attention
echo "Testing flash-attention..."
python3 -c "import flash_attn; print(f'✓ Flash-attention: {flash_attn.__version__}')" || echo -e "${YELLOW}⚠ Flash-attention not available${NC}"

# Test vLLM
echo "Testing vLLM..."
python3 -c "import vllm; print(f'✓ vLLM: {vllm.__version__}')" || echo -e "${YELLOW}⚠ vLLM not available${NC}"

# Test Custom Split LLaMA
echo "Testing Custom Split LLaMA registration..."
python3 -c "from verl.models.registry import ModelRegistry; registered = 'CustomSplitLLamaForCausalLM' in ModelRegistry.get_supported_archs(); print(f'✓ Custom Split LLaMA registered: {registered}')" || echo -e "${YELLOW}⚠ Custom Split LLaMA not registered${NC}"

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Print summary
cat << EOF
Summary:
--------
Installation Directory: $(pwd)
Virtual Environment: .venv
Log File: $LOG_FILE

To activate the environment:
    source .venv/bin/activate

To preprocess GSM8K data:
    source .venv/bin/activate
    python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

To run training:
    source .venv/bin/activate
    bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir

Environment Info:
    OS: $OS $OS_VERSION
    CUDA: $CUDA_VERSION
    PyTorch CUDA: $TORCH_CUDA
    Python: $PYTHON_VERSION
    GPU Driver: $DRIVER_VERSION

Happy Training! 🚀
EOF

# Create activation script
cat > activate_verl.sh << 'ACTIVATE_EOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "veRL environment activated!"
echo "Current directory: $(pwd)"
ACTIVATE_EOF
chmod +x activate_verl.sh

echo ""
echo -e "${GREEN}Quick activation script created: ./activate_verl.sh${NC}"

# Clear trap
trap - EXIT
