# veRL Setup Guide with UV

Complete guide for setting up veRL using [uv](https://github.com/astral-sh/uv), the fast Python package installer.

## Prerequisites

### 1. Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. CUDA Requirements

Ensure you have CUDA installed (for GPU support):
- CUDA 11.8 or 12.1 recommended
- NVIDIA driver compatible with your CUDA version

Check CUDA version:
```bash
nvcc --version
nvidia-smi
```

## Quick Setup (Recommended)

### Option 1: Install from Your Fork

```bash
# Clone your fork with Custom Split LLaMA integration
git clone https://github.com/manncodes/verl.git
cd verl
git checkout feat/custom-split-llama

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install PyTorch first (with CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install verl with all dependencies
uv pip install -e ".[gpu,vllm,math]"

# Install flash-attention separately (it takes time to compile)
uv pip install flash-attn --no-build-isolation
```

### Option 2: Install from Original veRL

```bash
# Clone original veRL
git clone https://github.com/volcengine/verl.git
cd verl

# Create virtual environment
uv venv
source .venv/bin/activate

# Install PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install verl
uv pip install -e ".[gpu,vllm,math]"
uv pip install flash-attn --no-build-isolation
```

## Detailed Installation Options

### Core Installation (Minimal)

Just the basic verl without optional dependencies:

```bash
uv pip install -e .
```

### GPU Training Support

For GPU training with Flash Attention and Liger kernel:

```bash
uv pip install -e ".[gpu]"
```

This installs:
- `flash-attn` - Fast attention implementation
- `liger-kernel` - Efficient kernels for training

### vLLM Integration

For fast inference with vLLM:

```bash
uv pip install -e ".[vllm]"
```

This installs:
- `vllm>=0.7.3,<=0.9.1` - vLLM inference engine
- Compatible tensordict version

### SGLang Integration

For SGLang-based rollout:

```bash
uv pip install -e ".[sglang]"
```

This installs:
- `sglang[srt,openai]==0.5.2` - SGLang inference engine
- `torch==2.8.0` - Required PyTorch version

### Math Verification

For math problem verification (GSM8K, MATH):

```bash
uv pip install -e ".[math]"
```

This installs:
- `math-verify` - Math answer verification tools

### Complete Installation (All Features)

Install everything:

```bash
uv pip install -e ".[gpu,vllm,math,test]"
```

## Installation by Backend

### FSDP Backend (PyTorch Native)

```bash
# Install PyTorch with CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install verl with GPU support
uv pip install -e ".[gpu,vllm,math]"

# Install flash-attention (takes 5-10 minutes)
uv pip install flash-attn --no-build-isolation
```

### Megatron-LM Backend

```bash
# Install PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install verl with Megatron support
uv pip install -e ".[gpu,vllm,math,mcore]"

# Install flash-attention
uv pip install flash-attn --no-build-isolation

# Install Megatron-LM (if not using mbridge)
cd ..
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

## CUDA Version Specific Installation

### CUDA 11.8

```bash
# Install PyTorch for CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install verl
uv pip install -e ".[gpu,vllm,math]"

# Flash-attention for CUDA 11.8
uv pip install flash-attn --no-build-isolation
```

### CUDA 12.1 (Recommended)

```bash
# Install PyTorch for CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install verl
uv pip install -e ".[gpu,vllm,math]"

# Flash-attention
uv pip install flash-attn --no-build-isolation
```

## Verify Installation

```bash
# Check verl installation
python -c "import verl; print(verl.__version__)"

# Check PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check flash-attention
python -c "import flash_attn; print(f'Flash-Attention: {flash_attn.__version__}')"

# Check vLLM (if installed)
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Check if Custom Split LLaMA is registered (if using your fork)
python -c "from verl.models.registry import ModelRegistry; print('Custom Split LLaMA registered:', 'CustomSplitLLamaForCausalLM' in ModelRegistry.get_supported_archs())"
```

## Troubleshooting

### Issue: Flash Attention Installation Fails

**Solution 1:** Install with pre-built wheels
```bash
uv pip install flash-attn --no-build-isolation
```

**Solution 2:** Use pre-built wheels from GitHub
```bash
# Check https://github.com/Dao-AILab/flash-attention/releases
# Download the appropriate wheel for your CUDA version
uv pip install flash_attn-2.x.x+cuXXX-*.whl
```

**Solution 3:** Skip flash-attention (slower training)
```bash
# Install without GPU extras
uv pip install -e ".[vllm,math]"
```

### Issue: CUDA Version Mismatch

Check your CUDA version:
```bash
nvcc --version
nvidia-smi
```

Match PyTorch CUDA version:
```bash
# For CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory During Installation

**Solution:** Limit parallel builds
```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

### Issue: vLLM Installation Fails

**Solution 1:** Install specific version
```bash
uv pip install vllm==0.8.4
```

**Solution 2:** Use pre-built wheels
```bash
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### Issue: Permission Denied

**Solution:** Install in user mode or virtual environment
```bash
# Create virtual environment first
uv venv
source .venv/bin/activate

# Then install
uv pip install -e ".[gpu,vllm,math]"
```

## Performance Tips

### 1. Use UV's Caching

UV automatically caches packages for faster reinstallation:
```bash
# Clear cache if needed
uv cache clean
```

### 2. Parallel Builds

For faster compilation:
```bash
# Use more CPU cores for compilation
export MAX_JOBS=8
uv pip install flash-attn --no-build-isolation
```

### 3. Pre-built Wheels

Download pre-built wheels to avoid compilation:
```bash
# Check for pre-built wheels
uv pip install --dry-run flash-attn
```

## Development Setup

For development with all tools:

```bash
# Clone repository
git clone https://github.com/manncodes/verl.git
cd verl
git checkout feat/custom-split-llama

# Create virtual environment
uv venv
source .venv/bin/activate

# Install in editable mode with all dependencies
uv pip install -e ".[gpu,vllm,math,test]"

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run tests
pytest tests/
```

## Environment Variables

Set these for optimal performance:

```bash
# CUDA optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# PyTorch optimization
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # Adjust for your GPU

# Flash Attention
export FLASH_ATTENTION_FORCE_BUILD=TRUE  # Force rebuild if needed
```

## Quick Start After Installation

### 1. Preprocess GSM8K Data

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

### 2. Test Installation with Custom Split LLaMA

```bash
python examples/test_custom_split_llama.py
```

### 3. Run Training

```bash
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir
```

## Updating veRL

To update veRL to the latest version:

```bash
cd verl
git pull origin feat/custom-split-llama

# Reinstall
uv pip install -e ".[gpu,vllm,math]" --force-reinstall
```

## Alternative: Using UV Without Virtual Environment

UV can install packages system-wide (not recommended for development):

```bash
# Install globally with uv
uv pip install --system -e ".[gpu,vllm,math]"
```

## Docker Setup with UV

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Clone verl
RUN git clone https://github.com/manncodes/verl.git /workspace/verl
WORKDIR /workspace/verl
RUN git checkout feat/custom-split-llama

# Install dependencies
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install --system -e ".[gpu,vllm,math]"
RUN uv pip install --system flash-attn --no-build-isolation

CMD ["/bin/bash"]
```

## Why Use UV?

- **Faster:** 10-100x faster than pip
- **Reliable:** Better dependency resolution
- **Disk efficient:** Shared package cache
- **Modern:** Written in Rust, actively maintained

## References

- [uv Documentation](https://github.com/astral-sh/uv)
- [veRL Documentation](https://verl.readthedocs.io/en/latest/)
- [Custom Split LLaMA Integration](CUSTOM_SPLIT_LLAMA_INTEGRATION.md)
- [Training Guide](TRAINING_GUIDE.md)
