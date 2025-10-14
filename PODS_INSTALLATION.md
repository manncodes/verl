# veRL Installation Guide for GPU Pods

Complete installation guide for running veRL on GPU cloud platforms (RunPod, Lambda Labs, Vast.ai, Paperspace, etc.)

## Quick Installation (One Command)

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/manncodes/verl/feat/custom-split-llama/install_verl_pods.sh)
```

Or clone first:
```bash
git clone https://github.com/manncodes/verl.git && cd verl && git checkout feat/custom-split-llama && bash install_verl_pods.sh
```

## Installation Options

### Option 1: Full Installation (Recommended)

Includes everything: Flash Attention, vLLM, all optimizations

```bash
# Download the script
wget https://raw.githubusercontent.com/manncodes/verl/feat/custom-split-llama/install_verl_pods.sh

# Make it executable
chmod +x install_verl_pods.sh

# Run it
bash install_verl_pods.sh
```

**Duration:** ~15-20 minutes (Flash Attention compilation takes the longest)

**What it installs:**
- ✅ UV package manager
- ✅ PyTorch with correct CUDA version (auto-detected)
- ✅ veRL core dependencies
- ✅ Flash Attention 2 (compiled from source)
- ✅ vLLM for fast inference
- ✅ Liger kernel optimizations
- ✅ Custom Split LLaMA model

### Option 2: Minimal Installation (Fast Testing)

No Flash Attention, basic functionality only

```bash
bash install_verl_minimal.sh
```

**Duration:** ~3-5 minutes

**Use this if:**
- You want to test quickly
- Flash Attention compilation is failing
- You don't need maximum performance

### Option 3: Docker Installation (Most Reliable)

```bash
# Build the image
docker build -f Dockerfile.pods -t verl-custom .

# Run with GPU
docker run --gpus all -it verl-custom

# Inside container, you're ready to go!
python examples/test_custom_split_llama.py
```

## Platform-Specific Instructions

### RunPod

1. **Select Template:** PyTorch 2.x or CUDA 12.x
2. **GPU:** Any A100, H100, or RTX 4090
3. **Storage:** At least 50GB

```bash
# SSH into your pod
# Run installation
bash <(curl -fsSL https://raw.githubusercontent.com/manncodes/verl/feat/custom-split-llama/install_verl_pods.sh)
```

**RunPod-specific tips:**
- Installation directory automatically set to `/workspace`
- Virtual environment persists across restarts
- Mount external storage for datasets

### Lambda Labs

1. **Instance:** Any GPU instance with CUDA
2. **Region:** Choose closest to you

```bash
# SSH into your instance
# Run installation
bash <(curl -fsSL https://raw.githubusercontent.com/manncodes/verl/feat/custom-split-llama/install_verl_pods.sh)
```

**Lambda-specific tips:**
- Pre-installed CUDA and drivers usually work perfectly
- `/home/ubuntu` is persistent storage

### Vast.ai

1. **Image:** Any PyTorch or CUDA image
2. **GPU:** A100 recommended for training

```bash
# Use Jupyter or SSH
# Run installation
bash <(curl -fsSL https://raw.githubusercontent.com/manncodes/verl/feat/custom-split-llama/install_verl_pods.sh)
```

**Vast.ai tips:**
- Check storage space before installation
- Some images need `apt-get update` first

### Paperspace Gradient

1. **Machine:** Any GPU machine
2. **Container:** PyTorch runtime

```bash
# In terminal
bash <(curl -fsSL https://raw.githubusercontent.com/manncodes/verl/feat/custom-split-llama/install_verl_pods.sh)
```

## Verification

After installation, verify everything works:

```bash
# Activate environment
source /workspace/verl/.venv/bin/activate

# Check veRL
python -c "import verl; print(f'veRL: {verl.__version__}')"

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Check Custom Split LLaMA
python -c "from verl.models.registry import ModelRegistry; print('Custom Split LLaMA:', 'CustomSplitLLamaForCausalLM' in ModelRegistry.get_supported_archs())"

# Run quick test
cd /workspace/verl
python examples/test_custom_split_llama.py
```

## Quick Start After Installation

### 1. Activate Environment

```bash
cd /workspace/verl
source .venv/bin/activate

# Or use the quick activation script
./activate_verl.sh
```

### 2. Preprocess GSM8K Data

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

This downloads and preprocesses the GSM8K dataset (~2-3 minutes).

### 3. Prepare Your Model

Create a model directory with `config.json`:

```json
{
  "architectures": ["CustomSplitLLamaForCausalLM"],
  "path8b": "/path/to/llama-8b-model",
  "path70b": "/path/to/llama-70b-model",
  "num_layers_8": 32,
  "num_layers_70": 8,
  "mlp": false,
  "vocab_size": 128256
}
```

### 4. Run Training

```bash
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model/dir
```

## Troubleshooting

### Issue: Flash Attention Compilation Failed

**Solution 1:** Use minimal installation
```bash
bash install_verl_minimal.sh
```

**Solution 2:** Install pre-built wheel
```bash
# Check your CUDA version
nvcc --version

# Download pre-built wheel from
# https://github.com/Dao-AILab/flash-attention/releases

# Install wheel
source .venv/bin/activate
pip install flash_attn-2.x.x+cuXXX-*.whl
```

**Solution 3:** Skip Flash Attention (slower but works)
```bash
# Training will work without Flash Attention
# Just slower memory usage and speed
```

### Issue: CUDA Version Mismatch

**Symptoms:**
- "CUDA driver version is insufficient"
- PyTorch can't find CUDA

**Solution:**
```bash
# Check actual CUDA version
nvidia-smi | grep "CUDA Version"

# Reinstall PyTorch with correct version
source .venv/bin/activate

# For CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Issue: Out of Memory During Installation

**Solution:**
```bash
# Limit compilation jobs
export MAX_JOBS=2
bash install_verl_pods.sh
```

### Issue: Permission Denied

**Solution:**
```bash
# Run with sudo (if needed)
sudo bash install_verl_pods.sh

# Or change installation directory
export INSTALL_DIR=$HOME/workspace
bash install_verl_pods.sh
```

### Issue: Git Clone Failed

**Solution:**
```bash
# If GitHub is slow, use mirror
git clone https://gitclone.com/github.com/manncodes/verl.git

# Or download as zip
wget https://github.com/manncodes/verl/archive/refs/heads/feat/custom-split-llama.zip
unzip custom-split-llama.zip
cd verl-feat-custom-split-llama
```

### Issue: vLLM Installation Failed

**Common Issue:** vLLM is picky about versions

**Solution:**
```bash
source .venv/bin/activate

# Try specific working version
uv pip install vllm==0.8.4

# Or skip vLLM (use transformers instead)
# veRL will work without vLLM, just slower inference
```

### Issue: Installation Log Shows Errors

**Check the log file:**
```bash
# Log file is created in current directory
cat verl_install_*.log

# Search for errors
grep -i error verl_install_*.log
```

## Performance Optimization

### 1. Enable Persistent Storage

Mount external storage for:
- Model checkpoints
- Datasets
- Logs

```bash
# Example for RunPod
mkdir -p /workspace/storage
# Mount your network storage here
```

### 2. Configure Environment Variables

```bash
cat >> ~/.bashrc << 'EOF'
# CUDA optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# PyTorch
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Paths
export PATH="/root/.cargo/bin:$PATH"
EOF

source ~/.bashrc
```

### 3. Pre-download Models

```bash
# Download models before training
python -c "
from transformers import AutoModel
AutoModel.from_pretrained('meta-llama/Llama-3-8B')
AutoModel.from_pretrained('meta-llama/Llama-3-70B')
"
```

## Cost Optimization Tips

### 1. Use Spot Instances

- RunPod: Enable "Spot" instances (50-80% cheaper)
- Vast.ai: Filter by "interruptible" instances
- Lambda Labs: Not available (always on-demand)

### 2. Auto-shutdown

```bash
# Install auto-shutdown script
cat > /workspace/auto_shutdown.sh << 'EOF'
#!/bin/bash
# Shutdown if idle for 30 minutes
if [ $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | awk '{if ($1 > 10) print 1}' | wc -l) -eq 0 ]; then
    shutdown -h now
fi
EOF
chmod +x /workspace/auto_shutdown.sh

# Add to crontab (check every 30 min)
(crontab -l 2>/dev/null; echo "*/30 * * * * /workspace/auto_shutdown.sh") | crontab -
```

### 3. Use Efficient Training Settings

```bash
# In your training script, use:
# - Lower batch size if possible
# - Gradient accumulation
# - Mixed precision (FP16/BF16)
# - LoRA instead of full fine-tuning

export BATCH_SIZE=128
export MICRO_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=16
```

## Monitoring

### 1. GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use nvtop
sudo apt-get install nvtop
nvtop
```

### 2. Training Progress

```bash
# WandB (already included)
# Login once
wandb login

# Training will automatically log to WandB
```

### 3. Disk Space

```bash
# Check space
df -h

# Clean up if needed
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
```

## Common Commands

```bash
# Activate environment
source /workspace/verl/.venv/bin/activate

# Update veRL
cd /workspace/verl
git pull origin feat/custom-split-llama
uv pip install -e . --force-reinstall

# Run training
bash examples/grpo_trainer/run_custom_split_llama_gsm8k_grpo.sh /path/to/model

# Check logs
tail -f nohup.out

# Kill training
pkill -f "python.*verl"
```

## Getting Help

If installation still fails:

1. **Check the log file:**
   ```bash
   cat verl_install_*.log
   ```

2. **Try minimal installation:**
   ```bash
   bash install_verl_minimal.sh
   ```

3. **Use Docker:**
   ```bash
   docker build -f Dockerfile.pods -t verl-custom .
   docker run --gpus all -it verl-custom
   ```

4. **Report issue with:**
   - OS version: `cat /etc/os-release`
   - CUDA version: `nvcc --version && nvidia-smi`
   - GPU model: `nvidia-smi --query-gpu=name --format=csv`
   - Error log: `cat verl_install_*.log`

## References

- [veRL Documentation](https://verl.readthedocs.io/en/latest/)
- [Custom Split LLaMA Guide](CUSTOM_SPLIT_LLAMA_INTEGRATION.md)
- [Training Guide](TRAINING_GUIDE.md)
- [GitHub Repository](https://github.com/manncodes/verl/tree/feat/custom-split-llama)

## Platform Comparison

| Platform | Pros | Cons | Best For |
|----------|------|------|----------|
| **RunPod** | Easy, persistent storage, spot pricing | Sometimes unstable | Development |
| **Lambda Labs** | Reliable, fast GPUs | More expensive, no spot | Production |
| **Vast.ai** | Cheapest, many options | Variable quality | Budget training |
| **Paperspace** | Good UI, Jupyter | Limited GPU selection | Experimentation |

---

**Installation Time by Method:**
- Full installation: ~15-20 minutes
- Minimal installation: ~3-5 minutes
- Docker build: ~25-30 minutes (first time)

**Storage Requirements:**
- Installation: ~10GB
- With models: ~100GB+
- With datasets: ~150GB+

Enjoy training with veRL! 🚀
