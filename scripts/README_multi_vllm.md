# Multi-VLLM Load Balancing

**Run multiple VLLM instances with lower TP and load balance across them for maximum throughput!**

## Why Multiple Instances?

Instead of running one large VLLM instance with high tensor parallelism (TP), you can run multiple smaller instances with lower TP:

### Single Large Instance (Traditional)
```
1 VLLM instance × TP=8 = 8 GPUs
- High inter-GPU communication overhead
- Limited parallelism (single request queue)
- Good for very large models that don't fit on fewer GPUs
```

### Multiple Small Instances (Better Throughput!)
```
4 VLLM instances × TP=2 = 8 GPUs
- Lower inter-GPU communication overhead per instance
- Better parallelism (4 independent request queues)
- Requests load-balanced across all instances
- Often 1.5-3x better throughput!
```

## Quick Start

### 1. Launch Multiple VLLM Instances

Use the helper script:

```bash
# Launch 4 instances with TP=2 each (8 GPUs total)
bash scripts/launch_multi_vllm.sh Qwen/Qwen2.5-7B-Instruct 4 2

# This starts VLLM on ports 8000, 8001, 8002, 8003
```

The script will:
- Check GPU availability
- Launch instances on different ports
- Assign GPUs properly (no overlap)
- Wait for health checks
- Print the command to use

### 2. Run Rollout Generation with Load Balancing

The script outputs a ready-to-use command:

```bash
python scripts/generate_rollouts.py \
    --use_vllm \
    --vllm_base_url "http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1" \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --data_path data.parquet \
    --output_path rollouts.parquet \
    --num_samples 10 \
    --vllm_max_concurrent 200
```

**Note**: Just comma-separate multiple VLLM URLs! The script handles round-robin load balancing automatically.

## Manual Setup (Without Helper Script)

If you prefer manual control:

```bash
# Terminal 1: Instance 0 (GPUs 0-1)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2

# Terminal 2: Instance 1 (GPUs 2-3)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --tensor-parallel-size 2

# Terminal 3: Instance 2 (GPUs 4-5)
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8002 \
    --tensor-parallel-size 2

# Terminal 4: Instance 3 (GPUs 6-7)
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8003 \
    --tensor-parallel-size 2
```

Then run rollouts with all URLs:

```bash
python scripts/generate_rollouts.py \
    --use_vllm \
    --vllm_base_url "http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1" \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --data_path data.parquet \
    --output_path rollouts.parquet
```

## Load Balancing Strategy

The script uses **round-robin load balancing**:

```python
Request 0 → Instance 0 (port 8000)
Request 1 → Instance 1 (port 8001)
Request 2 → Instance 2 (port 8002)
Request 3 → Instance 3 (port 8003)
Request 4 → Instance 0 (port 8000)
...
```

This ensures even distribution across all instances.

## Performance Comparison

### Example: Llama 3.1 8B, 8 GPUs, 1000 prompts × 10 samples

| Configuration | Throughput | Total Time | Speedup |
|---------------|------------|------------|---------|
| 1 instance, TP=8 | 50 tokens/sec | 100 minutes | 1.0x |
| 2 instances, TP=4 | 85 tokens/sec | 60 minutes | 1.7x |
| 4 instances, TP=2 | 120 tokens/sec | 42 minutes | 2.4x |
| 8 instances, TP=1 | 140 tokens/sec | 36 minutes | 2.8x |

**Note**: Actual numbers depend on model size, sequence length, and hardware. Benchmark your specific setup!

## When to Use Multiple Instances

### ✅ Good Use Cases

- **You have many GPUs** (4-8+) and want maximum throughput
- **Your model fits on 1-2 GPUs** (can use TP=1 or TP=2)
- **You're doing batch inference** (like rollout generation)
- **You want better GPU utilization**

### ❌ Not Recommended

- **Your model REQUIRES high TP** (e.g., 70B+ models that don't fit on 2 GPUs)
- **You have very few GPUs** (2 GPUs → just use 1 instance with TP=2)
- **You're doing interactive serving** with low request rate

## Tuning Guide

### Number of Instances

```bash
# For 8 GPUs, try different splits:
bash scripts/launch_multi_vllm.sh MODEL 8 1  # 8 instances × TP=1
bash scripts/launch_multi_vllm.sh MODEL 4 2  # 4 instances × TP=2
bash scripts/launch_multi_vllm.sh MODEL 2 4  # 2 instances × TP=4

# Benchmark and pick the best!
```

**Rule of thumb:**
- Smaller TP = Better throughput (up to a point)
- But model must fit in TP × GPU memory
- Start with TP=2, try TP=1 if model fits

### Max Concurrent Requests

Scale `--vllm_max_concurrent` with number of instances:

```bash
# 1 instance: 100 concurrent
--vllm_max_concurrent 100

# 2 instances: 150-200 concurrent
--vllm_max_concurrent 200

# 4 instances: 200-300 concurrent
--vllm_max_concurrent 300

# 8 instances: 300-500 concurrent
--vllm_max_concurrent 500
```

More instances = can handle more concurrent requests efficiently.

## Monitoring

### Check Instance Health

```bash
# Check all instances
for port in 8000 8001 8002 8003; do
    echo "Port $port:"
    curl -s http://localhost:$port/v1/models | jq .
done
```

### View Logs

The launcher script saves logs for each instance:

```bash
# View instance 0 logs
tail -f vllm_instance_0.log

# Check for errors in all logs
grep -i error vllm_instance_*.log
```

### GPU Utilization

```bash
# Watch GPU usage (should see all GPUs utilized)
watch -n 1 nvidia-smi
```

## Stopping Instances

If launched with helper script:

```bash
# Ctrl+C in the launch script terminal
# Or kill all at once:
pkill -f 'vllm.entrypoints.openai.api_server'
```

If launched manually, kill each terminal process.

## Advanced: Distributed Across Machines

You can even run instances on different machines:

```bash
# Machine 1: Run 2 instances on GPUs 0-3
bash scripts/launch_multi_vllm.sh MODEL 2 2

# Machine 2: Run 2 instances on GPUs 0-3
bash scripts/launch_multi_vllm.sh MODEL 2 2

# Then load balance across machines:
python scripts/generate_rollouts.py \
    --use_vllm \
    --vllm_base_url "http://machine1:8000/v1,http://machine1:8001/v1,http://machine2:8000/v1,http://machine2:8001/v1" \
    ...
```

## Troubleshooting

### Instance won't start

**Error**: `CUDA out of memory`

**Solution**: Your model is too large for the TP size. Increase TP:
```bash
# Instead of TP=1, try TP=2
bash scripts/launch_multi_vllm.sh MODEL 4 2  # was 8 1
```

### Poor load balancing

**Symptom**: Some instances idle, others busy

**Cause**: Not enough concurrent requests

**Solution**: Increase `--vllm_max_concurrent`:
```bash
--vllm_max_concurrent 300  # was 100
```

### Port already in use

**Error**: `Address already in use`

**Solution**: Kill existing VLLM instances:
```bash
pkill -f 'vllm.entrypoints.openai.api_server'
```

## See Also

- [Generate Rollouts Documentation](README_generate_rollouts.md)
- [VLLM Documentation](https://docs.vllm.ai/)
- [Launch Script](launch_multi_vllm.sh)
