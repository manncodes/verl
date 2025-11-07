#!/bin/bash
# Launch multiple VLLM instances with lower TP for better throughput
#
# Why multiple instances?
# - Lower TP = less inter-GPU communication overhead
# - Multiple instances = better parallelism and GPU utilization
# - Load balancing = requests distributed evenly
#
# Example: 8 GPUs available
# Option 1: 1 instance with TP=8 (high communication overhead)
# Option 2: 4 instances with TP=2 (lower overhead, better throughput!)

set -e

# Configuration
MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
NUM_INSTANCES="${2:-4}"
TP_SIZE="${3:-2}"
BASE_PORT=8000

echo "========================================="
echo "Multi-VLLM Instance Launcher"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Instances: $NUM_INSTANCES"
echo "TP per instance: $TP_SIZE"
echo "Total GPUs needed: $((NUM_INSTANCES * TP_SIZE))"
echo "Ports: $BASE_PORT - $((BASE_PORT + NUM_INSTANCES - 1))"
echo "========================================="
echo ""

# Check if enough GPUs available
AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
REQUIRED_GPUS=$((NUM_INSTANCES * TP_SIZE))

if [ $AVAILABLE_GPUS -lt $REQUIRED_GPUS ]; then
    echo "ERROR: Not enough GPUs!"
    echo "  Available: $AVAILABLE_GPUS"
    echo "  Required: $REQUIRED_GPUS"
    exit 1
fi

echo "Available GPUs: $AVAILABLE_GPUS ✓"
echo ""

# Launch instances
PIDS=()
URLS=()

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    GPU_START=$((i * TP_SIZE))
    GPU_END=$((GPU_START + TP_SIZE - 1))

    # Build CUDA_VISIBLE_DEVICES
    GPU_IDS=$(seq -s, $GPU_START $GPU_END)

    echo "Launching instance $i:"
    echo "  Port: $PORT"
    echo "  GPUs: $GPU_IDS"
    echo "  TP: $TP_SIZE"

    # Launch VLLM in background
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.95 \
        > "vllm_instance_${i}.log" 2>&1 &

    PID=$!
    PIDS+=($PID)
    URLS+=("http://localhost:${PORT}/v1")

    echo "  PID: $PID"
    echo "  Log: vllm_instance_${i}.log"
    echo ""

    # Small delay to stagger startup
    sleep 2
done

echo "========================================="
echo "All instances launched!"
echo "========================================="
echo ""

# Print URLs for copy-paste
echo "Load-balanced URL argument:"
echo "--vllm_base_url \"$(IFS=,; echo "${URLS[*]}")\""
echo ""

# Print full command
echo "Example command:"
echo "python scripts/generate_rollouts.py \\"
echo "    --use_vllm \\"
echo "    --vllm_base_url \"$(IFS=,; echo "${URLS[*]}")\" \\"
echo "    --model_path $MODEL_PATH \\"
echo "    --data_path data.parquet \\"
echo "    --output_path rollouts.parquet \\"
echo "    --num_samples 10 \\"
echo "    --vllm_max_concurrent 200"
echo ""

# Wait for health checks
echo "Waiting for instances to become healthy..."
echo "(This may take 1-2 minutes)"
echo ""

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    URL="http://localhost:${PORT}/v1/models"

    echo -n "Instance $i (port $PORT): "

    # Wait up to 120 seconds for health check
    for attempt in $(seq 1 120); do
        if curl -s -f "$URL" > /dev/null 2>&1; then
            echo "✓ Ready"
            break
        fi

        if [ $attempt -eq 120 ]; then
            echo "✗ Timeout"
            echo "  Check log: vllm_instance_${i}.log"
        fi

        sleep 1
    done
done

echo ""
echo "========================================="
echo "All instances ready!"
echo "========================================="
echo ""
echo "To stop all instances:"
echo "  kill ${PIDS[@]}"
echo ""
echo "Or run:"
echo "  pkill -f 'vllm.entrypoints.openai.api_server'"
echo ""

# Save PIDs to file for easy cleanup
echo "${PIDS[@]}" > vllm_instances.pids
echo "PIDs saved to: vllm_instances.pids"
echo ""

# Keep script running and trap Ctrl+C
cleanup() {
    echo ""
    echo "Shutting down all VLLM instances..."
    kill ${PIDS[@]} 2>/dev/null || true
    wait ${PIDS[@]} 2>/dev/null || true
    echo "All instances stopped."
    rm -f vllm_instances.pids
    exit 0
}

trap cleanup INT TERM

echo "Press Ctrl+C to stop all instances"
echo ""

# Wait for all background processes
wait
