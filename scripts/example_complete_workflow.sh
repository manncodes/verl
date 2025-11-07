#!/bin/bash
# Complete workflow example for difficulty filtering
# This demonstrates the separation of concerns approach:
# 1. Generate rollouts once (slow)
# 2. Analyze multiple times (fast)

set -e  # Exit on error

# Configuration
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="data/gsm8k_train.parquet"
VLLM_URL="http://localhost:8000/v1"
NUM_SAMPLES=10
TEMPERATURE=0.7

echo "========================================="
echo "Complete Difficulty Filtering Workflow"
echo "========================================="
echo ""

# Step 1: Generate rollouts using VLLM (do this once)
echo "Step 1: Generating rollouts..."
echo "This is the slow step (GPU-intensive)"
echo ""

python scripts/generate_rollouts.py \
    --use_vllm \
    --vllm_url "$VLLM_URL" \
    --data_path "$DATA_PATH" \
    --output_path "./rollouts/gsm8k_rollouts.parquet" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --prompt_key "prompt" \
    --max_concurrent 100

echo ""
echo "Rollouts generated successfully!"
echo ""

# Step 2: Analyze with different bucketing strategies (fast, can run many times)
echo "Step 2: Analyzing rollouts with different strategies..."
echo "These steps are fast (CPU-only)"
echo ""

# Strategy 1: Percentile bucketing
echo "  - Percentile bucketing..."
python scripts/analyze_rollouts.py \
    --rollouts "./rollouts/gsm8k_rollouts.parquet" \
    --model_path "$MODEL_PATH" \
    --output_dir "./results/percentile" \
    --bucketing_strategy percentile

# Strategy 2: Pass rate bucketing
echo "  - Pass rate bucketing..."
python scripts/analyze_rollouts.py \
    --rollouts "./rollouts/gsm8k_rollouts.parquet" \
    --model_path "$MODEL_PATH" \
    --output_dir "./results/pass_rate" \
    --bucketing_strategy pass_rate

# Strategy 3: Mean reward bucketing
echo "  - Mean reward bucketing..."
python scripts/analyze_rollouts.py \
    --rollouts "./rollouts/gsm8k_rollouts.parquet" \
    --model_path "$MODEL_PATH" \
    --output_dir "./results/mean_reward" \
    --bucketing_strategy mean_reward

# Strategy 4: Adaptive (K-means) bucketing
echo "  - Adaptive bucketing..."
python scripts/analyze_rollouts.py \
    --rollouts "./rollouts/gsm8k_rollouts.parquet" \
    --model_path "$MODEL_PATH" \
    --output_dir "./results/adaptive" \
    --bucketing_strategy adaptive

echo ""
echo "========================================="
echo "Workflow completed successfully!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - ./results/percentile/"
echo "  - ./results/pass_rate/"
echo "  - ./results/mean_reward/"
echo "  - ./results/adaptive/"
echo ""
echo "Each directory contains:"
echo "  - rollouts_with_rewards.parquet (all rollouts with rewards)"
echo "  - problem_metrics.json (per-problem metrics)"
echo "  - problem_buckets.json (difficulty assignments)"
echo "  - rollouts_{bucket}.parquet (bucketed datasets)"
echo "  - summary.json (overall statistics)"
echo ""

# Optional: Print summaries
echo "Quick comparison of strategies:"
echo ""
for strategy in percentile pass_rate mean_reward adaptive; do
    echo "--- $strategy ---"
    if [ -f "./results/$strategy/summary.json" ]; then
        cat "./results/$strategy/summary.json" | grep -E "(total_problems|total_rollouts|bucket_distribution)" | head -3
    fi
    echo ""
done
