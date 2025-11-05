#!/bin/bash
# Example script for using the difficulty filter

# This script demonstrates different ways to use the difficulty filter
# Modify the paths below to match your setup

# Configuration
MODEL_PATH="path/to/your/model"
DATA_PATH="path/to/your/data.parquet"
OUTPUT_BASE_DIR="./difficulty_results"

echo "========================================"
echo "Difficulty Filter Examples"
echo "========================================"
echo ""
echo "NOTE: Update MODEL_PATH and DATA_PATH in this script before running!"
echo ""

# Example 1: Basic usage with percentile bucketing
echo "Example 1: Percentile-based bucketing (default)"
echo "----------------------------------------"
python scripts/filter_difficulty.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE_DIR/percentile" \
    --num_samples 5 \
    --bucketing_strategy percentile

echo ""
echo "Example 2: Pass rate-based bucketing"
echo "----------------------------------------"
python scripts/filter_difficulty.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE_DIR/pass_rate" \
    --num_samples 5 \
    --bucketing_strategy pass_rate

echo ""
echo "Example 3: Mean reward-based bucketing"
echo "----------------------------------------"
python scripts/filter_difficulty.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE_DIR/mean_reward" \
    --num_samples 5 \
    --bucketing_strategy mean_reward

echo ""
echo "Example 4: Adaptive clustering (requires scikit-learn)"
echo "----------------------------------------"
python scripts/filter_difficulty.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE_DIR/adaptive" \
    --num_samples 10 \
    --bucketing_strategy adaptive

echo ""
echo "Example 5: High-quality estimates with more samples"
echo "----------------------------------------"
python scripts/filter_difficulty.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE_DIR/high_quality" \
    --num_samples 20 \
    --bucketing_strategy percentile \
    --temperature 0.8

echo ""
echo "========================================"
echo "All examples completed!"
echo "========================================"
echo ""
echo "Results are saved in: $OUTPUT_BASE_DIR"
echo ""
echo "To analyze results, check the following files:"
echo "  - summary_*.json: Overview of bucket distribution"
echo "  - difficulty_metrics_*.json: Detailed metrics per problem"
echo "  - data_*_<bucket>.parquet: Filtered datasets for each bucket"
