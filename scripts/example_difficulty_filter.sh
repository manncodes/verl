#!/bin/bash
# Example script for using the difficulty filter with Hydra configuration

# This script demonstrates different ways to use the difficulty filter
# Modify the paths below to match your setup

echo "========================================"
echo "Difficulty Filter Examples"
echo "========================================"
echo ""
echo "This script demonstrates usage of the refactored difficulty filter"
echo "which reuses verl's existing code infrastructure."
echo ""

# Example 1: Basic usage with command-line overrides
echo "Example 1: Command-line override (Percentile bucketing)"
echo "--------------------------------------------------------"
python scripts/filter_difficulty.py \
    model.path="path/to/your/model" \
    data.path="path/to/your/data.parquet" \
    output_dir="./difficulty_results/percentile" \
    num_samples=5 \
    bucketing_strategy=percentile \
    data.batch_size=4

echo ""
echo "Example 2: Pass rate-based bucketing"
echo "--------------------------------------------------------"
python scripts/filter_difficulty.py \
    model.path="path/to/your/model" \
    data.path="path/to/your/data.parquet" \
    output_dir="./difficulty_results/pass_rate" \
    num_samples=5 \
    bucketing_strategy=pass_rate

echo ""
echo "Example 3: Mean reward-based bucketing"
echo "--------------------------------------------------------"
python scripts/filter_difficulty.py \
    model.path="path/to/your/model" \
    data.path="path/to/your/data.parquet" \
    output_dir="./difficulty_results/mean_reward" \
    num_samples=5 \
    bucketing_strategy=mean_reward

echo ""
echo "Example 4: Adaptive clustering (requires scikit-learn)"
echo "--------------------------------------------------------"
python scripts/filter_difficulty.py \
    model.path="path/to/your/model" \
    data.path="path/to/your/data.parquet" \
    output_dir="./difficulty_results/adaptive" \
    num_samples=10 \
    bucketing_strategy=adaptive

echo ""
echo "Example 5: High-quality estimates with more samples and higher temperature"
echo "--------------------------------------------------------"
python scripts/filter_difficulty.py \
    model.path="path/to/your/model" \
    data.path="path/to/your/data.parquet" \
    output_dir="./difficulty_results/high_quality" \
    num_samples=20 \
    bucketing_strategy=percentile \
    generation.temperature=0.8 \
    generation.top_p=0.95

echo ""
echo "Example 6: Using a custom config file"
echo "--------------------------------------------------------"
echo "First, create a config file at scripts/config/my_filter.yaml"
echo "Then run:"
echo "python scripts/filter_difficulty.py --config-name=my_filter"

echo ""
echo "========================================"
echo "Configuration-based Usage (Recommended)"
echo "========================================"
echo ""
echo "Create a YAML config file with your settings:"
echo ""
echo "# scripts/config/my_filter.yaml"
echo "model:"
echo "  path: /path/to/model"
echo ""
echo "data:"
echo "  path: /path/to/data.parquet"
echo "  batch_size: 8"
echo ""
echo "generation:"
echo "  max_new_tokens: 1024"
echo "  temperature: 0.8"
echo ""
echo "output_dir: ./results"
echo "num_samples: 10"
echo "bucketing_strategy: percentile"
echo ""
echo "Then run:"
echo "python scripts/filter_difficulty.py --config-name=my_filter"
echo ""
echo "========================================"
echo "Key Features"
echo "========================================"
echo ""
echo "This refactored script reuses verl's existing infrastructure:"
echo "  - load_reward_manager(): Same reward calculation as PPO training"
echo "  - RLHFDataset: Same data loading as training"
echo "  - DataProto: Same data structures as training"
echo "  - dump_generations(): Same dump pattern as validation"
echo ""
echo "This ensures consistency between difficulty filtering and training!"
echo ""
