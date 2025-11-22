#!/bin/bash
# Complete curriculum learning pipeline
# From rollouts to preprocessed training data

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default parameters
ROLLOUT_DIR=""
OUTPUT_BASE="./curriculum_output"
NUM_STAGES=3
METRIC="mean_reward"
FILTER_MODE="all"
COMBINE=false
STAGE_NAMES=""

# Function to print colored output
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 -i <rollout_dir> [options]

Complete curriculum learning pipeline from rollouts to training data.

Required:
    -i, --input DIR         Input directory containing rollout JSONL files

Optional:
    -o, --output DIR        Output base directory (default: ./curriculum_output)
    -n, --num-stages N      Number of difficulty stages (default: 3)
    -m, --metric METRIC     Difficulty metric (default: mean_reward)
                           Options: mean_reward, pass_at_k, variance, max_reward, success_rate
    -f, --filter MODE       Filter mode (default: all)
                           Options: all, best_only, passing_only
    -c, --combine          Also create combined datasets for progressive training
    --stages NAMES         Custom stage names (space-separated, must match num-stages)
    -h, --help             Show this help message

Examples:
    # Basic 3-stage curriculum
    $0 -i ./rollouts

    # 5-stage curriculum with custom names
    $0 -i ./rollouts -n 5 --stages "warmup basic intermediate advanced expert"

    # Code generation with pass@k metric
    $0 -i ./code_rollouts -m pass_at_k -f passing_only

    # Create combined datasets for progressive training
    $0 -i ./rollouts -c

Output Structure:
    <output_base>/
    ├── bucketed/           # Bucketed JSONL files
    │   ├── easy.jsonl
    │   ├── medium.jsonl
    │   └── hard.jsonl
    ├── preprocessed/       # Individual stage parquet files
    │   ├── easy_train.parquet
    │   ├── easy_val.parquet
    │   ├── medium_train.parquet
    │   └── ...
    └── progressive/        # Combined datasets (if --combine used)
        ├── stage1_train.parquet
        ├── stage1_and_2_train.parquet
        └── all_stages_train.parquet
EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            ROLLOUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        -n|--num-stages)
            NUM_STAGES="$2"
            shift 2
            ;;
        -m|--metric)
            METRIC="$2"
            shift 2
            ;;
        -f|--filter)
            FILTER_MODE="$2"
            shift 2
            ;;
        -c|--combine)
            COMBINE=true
            shift
            ;;
        --stages)
            STAGE_NAMES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$ROLLOUT_DIR" ]; then
    print_error "Input directory is required!"
    usage
fi

if [ ! -d "$ROLLOUT_DIR" ]; then
    print_error "Input directory does not exist: $ROLLOUT_DIR"
    exit 1
fi

# Create output directories
BUCKET_DIR="${OUTPUT_BASE}/bucketed"
PREPROCESS_DIR="${OUTPUT_BASE}/preprocessed"
PROGRESSIVE_DIR="${OUTPUT_BASE}/progressive"

mkdir -p "$BUCKET_DIR"
mkdir -p "$PREPROCESS_DIR"

if [ "$COMBINE" = true ]; then
    mkdir -p "$PROGRESSIVE_DIR"
fi

# Print configuration
echo "========================================"
echo "Curriculum Learning Pipeline"
echo "========================================"
print_info "Input directory: $ROLLOUT_DIR"
print_info "Output directory: $OUTPUT_BASE"
print_info "Number of stages: $NUM_STAGES"
print_info "Difficulty metric: $METRIC"
print_info "Filter mode: $FILTER_MODE"
print_info "Create progressive datasets: $COMBINE"
if [ -n "$STAGE_NAMES" ]; then
    print_info "Stage names: $STAGE_NAMES"
fi
echo "========================================"
echo ""

# Step 1: Bucket rollouts by difficulty
print_step "Bucketing rollouts by difficulty..."

BUCKET_CMD="python bucket_rollouts_by_difficulty.py -i $ROLLOUT_DIR -o $BUCKET_DIR -m $METRIC -n $NUM_STAGES"

if [ -n "$STAGE_NAMES" ]; then
    BUCKET_CMD="$BUCKET_CMD --bucket-names $STAGE_NAMES"
fi

print_info "Running: $BUCKET_CMD"
eval $BUCKET_CMD

if [ $? -ne 0 ]; then
    print_error "Bucketing failed!"
    exit 1
fi

print_info "✓ Bucketing complete"
echo ""

# Step 2: Preprocess bucketed data
print_step "Preprocessing bucketed data for training..."

PREPROCESS_CMD="python preprocess_bucketed_rollouts.py -i $BUCKET_DIR -o $PREPROCESS_DIR --filter $FILTER_MODE"

print_info "Running: $PREPROCESS_CMD"
eval $PREPROCESS_CMD

if [ $? -ne 0 ]; then
    print_error "Preprocessing failed!"
    exit 1
fi

print_info "✓ Preprocessing complete"
echo ""

# Step 3: Create progressive/combined datasets (optional)
if [ "$COMBINE" = true ]; then
    print_step "Creating progressive training datasets..."

    # Get bucket names
    if [ -n "$STAGE_NAMES" ]; then
        # Use custom stage names
        IFS=' ' read -ra BUCKETS <<< "$STAGE_NAMES"
    else
        # Use default names based on num_stages
        if [ "$NUM_STAGES" -eq 3 ]; then
            BUCKETS=("easy" "medium" "hard")
        elif [ "$NUM_STAGES" -eq 5 ]; then
            BUCKETS=("very_easy" "easy" "medium" "hard" "very_hard")
        else
            # Auto-detect from directory
            BUCKETS=()
            for f in "$BUCKET_DIR"/*.jsonl; do
                if [ -f "$f" ]; then
                    BUCKETS+=($(basename "$f" .jsonl))
                fi
            done
        fi
    fi

    print_info "Creating cumulative datasets for: ${BUCKETS[*]}"

    # Create cumulative datasets
    CUMULATIVE=""
    STAGE=1
    for bucket in "${BUCKETS[@]}"; do
        CUMULATIVE="$CUMULATIVE $bucket"
        OUTPUT_NAME="stage${STAGE}"

        print_info "Creating $OUTPUT_NAME with buckets:$CUMULATIVE"

        python preprocess_bucketed_rollouts.py \
            -i "$BUCKET_DIR" \
            -o "$PROGRESSIVE_DIR" \
            --buckets $CUMULATIVE \
            --combine \
            --output-name "$OUTPUT_NAME" \
            --filter "$FILTER_MODE"

        if [ $? -ne 0 ]; then
            print_error "Failed to create progressive dataset for stage $STAGE"
            exit 1
        fi

        ((STAGE++))
    done

    print_info "✓ Progressive datasets created"
    echo ""
fi

# Print summary
print_step "Pipeline Complete!"
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Input: $ROLLOUT_DIR"
echo "Output: $OUTPUT_BASE"
echo ""
echo "Generated Files:"
echo "  Bucketed data: $BUCKET_DIR"
find "$BUCKET_DIR" -name "*.jsonl" -exec echo "    - {}" \;
echo ""
echo "  Preprocessed data: $PREPROCESS_DIR"
find "$PREPROCESS_DIR" -name "*.parquet" -exec echo "    - {}" \;
echo ""

if [ "$COMBINE" = true ]; then
    echo "  Progressive data: $PROGRESSIVE_DIR"
    find "$PROGRESSIVE_DIR" -name "*.parquet" -exec echo "    - {}" \;
    echo ""
fi

echo "========================================"
echo ""
print_info "Next steps:"
echo "  1. Review the bucketed data to understand difficulty distribution"
echo "  2. Start training with the preprocessed data:"

if [ "$COMBINE" = true ]; then
    echo ""
    echo "     # Progressive curriculum training:"
    for ((i=1; i<=$NUM_STAGES; i++)); do
        echo "     python train.py --data ${PROGRESSIVE_DIR}/stage${i}_train.parquet"
    done
else
    echo ""
    echo "     # Train on individual stages:"
    if [ -n "$STAGE_NAMES" ]; then
        IFS=' ' read -ra BUCKETS <<< "$STAGE_NAMES"
        for bucket in "${BUCKETS[@]}"; do
            echo "     python train.py --data ${PREPROCESS_DIR}/${bucket}_train.parquet"
        done
    else
        for f in "$PREPROCESS_DIR"/*_train.parquet; do
            if [ -f "$f" ]; then
                echo "     python train.py --data $f"
            fi
        done
    fi
fi

echo ""
echo "  3. Evaluate on validation sets:"
for f in "$PREPROCESS_DIR"/*_val.parquet; do
    if [ -f "$f" ]; then
        echo "     python evaluate.py --data $f"
    fi
done

echo ""
print_info "For more information, see CURRICULUM_PIPELINE_GUIDE.md"
echo ""
