#!/bin/bash
# Verify gpt-oss forward/backward correctness on a dequantized bf16 checkpoint.
# Run this once before launch_train_gpt_oss_20b.sh to fail fast on broken weights
# or environment issues (eager attention, dtype mismatches, MoE routing, ...).
set -euxo pipefail

MODEL_DIR=${MODEL_DIR:-$HOME/models/gpt-oss-20b-bf16}
SEQ_LEN=${SEQ_LEN:-64}
BATCH_SIZE=${BATCH_SIZE:-1}
DTYPE=${DTYPE:-bfloat16}

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "[run_check] dequantizing model -> ${MODEL_DIR}"
    python3 "$(dirname "$0")/prepare_model.py" --output-dir "${MODEL_DIR}"
fi

python3 "$(dirname "$0")/check_gpt_oss_fwd_bwd.py" \
    --model-dir "${MODEL_DIR}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    "$@"
