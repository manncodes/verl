#!/bin/bash
# End-to-end prompt distillation example: Language Classification
#
# This example demonstrates prompt distillation where a teacher model uses a
# detailed 70-line system prompt for language classification, and a student
# model learns to classify languages WITHOUT the system prompt.
#
# Prerequisites:
#   - A teacher model (e.g., Qwen/Qwen2.5-7B-Instruct)
#   - A student model (can be the same or smaller model)
#   - Input data with text samples to classify
#
# The script runs three stages:
#   1. Generate teacher labels (with system prompt)
#   2. Train student via SFT (without system prompt)
#   3. (Optional) On-policy distillation for further improvement

set -e

# Configuration - modify these for your setup
TEACHER_MODEL=${TEACHER_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
STUDENT_MODEL=${STUDENT_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
INPUT_DATA=${INPUT_DATA:-"~/data/language_classify/input.parquet"}
INPUT_KEY=${INPUT_KEY:-"text"}
OUTPUT_DIR=${OUTPUT_DIR:-"~/data/language_classify/distill"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"checkpoints/language_classify_distill"}
NUM_GPUS=${NUM_GPUS:-4}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_PROMPT_FILE="${SCRIPT_DIR}/system_prompt.txt"

echo "============================================"
echo "  Prompt Distillation: Language Classifier"
echo "============================================"
echo "Teacher model: ${TEACHER_MODEL}"
echo "Student model: ${STUDENT_MODEL}"
echo "System prompt: ${SYSTEM_PROMPT_FILE}"
echo ""

# -----------------------------------------------
# Stage 1: Generate teacher-labeled data
# -----------------------------------------------
echo "[Stage 1] Generating teacher labels with system prompt..."

python -m recipe.prompt_distillation.prepare_data \
    --model_path "${TEACHER_MODEL}" \
    --system_prompt_file "${SYSTEM_PROMPT_FILE}" \
    --input_file "${INPUT_DATA}" \
    --output_file "${OUTPUT_DIR}/train.parquet" \
    --input_key "${INPUT_KEY}" \
    --batch_size 64 \
    --temperature 0.3 \
    --max_new_tokens 10 \
    --tp_size ${NUM_GPUS}

echo "[Stage 1] Done. Teacher-labeled data saved to ${OUTPUT_DIR}/train.parquet"
echo ""

# -----------------------------------------------
# Stage 2: Train student via SFT (off-policy)
# -----------------------------------------------
echo "[Stage 2] Training student model via SFT (no system prompt)..."

bash "${SCRIPT_DIR}/../../run_sft.sh" \
    ${NUM_GPUS} \
    "${OUTPUT_DIR}/train.parquet" \
    "${STUDENT_MODEL}" \
    "${CHECKPOINT_DIR}" \
    data.prompt_key="${INPUT_KEY}" \
    data.response_key=response \
    data.max_length=512 \
    model.lora_rank=32 \
    optim.lr=1e-4 \
    trainer.total_epochs=5 \
    trainer.project_name=language-classify-distill \
    trainer.experiment_name=sft-distill

echo "[Stage 2] Done. Student model saved to ${CHECKPOINT_DIR}"
echo ""

echo "============================================"
echo "  Prompt Distillation Complete!"
echo "============================================"
echo ""
echo "The student model at ${CHECKPOINT_DIR} can now classify"
echo "languages WITHOUT the detailed system prompt."
echo ""
echo "To evaluate, generate predictions with the student model"
echo "and compare accuracy against the teacher (with system prompt)."
