#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-luna25_organized}"
NORMAL_SIZE="${NORMAL_SIZE:-0}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SAVE_PATH="${SAVE_PATH:-train/models/luna25_2d_dualhead_best.pth}"
LATEST_PATH="${LATEST_PATH:-train/models/luna25_2d_dualhead_latest.pth}"

EXTRA_ARGS=()
if [[ "${NO_PRETRAINED:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no_pretrained)
fi
if [[ "${NO_CUDA:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no_cuda)
fi
if [[ -n "${RESUME_PATH:-}" ]]; then
    EXTRA_ARGS+=(--resume_path "${RESUME_PATH}")
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_2d.py" \
    --data_root "${DATA_ROOT}" \
    --batch_size 4 \
    --steps_per_epoch "${STEPS_PER_EPOCH}" \
    --normal_size "${NORMAL_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --val_ratio "${VAL_RATIO}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --save_path "${SAVE_PATH}" \
    --latest_path "${LATEST_PATH}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
