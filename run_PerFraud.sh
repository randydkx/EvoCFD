#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/bin/python3}
DATASET=${DATASET:-PerFraud}
DATASET_TAG=${DATASET_TAG:-${DATASET}}
MODEL=${MODEL:-EvoCFD}
GPU=${GPU:-0,1}
WORKERS=${WORKERS:-0}
SEED=${SEED:-0}
SAMPLE_RATIO=${SAMPLE_RATIO:-1.0}
PRECISION=${PRECISION:-double}
MAX_EPOCH=${MAX_EPOCH:-100}
MODEL_PATH=${MODEL_PATH:-results_model_${DATASET_TAG}_best}
PERSIST_BEST_OR_LAST=${PERSIST_BEST_OR_LAST:-1}
STAGE_LIST=${STAGE_LIST:-"stage_1"}
EXP_PREFIX=${EXP_PREFIX:-${DATASET_TAG}_evocfd}
BASE_LR=${BASE_LR:-0.0001}
HEAD_LR=${HEAD_LR:-0.001}
BN_LR=${BN_LR:-0.00005}
SVD_LR=${SVD_LR:-0.0005}
TIME_LR=${TIME_LR:-0.001}
ETA_MIN=${ETA_MIN:-0.00001}
CLIP_ALPHA=${CLIP_ALPHA:-0.0067}

STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-3000}
STAGE1_BATCH_SIZE_PREV=${STAGE1_BATCH_SIZE_PREV:-3000}
STAGE1_TOKENIZER_LR=${STAGE1_TOKENIZER_LR:-0.0009}

STAGE2_KEEP_RATIO=${STAGE2_KEEP_RATIO:-0.456}
STAGE2_BATCH_SIZE=${STAGE2_BATCH_SIZE:-3000}
STAGE2_BATCH_SIZE_PREV=${STAGE2_BATCH_SIZE_PREV:-3000}
STAGE2_TOKENIZER_LR=${STAGE2_TOKENIZER_LR:-0.003}

STAGE3_KEEP_RATIO=${STAGE3_KEEP_RATIO:-0.318}
STAGE3_BATCH_SIZE=${STAGE3_BATCH_SIZE:-3000}
STAGE3_BATCH_SIZE_PREV=${STAGE3_BATCH_SIZE_PREV:-3000}
STAGE3_TOKENIZER_LR=${STAGE3_TOKENIZER_LR:-0.003}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

IFS=' ' read -r -a STAGES <<< "${STAGE_LIST}"

if [[ ${#STAGES[@]} -eq 0 ]]; then
    exit 1
fi

for stage in "${STAGES[@]}"; do
    keep_ratio=""

    case "${stage}" in
        stage_1)
            batch_size="${STAGE1_BATCH_SIZE}"
            batch_size_prev="${STAGE1_BATCH_SIZE_PREV}"
            tokenizer_lr="${STAGE1_TOKENIZER_LR}"
            exp_name="${EXP_PREFIX}_stage1_best_seed${SEED}"
            ;;
        stage_2)
            keep_ratio="${STAGE2_KEEP_RATIO}"
            batch_size="${STAGE2_BATCH_SIZE}"
            batch_size_prev="${STAGE2_BATCH_SIZE_PREV}"
            tokenizer_lr="${STAGE2_TOKENIZER_LR}"
            exp_name="${EXP_PREFIX}_stage2_best_r0.456_seed${SEED}"
            ;;
        stage_3)
            keep_ratio="${STAGE3_KEEP_RATIO}"
            batch_size="${STAGE3_BATCH_SIZE}"
            batch_size_prev="${STAGE3_BATCH_SIZE_PREV}"
            tokenizer_lr="${STAGE3_TOKENIZER_LR}"
            exp_name="${EXP_PREFIX}_stage3_best_r0.318_seed${SEED}"
            ;;
        *)
            exit 1
            ;;
    esac

    cmd=(
        "${PYTHON_BIN}" run_experiment.py
        --dataset "${DATASET}"
        --model "${MODEL}"
        --stage "${stage}"
        --gpu "${GPU}"
        --workers "${WORKERS}"
        --seed "${SEED}"
        --sample_ratio "${SAMPLE_RATIO}"
        --max_epoch "${MAX_EPOCH}"
        --precision "${PRECISION}"
        --persist_best_or_last "${PERSIST_BEST_OR_LAST}"
        --model_path "${MODEL_PATH}"
        --exp "${exp_name}"
        --apply_max_sv_clip
        --clipping_ratio_alpha "${CLIP_ALPHA}"
        --batch_size "${batch_size}"
        --batch_size_prev "${batch_size_prev}"
        --lr "${BASE_LR}"
        --head_lr "${HEAD_LR}"
        --bn_lr "${BN_LR}"
        --svd_lr "${SVD_LR}"
        --tokenizer_lr "${tokenizer_lr}"
        --time_lr "${TIME_LR}"
        --eta_min "${ETA_MIN}"
    )

    if [[ -n "${keep_ratio}" ]]; then
        cmd+=(--keep_ratio "${keep_ratio}")
    fi

    "${cmd[@]}"
done
