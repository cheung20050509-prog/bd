#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p optuna_output

PYTHON="${PYTHON:-/home/gmn/.conda/envs/bd_2/bin/python}"
ROOT="$(cd .. && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" "$PYTHON" -u optuna_search.py \
  --train_dir "${ROOT}/dataset/Train_Set" \
  --shortcut_name "${ROOT}/../deberta-v3-large" \
  --device gpu \
  --use_amp \
  --n_trials "${N_TRIALS:-30}" \
  --n_startup_trials "${N_STARTUP_TRIALS:-8}" \
  --pruning \
  --storage sqlite:///./optuna_output/doduo_pair_optuna.db \
  --artefact_root ./optuna_output \
  --study_name doduo_pair_deberta_score_final
