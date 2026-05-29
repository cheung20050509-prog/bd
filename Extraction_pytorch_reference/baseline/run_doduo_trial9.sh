#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" python -u train.py \
  --run_name doduo_pair_trial9_seed42 \
  --loss_type competition \
  --selection_metric score_final \
  --batch_size 24 \
  --gradient_accumulation_steps 2 \
  --lr 5.32e-5 \
  --warmup_ratio 0.062 \
  --max_length 160 \
  --dropout 0.29 \
  --weight_decay 0.041 \
  --tokens_per_column 8 \
  --pair_head interaction \
  --device gpu
