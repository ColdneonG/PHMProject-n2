#!/bin/bash
set -e

cd "$(dirname "$0")/src"

python train.py train_eval \
  --root ../data \
  --out_dir ../outputs \
  --model phm_project_n2 \
  --global_branch avg \
  --epochs 50 \
  --patience 0 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  --amp \
  --seed 67 \
  --phm_n 2 \
  --rates 12 24 36 \
  --lr 3e-4 \
  --weight_decay 1e-4
