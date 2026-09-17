#!/bin/bash
set -e

cd "$(dirname "$0")/src"

echo "=== Run A: repeat_seed42_a ==="
python train.py train_eval \
  --root ../data \
  --out_dir ../outputs \
  --exp_name repeat_seed42_a \
  --model phm_project_n2 \
  --global_branch avg \
  --epochs 50 \
  --patience 0 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  --amp \
  --seed 42 \
  --phm_n 2 \
  --rates 12 24 36 \
  --lr 3e-4 \
  --weight_decay 1e-4

echo ""
echo "=== Run B: repeat_seed42_b ==="
python train.py train_eval \
  --root ../data \
  --out_dir ../outputs \
  --exp_name repeat_seed42_b \
  --model phm_project_n2 \
  --global_branch avg \
  --epochs 50 \
  --patience 0 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  --amp \
  --seed 42 \
  --phm_n 2 \
  --rates 12 24 36 \
  --lr 3e-4 \
  --weight_decay 1e-4

echo ""
echo "=== Both runs complete ==="
