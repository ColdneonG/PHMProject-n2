#!/bin/bash
# =============================================================================
# PIDNet-S 单独训练 × seed=3407
#
# 统一条件（与 exo004 一致）：
#   - 相同 train/val/test 划分
#   - 相同输入尺寸 256×256
#   - 相同训练增强（EdgePadding + ValidMask）
#   - 相同采样策略（video_weight=0.5, condition_weight=0.5, small_weight=2.0）
#   - 相同训练轮数 50 epochs（no early stop）
#   - 相同 checkpoint 选择准则（best val selection_score）
#   - 相同测试阈值 0.5
#   - 相同评价代码
#   - 相同硬件
#   - 相同随机种子 3407
# =============================================================================

set -e

cd "$(dirname "$0")/src"

# ---------- 公共参数 ----------
ROOT="../data"
OUT_DIR="../outputs"
SEED=3407
EPOCHS=50
BATCH_SIZE=16
LR=3e-4
WEIGHT_DECAY=1e-4
THRESHOLD=0.5
INPUT_SIZE=256
AMP="--amp"
DEVICE="cuda"
NUM_WORKERS=4
PATIENCE=0

# 采样策略
VIDEO_WEIGHT_POWER=0.5
CONDITION_WEIGHT_POWER=0.5
SMALL_WEIGHT=2.0

echo "============================================"
echo "  PIDNet-S × seed=${SEED}  (exp006)"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exp006" \
  --model pidnet_s \
  --pidnet_pretrained ../checkpoints/PIDNet_S_ImageNet.pth.tar \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --threshold "${THRESHOLD}" \
  --input_size "${INPUT_SIZE}" \
  --video_weight_power "${VIDEO_WEIGHT_POWER}" \
  --condition_weight_power "${CONDITION_WEIGHT_POWER}" \
  --small_weight "${SMALL_WEIGHT}" \
  --device "${DEVICE}" \
  ${AMP}

echo ""
echo "============================================"
echo "  PIDNet-S seed=${SEED} 训练完成！"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
