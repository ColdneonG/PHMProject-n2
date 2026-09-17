#!/bin/bash
# =============================================================================
# 公平对比实验：8 个模型 × seed=3407
#
# 统一条件：
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
#   - CNN 模型统一使用 ImageNet 预训练 ResNet18
# =============================================================================

set -e

# HuggingFace 镜像（国内加速）
# Optional: set HF_ENDPOINT yourself if a mirror is required.

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
echo "  公平对比实验：8 模型 × seed=${SEED}"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# =============================================================================
# 1. U-Net-ResNet18（ImageNet 预训练）
# =============================================================================
echo ""
echo ">>> [1/8] U-Net-ResNet18 (ImageNet pretrained)"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exp004" \
  --model unet_resnet18_pretrained \
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

# =============================================================================
# 2. DeepLabV3+-ResNet18（A1，ImageNet 预训练）
# =============================================================================
echo ""
echo ">>> [2/8] DeepLabV3+-ResNet18 A1 (ImageNet pretrained)"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exo004" \
  --model deeplabv3plus_resnet18_pretrained \
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

# =============================================================================
# 3. SegFormer-B0（官方 HuggingFace 预训练）
# =============================================================================
echo ""
echo ">>> [3/8] SegFormer-B0 (HuggingFace pretrained)"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exo004" \
  --model segformer_b0_pretrained \
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

# =============================================================================
# 4. PSPNet-ResNet18（ImageNet 预训练）
# =============================================================================
echo ""
echo ">>> [4/8] PSPNet-ResNet18 (ImageNet pretrained)"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exo004" \
  --model pspnet_resnet18_pretrained \
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

# =============================================================================
# 5. A2-Conv：全普通卷积 LocalSK（无全局旁路，无投影）
# =============================================================================
echo ""
echo ">>> [5/8] A2-Conv: LocalSK (all Conv, no global, no projection)"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exo004" \
  --model localsk_conv_resnet18_pretrained \
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

# =============================================================================
# 6. A2-HC：超复数 LocalSK（无全局旁路，无投影）
# =============================================================================
echo ""
echo ">>> [6/8] A2-HC: LocalSK (HyperComplex mid/large, no global, no projection)"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exo004" \
  --model localsk_hc_resnet18_pretrained \
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

# =============================================================================
# 7. A3：LocalSK + 全局旁路 + 普通 1×1 Conv 投影
# =============================================================================
echo ""
echo ">>> [7/8] A3: LocalSK + global bypass + Conv1x1 projection"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exo004" \
  --model localsk_hc_global_conv_resnet18_pretrained \
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

# =============================================================================
# 8. A4：LocalSK + 全局旁路 + PHM 投影（原始模型）
# =============================================================================
echo ""
echo ">>> [8/8] A4: LocalSK + global bypass + PHM projection (original)"
python train.py train_eval \
  --root "${ROOT}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "exo004" \
  --model phm_project_n2 \
  --global_branch avg \
  --phm_n 2 \
  --rates 12 24 36 \
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
echo "  全部 8 个模型训练完成！"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  输出目录: ${OUT_DIR}/"
echo "============================================"
