#!/bin/bash
# =============================================================================
# 严格补充消融实验：11 个配置 × 3 个随机种子 = 33 次正式运行
#
# 固定协议：256×256、50 epochs、batch size 16、AdamW、CosineAnnealingLR、
# AMP、相同采样策略、相同验证选择和测试评估代码。
#
# 本脚本沿用现有 run_all_models_seed*.sh 的简单执行格式：
# train.py 成功即进入下一项；train.py 返回非零状态时由 set -e 停止。
# 不执行额外的结果数值断言，不跳过任何已有目录，全部任务从头运行。
# =============================================================================

set -e

# HuggingFace 镜像（国内加速）
# Optional: set HF_ENDPOINT yourself if a mirror is required.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/src"

# ---------- 公共参数 ----------
ROOT="../data"
OUT_DIR="../outputs/supplementary_ablation"
EPOCHS=50
PATIENCE=0
BATCH_SIZE=16
NUM_WORKERS=4
LR=3e-4
WEIGHT_DECAY=1e-4
THRESHOLD=0.5
INPUT_SIZE=256
VIDEO_WEIGHT_POWER=0.5
CONDITION_WEIGHT_POWER=0.5
SMALL_WEIGHT=2.0
GLOBAL_BRANCH="avg"
ENCODER_OUTPUT_STRIDE=8
ASPP_OUT_CH=256
DECODER_CH=256
LOW_CH=48
SK_REDUCTION=16
DEVICE="cuda"
AMP="--amp"

run_one () {
  EXP_NAME="$1"
  MODEL_NAME="$2"
  RUN_SEED="$3"
  PHM_ORDER="$4"
  RUN_LABEL="$5"

  echo ""
  echo "============================================"
  echo ">>> ${RUN_LABEL}"
  echo ">>> model=${MODEL_NAME}, seed=${RUN_SEED}, phm_n=${PHM_ORDER}"
  echo ">>> start: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================"

  python train.py train_eval \
    --root "${ROOT}" \
    --out_dir "${OUT_DIR}" \
    --exp_name "${EXP_NAME}" \
    --model "${MODEL_NAME}" \
    --seed "${RUN_SEED}" \
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
    --global_branch "${GLOBAL_BRANCH}" \
    --encoder_output_stride "${ENCODER_OUTPUT_STRIDE}" \
    --aspp_out_ch "${ASPP_OUT_CH}" \
    --decoder_ch "${DECODER_CH}" \
    --low_ch "${LOW_CH}" \
    --rates 12 24 36 \
    --sk_reduction "${SK_REDUCTION}" \
    --phm_n "${PHM_ORDER}" \
    --device "${DEVICE}" \
    --run_script "${SCRIPT_DIR}/run_supplementary_ablation.sh" \
    ${AMP}
}

echo "============================================"
echo "  严格补充消融实验：11 models × 3 seeds"
echo "  共 33 次正式训练，全部从头开始"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  输出目录: ${OUT_DIR}/"
echo "============================================"

# =============================================================================
# Phase 1 — shared A3/A4 anchors（6 runs）
# =============================================================================
for SEED in 42 3407 2026; do
  run_one "supp_anchor_a3" \
    "localsk_hc_global_conv_resnet18_pretrained" \
    "${SEED}" 2 "A3 anchor (${SEED})"

  run_one "supp_anchor_a4" \
    "phm_project_n2" \
    "${SEED}" 2 "A4 anchor (${SEED})"
done

# =============================================================================
# Phase 2 — P0-1 matched A0；P0-2 global-in-SK（6 runs）
# =============================================================================
for SEED in 42 3407 2026; do
  run_one "supp_p0_1_a0" \
    "matched_a0_standard_aspp_resnet18_pretrained" \
    "${SEED}" 2 "P0-1 matched custom A0 (${SEED})"

  run_one "supp_p0_2_ginsk" \
    "localsk_hc_global_insk_conv_resnet18_pretrained" \
    "${SEED}" 2 "P0-2 global-in-SK (${SEED})"
done

# =============================================================================
# Phase 3a — P1-1 local fusion（9 runs）
# =============================================================================
for SEED in 42 3407 2026; do
  run_one "supp_p1_1_concat" \
    "concat_hc_global_conv_resnet18_pretrained" \
    "${SEED}" 2 "P1-1 concat fusion (${SEED})"

  run_one "supp_p1_1_mean" \
    "mean_hc_global_conv_resnet18_pretrained" \
    "${SEED}" 2 "P1-1 mean fusion (${SEED})"

  run_one "supp_p1_1_sum" \
    "sum_hc_global_conv_resnet18_pretrained" \
    "${SEED}" 2 "P1-1 sum fusion (${SEED})"
done

# =============================================================================
# Phase 3b — P1-2 projector comparison（6 runs）
# =============================================================================
for SEED in 42 3407 2026; do
  run_one "supp_p1_2_group2" \
    "localsk_hc_global_group2_resnet18_pretrained" \
    "${SEED}" 2 "P1-2 grouped projection (${SEED})"

  run_one "supp_p1_2_lowrank85" \
    "localsk_hc_global_lowrank85_resnet18_pretrained" \
    "${SEED}" 2 "P1-2 low-rank projection (${SEED})"
done

# =============================================================================
# Phase 3c — P1-3 PHM order（6 runs）
# =============================================================================
for SEED in 42 3407 2026; do
  run_one "supp_p1_3_n1" \
    "phm_project_n1" \
    "${SEED}" 1 "P1-3 PHM n=1 (${SEED})"

  run_one "supp_p1_3_n4" \
    "phm_project_n4" \
    "${SEED}" 4 "P1-3 PHM n=4 (${SEED})"
done

echo ""
echo "============================================"
echo "  全部 33 次补充消融训练完成"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  输出目录: ${OUT_DIR}/"
echo "============================================"
