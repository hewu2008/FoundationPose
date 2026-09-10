#!/usr/bin/env bash
# 四类零件检测启动脚本
# 推荐：per_class + hybrid，结果写到 runtime/locate_anything_v2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-$ROOT/../model/LocateAnything-3B}"
INPUT_DIR="${INPUT_DIR:-$ROOT/gzl_locate_anything}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime/locate_anything_v2}"
MODE="${MODE:-per_class}"                 # per_class | joint
GENERATION_MODE="${GENERATION_MODE:-hybrid}"  # fast | hybrid | slow
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NMS_IOU="${NMS_IOU:-0.50}"
CROSS_NMS_IOU="${CROSS_NMS_IOU:-0.55}"
LIMIT="${LIMIT:-0}"

EXTRA=()
if [[ "${DUMP_RAW:-0}" == "1" ]]; then
  EXTRA+=(--dump_raw)
fi

echo "========================================"
echo " detect_four_parts"
echo " MODE=$MODE  GEN=$GENERATION_MODE"
echo " INPUT=$INPUT_DIR"
echo " OUTPUT=$OUTPUT_DIR"
echo "========================================"

python zerith/detect_four_parts.py \
  --model_path "$MODEL_PATH" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --generation_mode "$GENERATION_MODE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --nms_iou "$NMS_IOU" \
  --cross_nms_iou "$CROSS_NMS_IOU" \
  --limit "$LIMIT" \
  "${EXTRA[@]}" \
  "$@"
