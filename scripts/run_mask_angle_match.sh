#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUERY="${QUERY:-$ROOT/assets/imgs}"
REF="${REF:-$ROOT/assets/imgs/zero.png}"
OUT_DIR="${OUT_DIR:-$ROOT/runtime/mask_angle}"
STEP_DEG="${STEP_DEG:-5.0}"
CANVAS="${CANVAS:-256}"
TARGET_AREA="${TARGET_AREA:-12000.0}"

echo "========================================"
echo " mask_angle_match"
echo " Query:      $QUERY"
echo " Ref:        $REF"
echo " Out dir:    $OUT_DIR"
echo " Step(deg):  $STEP_DEG"
echo " Canvas:     $CANVAS"
echo " Target area:$TARGET_AREA"
echo "========================================"

python zerith/mask_angle_match.py \
  --query "$QUERY" \
  --ref "$REF" \
  --out_dir "$OUT_DIR" \
  --step_deg "$STEP_DEG" \
  --canvas "$CANVAS" \
  --target_area "$TARGET_AREA" \
  "$@"
