#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUERY="${QUERY:-$ROOT/runtime/debug_zerith_server/mask}"
REF="${REF:-$ROOT/assets/imgs/zero.png}"
REF_FLIP="${REF_FLIP:-$ROOT/assets/imgs/zero_flip.png}"
REF_CENTER="${REF_CENTER:-$ROOT/assets/imgs/zero.txt}"
REF_FLIP_CENTER="${REF_FLIP_CENTER:-$ROOT/assets/imgs/zero_flip.txt}"
OUT_DIR="${OUT_DIR:-$ROOT/runtime/mask_angle}"
STEP_DEG="${STEP_DEG:-5.0}"
CANVAS="${CANVAS:-256}"
TARGET_AREA="${TARGET_AREA:-12000.0}"

echo "========================================"
echo " mask_angle_match"
echo " Query:      $QUERY"
echo " Ref:        $REF"
echo " Ref flip:   $REF_FLIP"
echo " Ref center: $REF_CENTER"
echo " Flip center:$REF_FLIP_CENTER"
echo " Out dir:    $OUT_DIR"
echo " Step(deg):  $STEP_DEG"
echo " Canvas:     $CANVAS"
echo " Target area:$TARGET_AREA"
echo "========================================"

python zerith/mask_angle_match.py \
  --query "$QUERY" \
  --ref "$REF" \
  --ref_flip "$REF_FLIP" \
  --ref_center "$REF_CENTER" \
  --ref_flip_center "$REF_FLIP_CENTER" \
  --out_dir "$OUT_DIR" \
  --step_deg "$STEP_DEG" \
  --canvas "$CANVAS" \
  --target_area "$TARGET_AREA" \
  "$@"
