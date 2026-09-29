#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"

INPUT_DIR="runtime/debug_zerith_server/rgb"
OUTPUT_DIR="runtime/yolo_detect"

python scripts/detect_yolo_images.py \
    --yolo_weights assets/last_20260807_v0.pt \
    --yolo_confidence 0.5 \
    --yolo_iou 0.7 \
    --parts_config "zerith/parts_config.json" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"