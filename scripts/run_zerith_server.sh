#!/bin/bash
export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEBUG_DIR="runtime/debug_zerith_server"
LOG_FILE="runtime/zerith_server.log"

rm -rf "$DEBUG_DIR"
mkdir -p "runtime"

# yolo weights
# /home/chery/gzl/jszn/Foundation_file/Demo_Detector/pretrained_weights/last_20260810.pt 

python zerith/zerith_server_main.py \
    --yolo_weights assets/best-v2_20261017.pt \
    --yolo_confidence 0.55 \
    --yolo_iou 0.7 \
    --yolo_min_area_ratio 0.0114 \
    --parts_config "zerith/parts_config.json" \
    --debug_dir "$DEBUG_DIR" \
    --log_file "$LOG_FILE" \
    --zmq_port 5555
