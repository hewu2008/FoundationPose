#!/bin/bash

export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose

DEBUG_DIR="runtime/debug_zerith_server"

rm -rf "$DEBUG_DIR"

python zerith/zerith_server_main.py \
    --detector_id "/home/jszn/hewu/model_zoo/LocateAnything-3B" \
    --segmenter_id "/home/jszn/hewu/model_zoo/sam-vit-base" \
    --mesh_file "assets/DPPUB-204001196-AAX_01_01.obj" \
    --debug_dir "$DEBUG_DIR" \
    --zmq_port 5555