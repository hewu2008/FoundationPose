#!/bin/bash

export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose

python scripts/zerith_foundation_pose_server.py \
    --detector_id "/home/jszn/hewu/model_zoo/grounding-dino-tiny" \
    --segmenter_id "/home/jszn/hewu/model_zoo/sam-vit-base" \
    --mesh_file "assets/DPPUB-204001196-AAX_01_01.obj" \
    --test_scene_dir "demo_data/limiter" \
    --debug_dir "debug_limiter" \
    --zmq_port 5555