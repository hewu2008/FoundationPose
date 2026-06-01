#!/bin/bash

export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose

python zerith/zerith_foundation_pose_server.py \
    --detector_id "/home/jszn/hewu/model_zoo/grounding-dino-tiny" \
    --segmenter_id "/home/jszn/hewu/model_zoo/sam-vit-base" \
    --mesh_file "demo_data/mustard0/mesh/textured_simple.obj" \
    --test_scene_dir "demo_data/mustard0" \
    --debug_dir "debug_mard0" \
    --zmq_port 5555 
        