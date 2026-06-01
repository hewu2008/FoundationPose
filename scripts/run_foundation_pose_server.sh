#!/bin/bash

export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose

python zerith/zerith_foundation_pose_server.py \
    --mesh_file "demo_data/mustard0/mesh/textured_simple.obj" \
    --test_scene_dir "demo_data/mustard0" \
    --debug_dir "debug_mard0"
        