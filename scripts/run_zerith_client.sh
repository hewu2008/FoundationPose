#!/bin/bash

export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose

python zerith/zerith_client.py \
    --mesh_file "assets/DPPUB-204001196-AAX_01_01.obj" \
    --video_dir "demo_data/mustard0" \
    --debug_dir "debug_bottle" \
    --labels "translucent white brake fluid reservoir"