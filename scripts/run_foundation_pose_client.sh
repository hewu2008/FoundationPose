#!/bin/bash

export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose

python zerith/zerith_foundation_pose_client.py \
    --video_dir "demo_data/mustard0" \
    --debug_dir "client_debug" \
    --labels "mustard"