#!/bin/bash

export PYTHONPATH=/home/jszn/hewu/alg-product/FoundationPose

DEBUG_DIR="runtime/debug_zerith_client"

rm -rf "$DEBUG_DIR"

python zerith/zerith_client.py \
    --mesh_file "assets/DPPUB-204001196-AAX_01_01.obj" \
    --debug_dir "$DEBUG_DIR"