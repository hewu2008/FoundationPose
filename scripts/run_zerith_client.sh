#!/bin/bash

export PYTHONPATH=//home/chery/gzl/jszn/Foundation_file/FoundationPose

DEBUG_DIR="runtime/debug_zerith_client"

rm -rf "$DEBUG_DIR"

python zerith/zerith_client.py \
    --debug_dir "$DEBUG_DIR"
