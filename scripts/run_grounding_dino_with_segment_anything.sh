#!/bin/bash

python scripts/grounding_dino_with_segment_anything.py \
    --image_url "assets/zerith_rgb.png" \
    --labels "a black part" \
    --threshold 0.3 \
    --detector_id "/home/jszn/hewu/model_zoo/grounding-dino-tiny" \
    --segmenter_id "/home/jszn/hewu/model_zoo/sam-vit-base"
