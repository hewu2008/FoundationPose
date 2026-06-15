#!/bin/bash

python scripts/grounding_dino_with_segment_anything.py \
    --image_url "assets/zerith_rgb.png" \
    --labels "translucent white brake fluid reservoir" \
    --threshold 0.34 \
    --detector_id "/home/jszn/hewu/model_zoo/grounding-dino-tiny" \
    --segmenter_id "/home/jszn/hewu/model_zoo/sam-vit-base"
