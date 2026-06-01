#!/bin/bash

python scripts/grounding_dino_with_segment_anything.py \
    --image_url "assets/door_opening_limiter.jpg" \
    --labels "a door opening limiter." \
    --threshold 0.3 \
    --detector_id "/home/jszn/hewu/model_zoo/grounding-dino-tiny" \
    --segmenter_id "/home/jszn/hewu/model_zoo/sam-vit-base"
