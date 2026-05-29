#!/bin/bash

python grounding_dino_with_segment_anything.py \
    --image_url "http://images.cocodataset.org/val2017/000000039769.jpg" \
    --labels "a cat.", "a remote control." \
    --threshold 0.3 \
    --detector_id "/home/jszn/hewu/model_zoo/grounding-dino-tiny" \
    --segmenter_id "/home/jszn/hewu/model_zoo/sam-vit-base"
