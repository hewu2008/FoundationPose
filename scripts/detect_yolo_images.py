# encoding:utf8
"""Offline YOLO detection over saved debug RGB frames.

Runs the same YOLO26 instance-segmentation model used by the Zerith server on
every image in an input directory (default: the RGB frames dumped by
zerith_server_main.py) and writes annotated images plus a JSON summary into the
runtime directory.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "zerith"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from yolo26.perception import YOLOSegDetector  # noqa: E402
from zerith_yolo_segmentation import (  # noqa: E402
    DEFAULT_PARTS_CONFIG,
    load_yolo_part_configs,
)

DEFAULT_INPUT_DIR = ROOT / "runtime" / "debug_zerith_server" / "rgb"
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "yolo_detect"
DEFAULT_YOLO_WEIGHTS = ROOT / "assets" / "last_20260807_v0.pt"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

# BGR palette, cycled by part order in parts_config.json.
PALETTE = np.array(
    [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on saved debug RGB frames"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory holding the RGB images to detect",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save annotated images and detections.json",
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default=str(DEFAULT_YOLO_WEIGHTS),
        help="YOLO26 instance-segmentation weights",
    )
    parser.add_argument(
        "--yolo_confidence",
        type=float,
        default=0.88,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--yolo_iou", type=float, default=0.7, help="YOLO NMS IoU threshold"
    )
    parser.add_argument(
        "--parts_config",
        type=str,
        default=str(DEFAULT_PARTS_CONFIG),
        help="JSON part and mesh configuration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device; defaults to cuda:0 when available",
    )
    return parser.parse_args()


def predict_device(device):
    text = str(device)
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    return text


def collect_images(input_dir):
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def render(image_bgr, detections, category_colors):
    canvas = image_bgr.copy()
    overlay = canvas.copy()
    for detection in detections:
        color = category_colors[detection["category_id"]]
        overlay[detection["mask"] > 0] = color
    canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    for detection in detections:
        color = category_colors[detection["category_id"]]
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color.tolist(), 2)
        text = f'{detection["label"]} {detection["confidence"]:.2f}'
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_y = y1 - baseline if y1 - text_h - baseline >= 0 else y2 + text_h
        cv2.rectangle(
            canvas,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline),
            color.tolist(),
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main(args):
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir does not exist: {input_dir}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    weight_path = Path(args.yolo_weights).expanduser().resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"yolo weights do not exist: {weight_path}")

    configs = load_yolo_part_configs(args.parts_config)
    config_by_class = {config["yolo_class_name"]: config for config in configs}
    category_colors = {
        config["id"]: PALETTE[index % len(PALETTE)]
        for index, config in enumerate(configs)
    }

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = YOLOSegDetector(str(weight_path), device=predict_device(device))

    images = collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {input_dir}")

    summary = {}
    for image_path in images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[skip] cannot read {image_path}")
            continue

        raw_detections = detector.predict(
            image_bgr,
            confidence=args.yolo_confidence,
            iou_thresh=args.yolo_iou,
        )

        detections = []
        for raw in raw_detections:
            class_name = str(raw.get("class_name", ""))
            config = config_by_class.get(class_name)
            if config is None:
                continue
            mask = (np.asarray(raw["mask"]) > 0).astype(np.uint8)
            if not mask.any():
                continue
            x1, y1, x2, y2 = (int(value) for value in raw["bbox"])
            detections.append(
                {
                    "category_id": config["id"],
                    "label": config["label"],
                    "detector_class_name": class_name,
                    "confidence": float(raw.get("confidence", 0.0)),
                    "bbox": [x1, y1, x2, y2],
                    "mask": mask,
                }
            )
        detections.sort(key=lambda item: item["category_id"])

        annotated = render(image_bgr, detections, category_colors)
        cv2.imwrite(str(annotated_dir / image_path.name), annotated)

        summary[image_path.name] = [
            {
                "category_id": detection["category_id"],
                "label": detection["label"],
                "detector_class_name": detection["detector_class_name"],
                "confidence": round(detection["confidence"], 4),
                "bbox": detection["bbox"],
            }
            for detection in detections
        ]
        print(f"[ok] {image_path.name}: {len(detections)} instance(s)")

    summary_path = output_dir / "detections.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    total = sum(len(items) for items in summary.values())
    print(f"processed {len(summary)} image(s), {total} instance(s) total")
    print(f"annotated images -> {annotated_dir}")
    print(f"detections json  -> {summary_path}")


if __name__ == "__main__":
    main(parse_args())