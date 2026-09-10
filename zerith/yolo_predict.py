#!/usr/bin/env python3
"""Run YOLO26 instance segmentation on the configured image directory."""

from pathlib import Path

from ultralytics import YOLO


# ======================== 运行配置 ========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "assets/img/0810"
OUTPUT_DIR = PROJECT_ROOT / "runtime" / "new_yolo26_test_0810"
MODEL_PATH = Path(
    "/home/chery/gzl/jszn/Foundation_file/Demo_Detector/"
    "pretrained_weights/last_20260810.pt"
)

IMAGE_SIZE = 640
CONFIDENCE = 0.6
IOU_THRESHOLD = 0.7
DEVICE = 0
SAVE_LABELS = True
SAVE_CONFIDENCE = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ===================================================================


def main():
    input_dir = INPUT_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    model_path = MODEL_PATH.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint does not exist: {model_path}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    print(f"Input directory: {input_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Model: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        save=True,
        save_txt=SAVE_LABELS,
        save_conf=SAVE_CONFIDENCE,
        verbose=True,
    )

    detection_count = sum(
        len(result.boxes) if result.boxes is not None else 0
        for result in results
    )
    print(f"Processed inputs: {len(results)}")
    print(f"Detected instances: {detection_count}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
