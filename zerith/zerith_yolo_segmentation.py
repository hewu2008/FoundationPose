# encoding:utf8
"""YOLO26 instance-segmentation backend for Zerith FoundationPose.

Detection boxes and masks come from one YOLO inference.  Masks stay on the
server and are referenced by identifiers in the client registration request,
so FoundationPose receives the exact mask belonging to each detection without
uploading masks or running a second segmentation model.
"""

import gc
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_PARTS_CONFIG = Path(__file__).with_name("parts_config.json")
DEFAULT_YOLO_ROOT = Path(
    "/home/chery/gzl/jszn/Foundation_file/Demo_Detector"
)
DEFAULT_YOLO_WEIGHTS = DEFAULT_YOLO_ROOT / "pretrained_weights" / "last_tmp.pt"


def load_yolo_part_configs(config_path=None):
    """Load category-to-YOLO-class and category-to-CAD mappings."""
    path = Path(config_path or DEFAULT_PARTS_CONFIG).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    configs = payload.get("parts") if isinstance(payload, dict) else payload
    if not isinstance(configs, list) or not configs:
        raise ValueError(
            f"parts config must contain a non-empty 'parts' list: {path}"
        )

    normalized = []
    ids = set()
    detector_classes = set()
    for index, raw in enumerate(configs):
        if not isinstance(raw, dict):
            raise ValueError(f"parts[{index}] must be an object")
        required = {"id", "mesh_file", "yolo_class_name"}
        missing = required - raw.keys()
        if missing:
            raise ValueError(
                f"parts[{index}] missing YOLO fields: {sorted(missing)}"
            )

        config = dict(raw)
        config["id"] = str(config["id"])
        config["yolo_class_name"] = str(config["yolo_class_name"])
        config["label"] = str(
            config.get("label", config["yolo_class_name"])
        )
        if config["id"] in ids:
            raise ValueError(f"duplicate part id: {config['id']}")
        if config["yolo_class_name"] in detector_classes:
            raise ValueError(
                "duplicate yolo_class_name: "
                f"{config['yolo_class_name']}"
            )
        ids.add(config["id"])
        detector_classes.add(config["yolo_class_name"])

        mesh_path = Path(config["mesh_file"]).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        mesh_path = mesh_path.resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"mesh_file for {config['id']} does not exist: {mesh_path}"
            )
        config["mesh_file"] = str(mesh_path)
        normalized.append(config)
    return normalized


class ZerithYoloSegmentation:
    """Adapt Demo_Detector YOLO masks to the existing Zerith server API."""

    def __init__(
        self,
        weight_path=DEFAULT_YOLO_WEIGHTS,
        parts_config=DEFAULT_PARTS_CONFIG,
        confidence=0.6,
        iou_threshold=0.7,
        device=None,
        detector=None,
        keep_detector_resident=True,
    ):
        self.device = device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.keep_detector_resident = bool(keep_detector_resident)
        self.weight_path = str(Path(weight_path).expanduser().resolve())
        if detector is None:
            if not Path(self.weight_path).is_file():
                raise FileNotFoundError(
                    f"YOLO weights do not exist: {self.weight_path}"
                )
            try:
                from yolo26.perception import YOLOSegDetector
            except ModuleNotFoundError as error:
                if error.name == "ultralytics":
                    raise RuntimeError(
                        "Ultralytics is not installed. The YOLO checkpoint "
                        "was created with ultralytics 8.4.104."
                    ) from error
                raise
            detector = YOLOSegDetector(
                self.weight_path,
                device=self._predict_device(self.device),
            )

        self.detector = detector
        self._category_configs = load_yolo_part_configs(parts_config)
        self._config_by_yolo_class = {
            config["yolo_class_name"]: config
            for config in self._category_configs
        }
        self._detection_id = 0
        self._mask_cache = None
        self._detector_active = False
        self._category_order = {
            config["id"]: index
            for index, config in enumerate(self._category_configs)
        }
        self._validate_model_classes()
        if self.keep_detector_resident:
            try:
                self.activate_detector()
            except torch.cuda.OutOfMemoryError:
                logging.warning(
                    "Insufficient VRAM for resident YOLO; falling back to "
                    "per-detection GPU activation"
                )
                self.keep_detector_resident = False
                self.offload_detector()
                gc.collect()
                torch.cuda.empty_cache()
        else:
            
            self.offload_detector()

    @staticmethod
    def _predict_device(device):
        text = str(device)
        if text == "cuda":
            return 0
        if text.startswith("cuda:"):
            return int(text.split(":", 1)[1])
        return text

    def _validate_model_classes(self):
        names = getattr(self.detector, "names", None)
        if names is None:
            return
        available = set(names.values() if isinstance(names, dict) else names)
        missing = sorted(set(self._config_by_yolo_class) - available)
        if missing:
            raise ValueError(
                f"YOLO weights are missing configured classes: {missing}; "
                f"available={sorted(available)}"
            )

    def get_category_config(self, category_id=None, label=None):
        for config in self._category_configs:
            if category_id is not None and config["id"] == str(category_id):
                return config
            if label is not None and (
                config["label"] == str(label)
                or config["yolo_class_name"] == str(label)
            ):
                return config
        return None

    def activate_detector(self):
        if self._detector_active:
            return
        model = getattr(self.detector, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to(self.device)
        self.detector.device = self._predict_device(self.device)
        self._detector_active = True

    def offload_detector(self):
        if not self._detector_active and self.detector.device == "cpu":
            return
        model = getattr(self.detector, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
        self.detector.device = "cpu"
        self._detector_active = False

    def category_sort_key(self, category_id=None, label=None):
        """Return the stable output order declared by parts_config.json."""
        config = self.get_category_config(category_id=category_id, label=label)
        if config is None:
            return len(self._category_order)
        return self._category_order[config["id"]]

    @staticmethod
    def _rgb_digest(rgb):
        contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)
        return hashlib.sha1(contiguous.tobytes()).hexdigest()

    @staticmethod
    def _box_iou(first, second):
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if intersection <= 0:
            return 0.0
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return intersection / (first_area + second_area - intersection)

    @staticmethod
    def _as_rgb_array(image):
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}")
        return rgb

    def detect_part(self, image):
        """Run YOLO once and cache every instance mask on the server."""
        rgb = self._as_rgb_array(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        raw_detections = self.detector.predict(
            bgr,
            confidence=self.confidence,
            iou_thresh=self.iou_threshold,
        )

        detection_id = self._detection_id
        self._detection_id += 1
        height, width = rgb.shape[:2]
        boxes = []
        masks = {}
        records = []

        for raw_index, raw in enumerate(raw_detections):
            class_name = str(raw.get("class_name", ""))
            config = self._config_by_yolo_class.get(class_name)
            if config is None:
                logging.warning(
                    "Ignoring unmapped YOLO class %r", class_name
                )
                continue

            mask = (np.asarray(raw.get("mask")) > 0).astype(np.uint8)
            if mask.shape != (height, width):
                raise ValueError(
                    f"YOLO mask shape {mask.shape} does not match "
                    f"RGB shape {(height, width)}"
                )
            if not mask.any():
                logging.warning(
                    "Ignoring empty YOLO mask for %s", class_name
                )
                continue

            x1, y1, x2, y2 = [int(value) for value in raw["bbox"]]
            x1 = min(max(x1, 0), width)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height)
            y2 = min(max(y2, 0), height)
            if x2 <= x1 or y2 <= y1:
                logging.warning(
                    "Ignoring invalid YOLO box %s for %s",
                    [x1, y1, x2, y2],
                    class_name,
                )
                continue

            mask_id = f"{detection_id}:{raw_index}"
            confidence = float(raw.get("confidence", 0.0))
            box = {
                "label": config["label"],
                "category_id": config["id"],
                "mesh_file": config["mesh_file"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "detector_class_name": class_name,
                "detection_id": detection_id,
                "mask_id": mask_id,
            }
            boxes.append(box)
            masks[mask_id] = mask
            records.append({
                "mask_id": mask_id,
                "category_id": config["id"],
                "box": [x1, y1, x2, y2],
                "confidence": confidence,
            })

        boxes.sort(
            key=lambda box: self.category_sort_key(
                category_id=box["category_id"]
            )
        )
        records.sort(
            key=lambda record: self.category_sort_key(
                category_id=record["category_id"]
            )
        )

        self._mask_cache = {
            "detection_id": detection_id,
            "rgb_digest": self._rgb_digest(rgb),
            "masks": masks,
            "records": records,
        }
        logging.info(
            "YOLO detection %d completed: %d mapped instances",
            detection_id,
            len(boxes),
        )
        return boxes

    def _ensure_matching_cache(self, rgb):
        rgb = self._as_rgb_array(rgb)
        digest = self._rgb_digest(rgb)
        if (
            self._mask_cache is not None
            and self._mask_cache["rgb_digest"] == digest
        ):
            return

        logging.info("YOLO mask cache miss; running one new inference")
        self.activate_detector()
        try:
            self.detect_part(Image.fromarray(rgb))
        finally:
            if not self.keep_detector_resident:
                self.offload_detector()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _mask_for_item(self, item, used_mask_ids):
        cache = self._mask_cache
        supplied_detection_id = item.get("detection_id")
        if (
            supplied_detection_id is not None
            and int(supplied_detection_id) != cache["detection_id"]
        ):
            raise ValueError(
                f"Stale detection_id {supplied_detection_id}; "
                f"current={cache['detection_id']}"
            )

        mask_id = item.get("mask_id")
        if mask_id is not None:
            mask_id = str(mask_id)
            if mask_id not in cache["masks"]:
                raise ValueError(f"Unknown YOLO mask_id: {mask_id}")
            if mask_id in used_mask_ids:
                raise ValueError(f"YOLO mask_id used more than once: {mask_id}")
            used_mask_ids.add(mask_id)
            return cache["masks"][mask_id].copy()

        # Backward-compatible matching for an older client that only returns
        # category and box. Updated clients always use mask_id.
        category_id = item.get("category_id")
        item_box = item.get("box")
        candidates = [
            record
            for record in cache["records"]
            if record["mask_id"] not in used_mask_ids
            and (
                category_id is None
                or record["category_id"] == str(category_id)
            )
        ]
        if not candidates:
            raise ValueError(
                f"No unused YOLO mask for category_id={category_id!r}"
            )
        if item_box is not None:
            selected = max(
                candidates,
                key=lambda record: self._box_iou(record["box"], item_box),
            )
        else:
            selected = max(
                candidates,
                key=lambda record: record["confidence"],
            )
        used_mask_ids.add(selected["mask_id"])
        return cache["masks"][selected["mask_id"]].copy()

    def segment_many(self, rgb, objects):
        """Return cached YOLO instance masks in client request order."""
        if not objects:
            return []
        self._ensure_matching_cache(rgb)
        used_mask_ids = set()
        return [
            self._mask_for_item(item, used_mask_ids)
            for item in objects
        ]

    def segment(self, rgb, label, box=None, threshold=0.3):
        """Compatibility path for the legacy single-register endpoint."""
        config = self.get_category_config(label=label)
        item = {
            "label": label,
            "category_id": config["id"] if config else None,
            "box": box,
        }
        masks = self.segment_many(rgb, [item])
        return masks[0] if masks else None
