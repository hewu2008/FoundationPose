"""YOLO instance segmentation copied from Demo_Detector/detector/perception.py.

Keep the source Demo_Detector project read-only. Zerith imports this local copy
when it runs detection and instance segmentation.
"""

import cv2
import numpy as np

from ultralytics import YOLO


class YOLOSegDetector:
    def __init__(self, weight_path, device=0):
        self.model = YOLO(weight_path)
        self.device = device
        self.names = self.model.names

    def predict(self, image, confidence=0.25, iou_thresh=0.7):
        result = self.model.predict(
            image,
            conf=confidence,
            iou=iou_thresh,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        h, w = image.shape[:2]

        if result.boxes is None:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
        else:
            masks = [None] * len(xyxy)

        for i in range(len(xyxy)):
            bbox = xyxy[i].astype(int)

            if masks[i] is not None:
                mask = cv2.resize(
                    masks[i],
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask = (mask > 0.5).astype(np.uint8)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)

            detections.append(
                {
                    "class_id": int(cls_ids[i]),
                    "class_name": self.names[cls_ids[i]],
                    "confidence": float(confs[i]),
                    "bbox": bbox.tolist(),
                    "mask": mask,
                }
            )

        return detections
