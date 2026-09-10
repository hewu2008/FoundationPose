# encoding:utf8
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Optional


class DetectionResult:
    def __init__(self, score: float, label: str, box: List[int], mask: Optional[np.array] = None):
        self.score = score
        self.label = label
        self.box = box
        self.mask = mask


class ZerithSegmentation:
    def __init__(self, detector_id=None, segmenter_id=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_id = detector_id if detector_id else "IDEA-Research/grounding-dino-tiny"
        self.segmenter_id = segmenter_id if segmenter_id else "facebook/sam-vit-base"
        self._init_models()

    def _init_models(self):
        from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

        self.object_detector = pipeline(
            model=self.detector_id,
            task="zero-shot-object-detection",
            device=self.device
        )

        self.segmentator = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.segmenter_id)

    def _refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon = largest_contour.reshape(-1, 2).tolist()
                    new_mask = np.zeros(mask.shape, dtype=np.uint8)
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(new_mask, [pts], color=255)
                    masks[idx] = new_mask

        return masks

    def _detect(self, image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[DetectionResult]:
        labels = [label if label.endswith(".") else label + "." for label in labels]
        results = self.object_detector(image, candidate_labels=labels, threshold=threshold)

        detections = []
        for result in results:
            box = [
                result['box']['xmin'],
                result['box']['ymin'],
                result['box']['xmax'],
                result['box']['ymax']
            ]
            detections.append(DetectionResult(
                score=result['score'],
                label=result['label'],
                box=box
            ))

        return detections

    def _segment(self, image: Image.Image, detections: List[DetectionResult], polygon_refinement: bool = False) -> List[DetectionResult]:
        if not detections:
            return detections

        boxes = [[det.box] for det in detections]

        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)
        outputs = self.segmentator(**inputs)

        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = self._refine_masks(masks, polygon_refinement)

        for detection, mask in zip(detections, masks):
            detection.mask = mask

        return detections

    def segment(self, rgb: np.ndarray, labels: List[str], threshold: float = 0.3) -> Optional[np.ndarray]:
        image = Image.fromarray(rgb.astype(np.uint8))
        detections = self._detect(image, labels, threshold)

        if not detections:
            return None

        detections = self._segment(image, detections, polygon_refinement=True)

        if detections:
            return detections[0].mask

        return None
