# encoding:utf8
import gc
import logging

import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Optional

#修改 transformers 4.57.6 中 AutoImageProcessor 存在，AutoProcessor 也存在
from transformers import AutoProcessor
from transformers import AutoModel

# 将 AutoModel 映射为 AutoModelForMaskGeneration（如果代码中使用了这个类）
AutoModelForMaskGeneration = AutoModel

try:
    from .zerith_locate_anything import LocateAnythingWorker
except ImportError:  # Direct execution via zerith/zerith_server_main.py.
    from zerith_locate_anything import LocateAnythingWorker

class DetectionResult:
    def __init__(self, score: float, label: str, box: List[int], mask: Optional[np.array] = None):
        self.score = score
        self.label = label
        self.box = box
        self.mask = mask

class ZerithSegmentation:
    def __init__(self, detector_id: str, segmenter_id: str, parts_config=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.parts_config = parts_config
        self.object_detector = None
        self.segmenter = None
        self._init_models()
        
    def _init_models(self):
        logging.info("Initializing segmentation models...")
        self.object_detector = LocateAnythingWorker(
            self.detector_id, parts_config=self.parts_config
        )
        self.segmenter = AutoModelForMaskGeneration.from_pretrained(
            self.segmenter_id
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.segmenter_id)
        # LocateAnything-3B and FoundationPose cannot both retain enough
        # inference workspace on a 20 GiB GPU. Keep the detector on CPU while
        # the server is idle or running SAM/FoundationPose, and move it back
        # only for a Detection request.
        self.offload_detector()
        logging.info("Segmentation models initialized")

    def activate_detector(self):
        self.object_detector.move_model_to(self.device)

    def offload_detector(self):
        if self.object_detector is None or not str(self.device).startswith("cuda"):
            return
        self.object_detector.move_model_to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _refine_masks(masks: torch.BoolTensor, polygon_refinement=False) -> List[np.ndarray]:
        masks = masks.cpu().float().permute(0, 2, 3, 1).mean(axis=-1)
        masks = list((masks > 0).numpy().astype(np.uint8))
        if polygon_refinement:
            for index, mask in enumerate(masks):
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                refined = np.zeros(mask.shape, dtype=np.uint8)
                polygon = max(contours, key=cv2.contourArea).reshape(-1, 2)
                cv2.fillPoly(refined, [polygon.astype(np.int32)], color=255)
                masks[index] = refined
        return masks

    def _detect(self, image: Image.Image, labels: List[str], threshold=0.3):
        detections = []
        for box in self.object_detector.detect_part(image):
            if box["label"] not in labels:
                continue
            detections.append(DetectionResult(
                score=threshold,
                label=box["label"],
                box=[int(box[k]) for k in ("x1", "y1", "x2", "y2")],
            ))
        return detections

    def _segment(self, image: Image.Image, detections, polygon_refinement=False):
        if not detections:
            return detections
        # One image with N prompt boxes.  The previous N x 1 nesting only
        # happened to be equivalent for a single detection and would make the
        # processor treat a multi-box request as an image batch.
        inputs = self.processor(
            images=image,
            input_boxes=[
                [detection.box for detection in detections]
            ],
            return_tensors="pt",
        ).to(self.device)
        if len(detections) > 1 and hasattr(
            self.segmenter, "get_image_embeddings"
        ):
            # Encode the image once (the expensive SAM ViT stage), but decode
            # each prompt box through the same one-box path as the legacy
            # implementation.  A fully batched prompt decoder changed one
            # boundary pixel in an 8-object regression frame; this cached
            # encoder path was pixel-identical for every mask.
            masks = []
            with torch.inference_mode():
                image_embeddings = self.segmenter.get_image_embeddings(
                    inputs.pixel_values
                )
                for index in range(len(detections)):
                    outputs = self.segmenter(
                        image_embeddings=image_embeddings,
                        input_boxes=inputs.input_boxes[:, index:index + 1],
                    )
                    prompt_masks = self.processor.post_process_masks(
                        masks=outputs.pred_masks,
                        original_sizes=inputs.original_sizes,
                        reshaped_input_sizes=inputs.reshaped_input_sizes,
                    )[0]
                    masks.extend(
                        self._refine_masks(
                            prompt_masks,
                            polygon_refinement,
                        )
                    )
        else:
            with torch.inference_mode():
                outputs = self.segmenter(**inputs)
            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes,
            )[0]
            masks = self._refine_masks(masks, polygon_refinement)
        for detection, mask in zip(detections, masks):
            detection.mask = mask
        return detections

    def segment(self, rgb, label, box=None, threshold=0.3):
        """Return one SAM mask, using a supplied box or detecting by label."""
        image = Image.fromarray(rgb.astype(np.uint8))
        if box is None:
            logging.info("Segmenting %s without a bounding box", label)
            detections = self._detect(image, [label], threshold)
        else:
            logging.info("Segmenting %s with bounding box %s", label, box)
            detections = [DetectionResult(threshold, label, box)]
        detections = self._segment(image, detections, polygon_refinement=True)
        return detections[0].mask if detections else None

    def segment_many(self, rgb, objects):
        """Segment multiple boxes from one RGB image in a single SAM pass.

        ``objects`` preserves request order and each item supplies ``label``,
        ``box`` and optionally ``threshold``.  Detection is intentionally not
        repeated here: the batch-register caller already has LocateAnything
        boxes from the immediately preceding detection request.
        """
        if not objects:
            return []

        image = Image.fromarray(rgb.astype(np.uint8))
        detections = []
        for item in objects:
            box = item.get("box")
            if box is None:
                raise ValueError("Batch segmentation requires a box per object")
            label = item.get("label")
            threshold = float(item.get("threshold", 0.3))
            logging.info("Batch segmenting %s with box %s", label, box)
            detections.append(
                DetectionResult(threshold, label, list(box))
            )

        detections = self._segment(
            image,
            detections,
            polygon_refinement=True,
        )
        return [detection.mask for detection in detections]
