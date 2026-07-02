# encoding:utf8 
import re
import os
import torch
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    _category_configs = [
        {
            "label": "white translucent plastic brake fluid reservoir with a blue or black cap",
            "mesh_file": "assets/DPPUB-204001196-AAX_01_01.obj",
            "enabled": True,
            "area_range": (0.020, 0.08),
            "color": "blue"
        },
        {
            "label": "a T-shaped black metal car door checker with a wide top head and a narrow bottom stem",
            "mesh_file": "assets/DPUB-551004004-AAX_04_01.obj",
            "enabled": True,
            "area_range": (0.025, 0.08),
            "color": "red"
        },
        {
            "label": "large black rectangular box or foam block base",
            "enabled": False,
            "area_range": None,
            "color": None
        },
        {
            "label": "black robotic arm or gripper",
            "enabled": False,
            "area_range": None,
            "color": None
        }
    ]
    
    @property
    def optimized_categories(self):
        return [config["label"] for config in self._category_configs]

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",   # "fast" (MTP) | "slow" (NTP/AR) | "hybrid"
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # ---- Convenience methods for each task ----

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)
    
    def _filter_boxes(self, boxes: list[dict], image_area: int) -> list[dict]:
        """Filter boxes based on category config."""
        label_to_config = {config["label"]: config for config in self._category_configs}
        filtered = []
        for box in boxes:
            config = label_to_config.get(box["label"])
            if config is None or not config["enabled"]:
                continue
            box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
            box_area_ratio = box_area / image_area
            min_area, max_area = config["area_range"]
            if min_area <= box_area_ratio <= max_area:
                filtered.append(box)
        return filtered
    
    def detect_part(self, image: Image.Image) -> list[dict]:
        """Part detection."""
        result = self.detect(image, self.optimized_categories)
        w, h = image.size
        image_area = w * h
        boxes = self.parse_boxes(result["answer"], w, h)
        return self._filter_boxes(boxes, image_area)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output ----

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        pattern = r"<ref>([^<]+)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
        for m in re.finditer(pattern, answer):
            label = m.group(1).strip()
            x1, y1, x2, y2 = [int(g) for g in m.groups()[1:]]
            boxes.append({
                "label": label,
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points

def main():
    parser = argparse.ArgumentParser(description='LocateAnything Worker')
    parser.add_argument('--model_path', type=str, default='nvidia/LocateAnything-3B', help='Path to the LocateAnything model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output images')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    worker = LocateAnythingWorker(args.model_path)

    for img_path in input_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            print(f"Processing: {img_path.name}")
            img = Image.open(img_path).convert("RGB")

            filtered_boxes = worker.detect_part(img)
            print("Filtered Boxes:", filtered_boxes)

            draw = ImageDraw.Draw(img)
            w, h = img.size
            image_area = w * h
            label_to_config = {config["label"]: config for config in worker._category_configs}
            for box in filtered_boxes:
                config = label_to_config.get(box["label"])
                box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                box_area_ratio = box_area / image_area
                
                color = config["color"]
                draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline=color, width=2)
                draw.text((box["x1"], box["y1"]), f"{box['label']}\n({box_area_ratio:.1%})", fill=color)
            output_path = output_dir / f"{img_path.stem}_boxes{img_path.suffix}"
            img.save(output_path)
            print(f"Saved: {output_path.name}")
        else:
            print(f"Skipping: {img_path.name}")


if __name__ == "__main__":
    main()