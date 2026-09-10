#!/usr/bin/env python3
# encoding: utf-8
"""
四类工业零件检测脚本（LocateAnything-3B）

策略（针对 runtime/locate_anything 中常见的误检/漏检）：
  1. 每类单独 ground_multi（比 4 类 joint 更易检出多实例）
  2. 按类 area_range 过滤
  3. 同类 NMS + 跨类 NMS（消掉同一物体被标成 cat1+cat2+cat3）
  4. 白罐几何消歧：大框偏 cat1，小直筒偏 cat2/cat3（辅助，不替代模型）

用法示例：
  conda activate foundationpose
  python zerith/detect_four_parts.py \\
      --model_path /path/to/LocateAnything-3B \\
      --input_dir gzl_locate_anything \\
      --output_dir runtime/locate_anything_v2

  # 只跑前 5 张、打印原始模型输出
  python zerith/detect_four_parts.py ... --limit 5 --dump_raw
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer


# ---------------------------------------------------------------------------
# 类别配置：label = 检测 prompt；area_range = 框面积 / 整图面积
# ---------------------------------------------------------------------------
CATEGORY_CONFIGS = [
    {
        "id": "cat1",
        "name": "brake_wedge",
        "label": (
            "a white translucent plastic brake fluid reservoir with a non-cylindrical "
            "irregular polygonal shape, featuring a black cap and a distinct blue "
            "protective cover on the side outlet"
        ),
        # 真楔形罐通常偏大；过小框多为圆柱罐误检
        "area_range": (0.030, 0.12),
        "color": "red",
        "mesh_file": "assets/category_1.obj",
    },
    {
        "id": "cat2",
        "name": "clutch_waisted",
        "label": (
            "a white clutch fluid reservoir with a white cap, a waisted cylindrical body "
            "(slightly narrower in the middle), and a full circumferential mounting clamp "
            "wrapping the side. A screw may be visible on this clamp"
        ),
        # 收腰罐中等偏小；过大框多为楔形罐误分到 cat2
        "area_range": (0.012, 0.050),
        "color": "lime",
        "mesh_file": "assets/category_2_1.obj",
    },
    {
        "id": "cat3",
        "name": "cylinder_two_ears",
        "label": (
            "a white plastic fluid reservoir with a black cap, a straight cylindrical body "
            "with uniform width from top to bottom, and two circular mounting ears extending "
            "symmetrically outward on each side, with a bottom tube"
        ),
        "area_range": (0.012, 0.050),
        "color": "cyan",
        "mesh_file": "assets/category_3.obj",
    },
    {
        "id": "cat4",
        "name": "yellow_bellows",
        "label": (
            "a yellow silicone component with a tapered shape (narrower at the top and "
            "wider at the bottom), featuring multiple layers of annular folds on the surface"
        ),
        # 黄波纹套可多可小；上限压掉异常大框
        "area_range": (0.006, 0.055),
        "color": "yellow",
        "mesh_file": "assets/category_4.obj",
    },
]


def iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    bb = max(0.0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def box_area_ratio(box: dict, image_area: float) -> float:
    return max(0.0, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])) / max(image_area, 1.0)


def parse_boxes(answer: str, image_width: int, image_height: int) -> list:
    """解析 <ref>...</ref><box><x1><y1><x2><y2></box>，坐标为 [0,1000] 归一化整数。"""
    boxes = []
    current_label = ""
    token_pattern = re.compile(
        r"<ref>(?P<ref>.*?)</ref>"
        r"|<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>",
        flags=re.DOTALL,
    )
    for m in token_pattern.finditer(answer or ""):
        if m.group("ref") is not None:
            current_label = re.sub(r"\s+", " ", m.group("ref")).strip()
            continue
        x1, y1, x2, y2 = [int(m.group(k)) for k in ("x1", "y1", "x2", "y2")]
        if not all(0 <= v <= 1000 for v in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "label": current_label,
                "x1": x1 / 1000.0 * image_width,
                "y1": y1 / 1000.0 * image_height,
                "x2": x2 / 1000.0 * image_width,
                "y2": y2 / 1000.0 * image_height,
            }
        )
    return boxes


def nms_same_class(boxes: list, iou_thresh: float = 0.5) -> list:
    if not boxes:
        return []
    ordered = sorted(
        boxes,
        key=lambda b: (
            b.get("category_id", ""),
            -((b["x2"] - b["x1"]) * (b["y2"] - b["y1"])),
        ),
    )
    suppressed = [False] * len(ordered)
    kept = []
    for i, bi in enumerate(ordered):
        if suppressed[i]:
            continue
        kept.append(bi)
        for j in range(i + 1, len(ordered)):
            if suppressed[j]:
                continue
            bj = ordered[j]
            if bi.get("category_id") != bj.get("category_id"):
                continue
            if iou(bi, bj) >= iou_thresh:
                suppressed[j] = True
    return kept


def white_tank_priority(box: dict) -> float:
    """
    跨类冲突时给白罐打分，分高者保留。
    - 面积大更像 cat1（楔形罐通常更大）
    - cat4 颜色特征强，优先保留
    - cat2/cat3 体积接近，略偏小框
    """
    cid = box.get("category_id", "")
    area = box.get("area_ratio", 0.0)
    w = max(1e-6, box["x2"] - box["x1"])
    h = max(1e-6, box["y2"] - box["y1"])
    aspect = max(w, h) / min(w, h)

    if cid == "cat4":
        return 100.0 + area  # 黄色几乎不与白罐真重叠
    if cid == "cat1":
        # 大面积 + 略扁/不规则 偏楔形
        return 50.0 + area * 200.0 + max(0.0, aspect - 1.2) * 2.0
    if cid == "cat3":
        # 小直筒双耳：面积中小
        return 30.0 + (0.04 - abs(area - 0.025)) * 100.0
    if cid == "cat2":
        return 28.0 + (0.035 - abs(area - 0.028)) * 100.0
    return area


def nms_cross_class(boxes: list, iou_thresh: float = 0.55) -> list:
    """同一物理位置只保留一类（按 white_tank_priority）。"""
    if not boxes:
        return []
    ordered = sorted(boxes, key=white_tank_priority, reverse=True)
    kept = []
    for b in ordered:
        if any(iou(b, k) >= iou_thresh for k in kept):
            continue
        kept.append(b)
    return kept


def filter_by_area(boxes: list, config: dict, image_area: float) -> list:
    lo, hi = config["area_range"]
    out = []
    for box in boxes:
        ratio = box_area_ratio(box, image_area)
        if ratio >= 0.90:  # 整图级异常框
            continue
        if not (lo <= ratio <= hi):
            continue
        item = dict(box)
        item["label"] = config["label"]
        item["category_id"] = config["id"]
        item["category_name"] = config["name"]
        item["area_ratio"] = ratio
        item["mesh_file"] = config.get("mesh_file")
        out.append(item)
    return out


class PartDetector:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(f"Loading model: {model_path}")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, fix_mistral_regex=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.model = (
            AutoModel.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    def _build_inputs(self, image: Image.Image, question: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        return {
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_hws": inputs.get("image_grid_hws", None),
        }

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        do_sample: bool = False,
        verbose: bool = False,
    ) -> str:
        packed = self._build_inputs(image, question)
        gen_kwargs = dict(
            pixel_values=packed["pixel_values"],
            input_ids=packed["input_ids"],
            attention_mask=packed["attention_mask"],
            image_grid_hws=packed["image_grid_hws"],
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=1.1,
            verbose=verbose,
        )
        if do_sample:
            gen_kwargs["top_p"] = 0.9
        response = self.model.generate(**gen_kwargs)
        return response[0] if isinstance(response, tuple) else response

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> str:
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_joint(self, image: Image.Image, phrases: list, **kwargs) -> str:
        cats = "</c>".join(phrases)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def detect_image(
        self,
        image: Image.Image,
        mode: str = "per_class",
        generation_mode: str = "hybrid",
        max_new_tokens: int = 4096,
        nms_iou: float = 0.5,
        cross_nms_iou: float = 0.55,
        dump_raw: bool = False,
        verbose: bool = True,
    ) -> list:
        w, h = image.size
        image_area = float(w * h)
        gen_kw = dict(
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            verbose=False,
        )
        all_boxes: list = []
        t0 = time.perf_counter()

        if mode == "joint":
            labels = [c["label"] for c in CATEGORY_CONFIGS]
            answer = self.detect_joint(image, labels, **gen_kw)
            if dump_raw:
                print("[raw joint]", answer[:1000])
            parsed = parse_boxes(answer, w, h)
            # joint 时尽量用模糊匹配把 ref 映回类别；失败则丢弃
            for box in parsed:
                matched = None
                rl = (box.get("label") or "").lower()
                for cfg in CATEGORY_CONFIGS:
                    if cfg["id"] in rl or cfg["name"] in rl:
                        matched = cfg
                        break
                    # 关键词
                    if "yellow" in rl and "fold" in rl:
                        matched = CATEGORY_CONFIGS[3]
                        break
                    if "polygonal" in rl or "wedge" in rl or "brake" in rl or "blue protective" in rl:
                        matched = CATEGORY_CONFIGS[0]
                        break
                    if "waisted" in rl or "clamp" in rl or "clutch" in rl:
                        matched = CATEGORY_CONFIGS[1]
                        break
                    if "mounting ears" in rl or "two circular" in rl or "straight cylindrical" in rl:
                        matched = CATEGORY_CONFIGS[2]
                        break
                if matched is None:
                    # 按 label 子串
                    for cfg in CATEGORY_CONFIGS:
                        if cfg["label"][:40].lower() in rl or rl in cfg["label"].lower():
                            matched = cfg
                            break
                if matched is None:
                    continue
                forced = filter_by_area([box], matched, image_area)
                for b in forced:
                    b["source"] = "joint"
                all_boxes.extend(forced)
        else:
            # per_class：召回最好
            for cfg in CATEGORY_CONFIGS:
                if verbose:
                    print(f"  -> [{cfg['id']}] {cfg['name']} ...")
                t1 = time.perf_counter()
                answer = self.ground_multi(image, cfg["label"], **gen_kw)
                if dump_raw:
                    print(f"[raw {cfg['id']}]", answer[:800])
                    print("-" * 50)
                parsed = parse_boxes(answer, w, h)
                forced = filter_by_area(parsed, cfg, image_area)
                for b in forced:
                    b["source"] = "per_class"
                if verbose:
                    print(
                        f"     raw={len(parsed)} kept={len(forced)} "
                        f"({time.perf_counter() - t1:.2f}s)"
                    )
                all_boxes.extend(forced)

        all_boxes = nms_same_class(all_boxes, iou_thresh=nms_iou)
        all_boxes = nms_cross_class(all_boxes, iou_thresh=cross_nms_iou)
        all_boxes.sort(key=lambda b: (b.get("category_id", ""), (b["x1"] + b["x2"]) / 2))
        if verbose:
            print(f"  total boxes={len(all_boxes)}  time={time.perf_counter() - t0:.2f}s")
        return all_boxes


def draw_boxes(img: Image.Image, boxes: list) -> Image.Image:
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    id2color = {c["id"]: c["color"] for c in CATEGORY_CONFIGS}
    for idx, b in enumerate(boxes):
        color = id2color.get(b.get("category_id"), "red")
        draw.rectangle((b["x1"], b["y1"], b["x2"], b["y2"]), outline=color, width=3)
        tag = f"[{idx}] {b.get('category_id')} {b.get('area_ratio', 0):.1%}"
        y = max(0, b["y1"] - 14)
        draw.text((b["x1"], y), tag, fill=color)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Four-part LocateAnything detector")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/chery/gzl/jszn/Foundation_file/model/LocateAnything-3B",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/chery/gzl/jszn/Foundation_file/FoundationPose/gzl_locate_anything",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/chery/gzl/jszn/Foundation_file/FoundationPose/runtime/locate_anything_v2",
    )
    parser.add_argument("--mode", choices=["per_class", "joint"], default="per_class")
    parser.add_argument(
        "--generation_mode", choices=["fast", "hybrid", "slow"], default="hybrid"
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--cross_nms_iou", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--dump_raw", action="store_true")
    parser.add_argument("--no_json", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {input_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.name),
    )
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {input_dir}")

    print("=" * 60)
    print("Four-part detection")
    print(f"  mode={args.mode}  gen={args.generation_mode}  n={len(images)}")
    print(f"  input={input_dir}")
    print(f"  output={output_dir}")
    print("=" * 60)
    for c in CATEGORY_CONFIGS:
        print(f"  {c['id']}: area={c['area_range']}  {c['label'][:70]}...")

    detector = PartDetector(args.model_path, device=args.device)

    t_all = time.perf_counter()
    for i, path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {path.name}")
        img = Image.open(path).convert("RGB")
        boxes = detector.detect_image(
            img,
            mode=args.mode,
            generation_mode=args.generation_mode,
            max_new_tokens=args.max_new_tokens,
            nms_iou=args.nms_iou,
            cross_nms_iou=args.cross_nms_iou,
            dump_raw=args.dump_raw,
            verbose=True,
        )
        for j, b in enumerate(boxes):
            print(
                f"    [{j}] {b['category_id']:5s} area={b['area_ratio']:.2%} "
                f"({b['x1']:.0f},{b['y1']:.0f},{b['x2']:.0f},{b['y2']:.0f})"
            )

        vis = draw_boxes(img, boxes)
        out_img = output_dir / f"{path.stem}_boxes{path.suffix}"
        vis.save(out_img)
        print(f"  saved {out_img.name}")

        if not args.no_json:
            payload = {
                "image": str(path),
                "width": img.width,
                "height": img.height,
                "detections": boxes,
            }
            out_json = output_dir / f"{path.stem}_detections.json"
            out_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    elapsed = time.perf_counter() - t_all
    n = len(images)
    print(f"\nDone: {n} images in {elapsed:.1f}s ({elapsed / max(n, 1):.1f}s/img)")
    print(f"Results -> {output_dir}")


if __name__ == "__main__":
    main()
