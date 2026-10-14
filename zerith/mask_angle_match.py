"""Estimate an object's in-plane rotation angle by matching its segmentation mask
against a 0-degree reference mask.

Pipeline (as specified):
  1. crop each mask to its foreground bounding box,
  2. resize it to a fixed size on a fixed canvas: the foreground is scaled so
     that its area (and therefore its equivalent diameter) is constant and its
     centroid sits on the canvas centre.  A fixed area is rotation invariant,
     unlike a fixed bounding-box side, so a rotated object keeps the same size
     and the IoU peak stays on the true angle,
  3. rotate the reference mask at a fixed angular step and keep the angle with
     the highest IoU against the query mask,
  4. render a side-by-side visualization of the reference, the query, the best
     rotated overlay and the predicted angle.

Angle convention: the returned ``angle_deg`` is the rotation applied to the
reference mask, in the ``cv2.getRotationMatrix2D`` convention (positive is
counter-clockwise on screen).  0 degrees means "same orientation as the
reference mask" (``/home/jszn/hewu/imgs/zero.png``).

Usage:
    python zerith/mask_angle_match.py \
        --query runtime/debug_zerith_server/mask/1.png \
        --ref /home/jszn/hewu/imgs/zero.png \
        --out_dir runtime/mask_angle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_REF = "assets/zero.png"
DEFAULT_OUT_DIR = "runtime/mask_angle"

CANVAS = 256
TARGET_AREA = 12000.0  # fixed foreground area in pixels after normalization
STEP_DEG = 5.0

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_binary_mask(path) -> np.ndarray:
    """Load an image and return a boolean foreground mask (any non-zero pixel)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    mask = np.asarray(img) > 0
    if not mask.any():
        raise ValueError(f"Mask has no foreground pixels: {path}")
    return mask


def normalize_mask(
    mask: np.ndarray, canvas: int = CANVAS, target_area: float = TARGET_AREA
) -> np.ndarray:
    """Crop to the foreground bbox and place it on a fixed square canvas.

    The foreground is scaled so that its area equals ``target_area`` (a fixed
    equivalent diameter).  Area is invariant under rotation, whereas a fixed
    bounding-box side is not: rotating an elongated mask changes its bbox aspect
    ratio, so scaling by the longest side would change the object size between
    the reference and a rotated query and pull the IoU peak off the true angle.
    The foreground centroid is placed on the canvas centre so that rotating the
    canvas about its centre is the same as rotating the object about its own
    centroid.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot normalize an empty mask")
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = crop.shape
    scale = float(np.sqrt(target_area / max(float(crop.sum()), 1.0)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop * 255, (nw, nh), interpolation=cv2.INTER_LINEAR)
    resized = resized > 127

    rys, rxs = np.nonzero(resized)
    if len(rxs) == 0:
        raise ValueError("Mask vanished after resizing")
    center = (canvas - 1) / 2.0
    oy = int(round(center - rys.mean()))
    ox = int(round(center - rxs.mean()))

    out = np.zeros((canvas, canvas), dtype=bool)
    y0, y1 = max(0, oy), min(canvas, oy + nh)
    x0, x1 = max(0, ox), min(canvas, ox + nw)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Mask fell outside the canvas after normalization")
    sub = resized[y0 - oy:y0 - oy + (y1 - y0), x0 - ox:x0 - ox + (x1 - x0)]
    out[y0:y1, x0:x1] = sub
    if not out.any():
        raise ValueError("Mask fell outside the canvas after normalization")
    return out


def _rotation_matrix(angle_deg: float, canvas: int = CANVAS) -> np.ndarray:
    center = ((canvas - 1) / 2.0, (canvas - 1) / 2.0)
    return cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)


def rotate_mask(mask: np.ndarray, angle_deg: float, canvas: int = CANVAS) -> np.ndarray:
    """Rotate a canvas-sized mask about the canvas center."""
    warped = cv2.warpAffine(
        (mask.astype(np.uint8) * 255),
        _rotation_matrix(angle_deg, canvas=canvas),
        (canvas, canvas),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return warped > 127


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def predict_angle(
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
    step_deg: float = STEP_DEG,
    angle_range=(0.0, 360.0),
    canvas: int = CANVAS,
    target_area: float = TARGET_AREA,
):
    """Return ``(angle_deg, best_iou, angles, ious)``.

    ``angle_deg`` is the rotation applied to the reference that best matches the
    query mask (mod 360).  ``angles``/``ious`` are the raw search curve at the
    requested step, useful for plotting or sanity checks.
    """
    query = normalize_mask(query_mask, canvas=canvas, target_area=target_area)
    ref = normalize_mask(ref_mask, canvas=canvas, target_area=target_area)

    angles = np.arange(angle_range[0], angle_range[1], float(step_deg))
    ious = np.array([mask_iou(rotate_mask(ref, a, canvas=canvas), query) for a in angles])
    best = int(np.argmax(ious))
    return float(angles[best] % 360.0), float(ious[best]), angles, ious


def _axis_vector(mask: np.ndarray) -> np.ndarray:
    """Unit vector of the mask's principal (longest) axis, in image coordinates."""
    ys, xs = np.nonzero(mask)
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()
    mu20 = (x * x).mean()
    mu02 = (y * y).mean()
    mu11 = (x * y).mean()
    angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return np.array([np.cos(angle), np.sin(angle)])


def _draw_axis(panel: np.ndarray, direction: np.ndarray, color) -> None:
    canvas = panel.shape[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return
    direction = np.asarray(direction, dtype=np.float64) / norm
    offset = direction * (canvas * 0.42)
    center = np.array([canvas // 2, canvas // 2], dtype=np.float64)
    p1 = tuple(np.round(center + offset).astype(int))
    p2 = tuple(np.round(center - offset).astype(int))
    cv2.line(panel, p1, p2, color, 1, cv2.LINE_AA)


def _panel(mask: np.ndarray, title: str, axis=None) -> np.ndarray:
    canvas = mask.shape[0]
    panel = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    panel[mask] = (235, 235, 235)
    if axis is not None:
        _draw_axis(panel, axis, (0, 200, 255))
    cv2.putText(panel, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def build_visualization(
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
    angle_deg: float,
    iou: float,
    canvas: int = CANVAS,
    target_area: float = TARGET_AREA,
) -> np.ndarray:
    """Three-panel figure: reference (0 deg), query, and the best overlay."""
    query = normalize_mask(query_mask, canvas=canvas, target_area=target_area)
    ref = normalize_mask(ref_mask, canvas=canvas, target_area=target_area)
    rotated = rotate_mask(ref, angle_deg, canvas=canvas)

    ref_axis = _axis_vector(ref)
    query_axis = _axis_vector(query)
    # rotate the reference long axis exactly like the mask was rotated
    predicted_axis = _rotation_matrix(angle_deg, canvas=canvas)[:, :2] @ ref_axis

    p_ref = _panel(ref, "reference 0deg", ref_axis)
    p_query = _panel(query, "query", query_axis)

    overlay = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    overlay[~(query | rotated)] = (20, 20, 20)
    overlay[query & ~rotated] = (60, 60, 230)   # query only -> red
    overlay[rotated & ~query] = (80, 210, 80)   # ref only   -> green
    overlay[query & rotated] = (60, 230, 230)   # overlap    -> yellow
    _draw_axis(overlay, predicted_axis, (0, 200, 255))
    cv2.putText(
        overlay,
        f"angle={angle_deg:.1f}deg  IoU={iou:.3f}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    pad = np.full((canvas, 8, 3), 40, dtype=np.uint8)
    return np.hstack([p_ref, pad, p_query, pad, overlay])


def _collect_queries(query: Path):
    if query.is_dir():
        return sorted(p for p in query.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [query]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="mask image file or a directory of masks")
    parser.add_argument("--ref", default=DEFAULT_REF, help="0-degree reference mask image")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="directory for visualized results")
    parser.add_argument("--step_deg", type=float, default=STEP_DEG, help="rotation search step in degrees")
    parser.add_argument("--canvas", type=int, default=CANVAS, help="square canvas size after normalization")
    parser.add_argument("--target_area", type=float, default=TARGET_AREA, help="fixed foreground area in pixels after normalization")
    args = parser.parse_args()

    ref_mask = load_binary_mask(Path(args.ref))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for query_path in _collect_queries(Path(args.query)):
        query_mask = load_binary_mask(query_path)
        angle, iou, _, _ = predict_angle(
            query_mask,
            ref_mask,
            step_deg=args.step_deg,
            canvas=args.canvas,
            target_area=args.target_area,
        )
        figure = build_visualization(
            query_mask, ref_mask, angle, iou, canvas=args.canvas, target_area=args.target_area
        )
        out_path = out_dir / f"{query_path.stem}_angle.png"
        cv2.imwrite(str(out_path), figure)
        line = f"{query_path.name}: angle={angle:.1f} deg  IoU={iou:.3f}  -> {out_path}"
        print(line)
        lines.append(line)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
