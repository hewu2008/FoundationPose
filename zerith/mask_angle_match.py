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

The same match also carries the part's center point: each reference image comes
with an ``x y`` text file giving a point rigidly attached to the part, and
:func:`predict_center_point` transfers that point onto the query using the
estimated angle, returning its xy coordinates in the query image.

A single reference cannot describe a part that has been flipped over, whose
projected silhouette is mirrored.  Two references are therefore searched: the
nominal 0-degree mask and its flipped counterpart, and the match with the
highest IoU wins.

Angle convention: the returned ``angle_deg`` is the rotation applied to the
winning reference mask, in the ``cv2.getRotationMatrix2D`` convention (positive
is counter-clockwise on screen).  0 degrees means "same orientation as the
reference mask" (``assets/imgs/zero.png``; the flipped match uses
``assets/imgs/zero_flip.png``).

Usage:
    python zerith/mask_angle_match.py \
        --query runtime/debug_zerith_server/mask/1.png \
        --ref assets/imgs/zero.png \
        --ref_flip assets/imgs/zero_flip.png \
        --ref_center assets/imgs/zero.txt \
        --ref_flip_center assets/imgs/zero_flip.txt \
        --ref_depth assets/imgs/zerith_ref_depth.npy \
        --out_dir runtime/mask_angle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_REF = "assets/imgs/zero.png"
DEFAULT_REF_FLIP = "assets/imgs/zero_flip.png"
DEFAULT_REF_CENTER = "assets/imgs/zero.txt"
DEFAULT_REF_FLIP_CENTER = "assets/imgs/zero_flip.txt"
DEFAULT_REF_DEPTH = "assets/imgs/zerith_ref_depth.npy"
DEFAULT_OUT_DIR = "runtime/mask_angle"

# Intrinsics of the reference capture (aligned color/depth of ``rs/cam_high``),
# the same K used across the pipeline (e.g. scripts/zerith_foundation_pose_server.py).
DEFAULT_REF_K = np.array(
    [
        [607.62, 0.0, 329.68],
        [0.0, 608.40, 243.36],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CANVAS = 256
TARGET_AREA = 12000.0  # fixed foreground area in pixels after normalization
STEP_DEG = 5.0

# Side of the square of pixels read around the center point.  A single pixel is
# fragile: when it lands on a depth hole the point is lost, and a lone reading is
# noisy.  A small odd window takes the median of the valid readings instead.
DEPTH_WINDOW = 5

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


def load_center_point(path) -> np.ndarray:
    """Load an ``x y`` text file and return the point as ``np.array([x, y])``.

    The file holds the xy coordinates, in the reference image's own pixel
    frame, of a point rigidly attached to the part (``assets/imgs/zero.txt``
    for ``zero.png``, ``assets/imgs/zero_flip.txt`` for ``zero_flip.png``).
    """
    values = Path(path).read_text(encoding="utf-8").split()
    if len(values) < 2:
        raise ValueError(f"Center-point file must contain two numbers: {path}")
    return np.array([float(values[0]), float(values[1])], dtype=np.float64)


def _foreground_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot take the centroid of an empty mask")
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


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


def predict_angle_multi_ref(
    query_mask: np.ndarray,
    ref_masks,
    step_deg: float = STEP_DEG,
    angle_range=(0.0, 360.0),
    canvas: int = CANVAS,
    target_area: float = TARGET_AREA,
):
    """Search every reference and return the best match.

    Handles flipped parts by trying the nominal reference and its flipped
    counterpart (e.g. ``zero.png`` and ``zero_flip.png``): the query is matched
    against each reference independently and the highest IoU wins.

    Returns ``(angle_deg, best_iou, ref_index, angles, ious)`` where
    ``ref_index`` selects the winning entry of ``ref_masks`` and
    ``angles``/``ious`` are that reference's search curve.
    """
    best = None
    for ref_index, ref_mask in enumerate(ref_masks):
        angle, iou, angles, ious = predict_angle(
            query_mask,
            ref_mask,
            step_deg=step_deg,
            angle_range=angle_range,
            canvas=canvas,
            target_area=target_area,
        )
        if best is None or iou > best[1]:
            best = (angle, iou, ref_index, angles, ious)
    if best is None:
        raise ValueError("No reference masks provided")
    return best


def predict_center_point(
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
    ref_center,
    angle_deg: float,
) -> np.ndarray:
    """Transfer the reference's center point onto the query, returning ``(x, y)``.

    ``ref_center`` is the reference image coordinate from :func:`load_center_point`
    -- a point rigidly attached to the part.  Matching the angle only says how the
    reference must be *rotated*; translating it onto the query needs the centers
    of the two masks.  ``normalize_mask`` already aligns both foreground
    centroids on one canvas, so the mapping boils down to a rotation about the
    centroid plus the area ratio that the normalization removes:

        query_center = query_centroid
                     + sqrt(query_area / ref_area) * R(angle) * (ref_center - ref_centroid)

    The area ratio is the scale factor between the two masks (a fixed area is
    rotation invariant, so it equals their equivalent-diameter ratio); ``R`` is
    the same ``cv2`` rotation applied to the reference, so the offset from the
    centroid turns exactly as the object does.  Because both the reference and
    the query are mapped to a common size, ``canvas`` and ``target_area`` cancel
    out and the result does not depend on them.

    Pass the reference that actually won the match (nominal or flipped) together
    with its own center-point file: a flipped part carries a mirrored anchor, so
    mixing the two references yields a mirrored center.
    """
    query_area = max(float(query_mask.sum()), 1.0)
    ref_area = max(float(ref_mask.sum()), 1.0)
    scale = float(np.sqrt(query_area / ref_area))

    offset = np.asarray(ref_center, dtype=np.float64) - _foreground_centroid(ref_mask)
    offset = _rotation_matrix(angle_deg)[:, :2] @ (offset * scale)
    return _foreground_centroid(query_mask) + offset


def load_depth_map(path) -> np.ndarray:
    """Load a float depth map in meters (``assets/imgs/zerith_ref_depth.npy``)."""
    depth = np.load(str(path)).astype(np.float64)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be a single 2D image: {path}")
    return depth


def depth2xyz(
    depth: np.ndarray, uv, K: np.ndarray = DEFAULT_REF_K, window: int = DEPTH_WINDOW
) -> np.ndarray:
    """Back-project one depth pixel ``uv = (u, v)`` into camera XYZ, in meters.

    ``K`` is the 3x3 pinhole intrinsics of the depth frame.  The pixel is rounded
    the same way ``FoundationPose.Utils.depth2xyzmap`` rounds it.

    ``window`` selects a ``window x window`` square centred on that pixel and the
    depth used is the median of its readings; a reading below 1 mm -- the invalid
    marker used by ``depth2xyzmap`` -- is dropped, so a hole under the centre is
    filled by its neighbours.  ``window=1`` reads the single pixel and returns
    ``NaN`` for an invalid reading; a window with no valid reading at all also
    returns ``NaN``.
    """
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError(f"uv {(u, v)} is outside the depth map of shape {depth.shape}")
    half = max(int(window), 1) // 2
    patch = depth[max(0, v - half):v + half + 1, max(0, u - half):u + half + 1]
    valid = patch[patch >= 0.001]
    if valid.size == 0:
        return np.full(3, np.nan, dtype=np.float64)
    z = float(np.median(valid))
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def predict_center_xyz(
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
    ref_center,
    angle_deg: float,
    ref_depth: np.ndarray,
    K: np.ndarray = DEFAULT_REF_K,
    window: int = DEPTH_WINDOW,
) -> np.ndarray:
    """Return the query center point as camera XYZ, in meters.

    The query center from :func:`predict_center_point` is taken *directly* as the
    ``uv`` of the reference depth map, then back-projected by
    :func:`depth2xyz`.  That assumes the two masks share one camera view, i.e.
    the query pixel coordinates live in the reference frame; if the query was
    captured from another pose the point needs a pose/homography transform
    first.

    Returns ``np.array([X, Y, Z])`` with X right, Y down and Z along the optical
    axis, in the reference camera frame, in **meters** (the overlay and the
    console report convert it to mm).  It is ``NaN`` only when no depth reading
    around the center pixel is valid (hole or occluded).
    """
    center = predict_center_point(query_mask, ref_mask, ref_center, angle_deg)
    return depth2xyz(ref_depth, center, K, window=window)


def to_signed_angle(angle_deg: float) -> float:
    """Convert a cv2 rotation angle to the clockwise-positive signed range.

    ``predict_angle`` returns the angle in ``[0, 360)`` following the
    ``cv2.getRotationMatrix2D`` convention (positive is counter-clockwise on
    screen).  This re-expresses it in the convention used by
    ``zerith_mesh_utils.blue_z_plane_angle_deg``: clockwise is positive and
    counter-clockwise is negative, mapped into ``[-180, 180)``.

    Examples: ``90 -> -90``, ``270 -> 90``, ``200 -> 160``, ``180 -> -180``.
    """
    return (180.0 - float(angle_deg)) % 360.0 - 180.0


def to_undirected_angle(angle_deg: float) -> float:
    """Fold a cv2 rotation angle onto the undirected-line range ``[-90, 90]``.

    A line has no direction: the angles ``a`` and ``a + 180`` describe exactly
    the same undirected line, matching the folding used by
    ``zerith_mesh_utils.blue_z_plane_angle_deg``.  :func:`to_signed_angle` is
    applied first, so the result is also clockwise-positive; values above 90
    degrees have 180 degrees subtracted and values below -90 degrees have 180
    degrees added, giving a unique representative in ``[-90, 90]``.

    Examples: ``100 -> 80``, ``-100 -> -80``, ``180 -> 0``, ``270 -> 90``.
    """
    angle = to_signed_angle(angle_deg)
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


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


def _draw_point(panel: np.ndarray, point, color) -> None:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    cv2.drawMarker(panel, (x, y), color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    cv2.circle(panel, (x, y), 6, color, 1, cv2.LINE_AA)


def _to_canvas_frame(
    point, mask: np.ndarray, canvas: int = CANVAS, target_area: float = TARGET_AREA
) -> np.ndarray:
    """Map an image-space point of ``mask`` onto the normalized canvas frame."""
    scale = float(np.sqrt(target_area / max(float(mask.sum()), 1.0)))
    center = (canvas - 1) / 2.0
    offset = np.asarray(point, dtype=np.float64) - _foreground_centroid(mask)
    return np.array([center, center], dtype=np.float64) + offset * scale


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
    ref_title: str = "reference 0deg",
    center=None,
    xyz=None,
) -> np.ndarray:
    """Three-panel figure: reference (0 deg), query, and the best overlay.

    ``ref_mask`` must be the reference that actually won (nominal or flipped);
    ``ref_title`` labels it in the figure.  ``angle_deg`` stays the cv2
    (counter-clockwise-positive) rotation applied to that reference so the
    overlay is geometrically correct.  Both printed readings use the
    clockwise-positive convention: ``angle`` is the signed, directed angle from
    :func:`to_signed_angle`, and ``line`` is the same orientation folded onto an
    undirected line by :func:`to_undirected_angle`.

    ``center``, when given, is the query center point from
    :func:`predict_center_point` in *query image* pixels; it is mapped onto the
    normalized canvas and marked on the overlay so the transferred anchor can be
    checked against the query silhouette.
    """
    query = normalize_mask(query_mask, canvas=canvas, target_area=target_area)
    ref = normalize_mask(ref_mask, canvas=canvas, target_area=target_area)
    rotated = rotate_mask(ref, angle_deg, canvas=canvas)

    ref_axis = _axis_vector(ref)
    query_axis = _axis_vector(query)
    # rotate the reference long axis exactly like the mask was rotated
    predicted_axis = _rotation_matrix(angle_deg, canvas=canvas)[:, :2] @ ref_axis

    p_ref = _panel(ref, ref_title, ref_axis)
    p_query = _panel(query, "query", query_axis)

    overlay = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    overlay[~(query | rotated)] = (20, 20, 20)
    overlay[query & ~rotated] = (60, 60, 230)   # query only -> red
    overlay[rotated & ~query] = (80, 210, 80)   # ref only   -> green
    overlay[query & rotated] = (60, 230, 230)   # overlap    -> yellow
    _draw_axis(overlay, predicted_axis, (0, 200, 255))
    if center is not None:
        _draw_point(
            overlay,
            _to_canvas_frame(center, query_mask, canvas=canvas, target_area=target_area),
            (255, 0, 255),
        )
    cv2.putText(
        overlay,
        f"angle={to_signed_angle(angle_deg):.1f}deg  IoU={iou:.3f}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"line={to_undirected_angle(angle_deg):.1f}deg",
        (6, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if center is not None:
        cv2.putText(
            overlay,
            f"center=({center[0]:.1f}, {center[1]:.1f})",
            (6, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if xyz is not None:
        cv2.putText(
            overlay,
            f"xyz=({xyz[0] * 1000:.1f}, {xyz[1] * 1000:.1f}, {xyz[2] * 1000:.1f})mm",
            (6, 66),
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
    parser.add_argument("--ref_flip", default=DEFAULT_REF_FLIP, help="flipped reference mask image")
    parser.add_argument("--ref_center", default=DEFAULT_REF_CENTER, help="center-point file for --ref ('x y' in reference pixels)")
    parser.add_argument("--ref_flip_center", default=DEFAULT_REF_FLIP_CENTER, help="center-point file for --ref_flip")
    parser.add_argument("--ref_depth", default=DEFAULT_REF_DEPTH, help="reference depth map (.npy, meters) for the center point's XYZ")
    parser.add_argument("--fx", type=float, default=DEFAULT_REF_K[0, 0], help="reference depth intrinsics focal length fx")
    parser.add_argument("--fy", type=float, default=DEFAULT_REF_K[1, 1], help="reference depth intrinsics focal length fy")
    parser.add_argument("--cx", type=float, default=DEFAULT_REF_K[0, 2], help="reference depth intrinsics principal point cx")
    parser.add_argument("--cy", type=float, default=DEFAULT_REF_K[1, 2], help="reference depth intrinsics principal point cy")
    parser.add_argument("--depth_window", type=int, default=DEPTH_WINDOW, help="side of the pixel window around the center point used for the depth median")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="directory for visualized results")
    parser.add_argument("--step_deg", type=float, default=STEP_DEG, help="rotation search step in degrees")
    parser.add_argument("--canvas", type=int, default=CANVAS, help="square canvas size after normalization")
    parser.add_argument("--target_area", type=float, default=TARGET_AREA, help="fixed foreground area in pixels after normalization")
    args = parser.parse_args()

    ref_masks = [load_binary_mask(Path(args.ref)), load_binary_mask(Path(args.ref_flip))]
    ref_centers = [load_center_point(Path(args.ref_center)), load_center_point(Path(args.ref_flip_center))]
    ref_titles = ["reference 0deg", "reference flip"]
    ref_depth = load_depth_map(Path(args.ref_depth))
    K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for query_path in _collect_queries(Path(args.query)):
        query_mask = load_binary_mask(query_path)
        angle, iou, ref_index, _, _ = predict_angle_multi_ref(
            query_mask,
            ref_masks,
            step_deg=args.step_deg,
            canvas=args.canvas,
            target_area=args.target_area,
        )
        # the winning reference carries both the rotation and its own anchor point
        center = predict_center_point(query_mask, ref_masks[ref_index], ref_centers[ref_index], angle)
        # the center pixel is read straight off the reference depth map
        xyz = depth2xyz(ref_depth, center, K, window=args.depth_window)
        # ``angle`` stays the cv2 (counter-clockwise) rotation used to build the
        # overlay; the reported value is re-expressed clockwise-positive.
        angle_signed = to_signed_angle(angle)
        figure = build_visualization(
            query_mask,
            ref_masks[ref_index],
            angle,
            iou,
            canvas=args.canvas,
            target_area=args.target_area,
            ref_title=ref_titles[ref_index],
            center=center,
            xyz=xyz,
        )
        out_path = out_dir / f"{query_path.stem}_angle.png"
        cv2.imwrite(str(out_path), figure)
        line = (
            f"{query_path.name}: angle={angle_signed:.1f} deg  IoU={iou:.3f}  "
            f"center=({center[0]:.1f}, {center[1]:.1f})  "
            f"xyz=({xyz[0] * 1000:.1f}, {xyz[1] * 1000:.1f}, {xyz[2] * 1000:.1f})mm  "
            f"ref={ref_titles[ref_index]}  -> {out_path}"
        )
        print(line)
        lines.append(line)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
