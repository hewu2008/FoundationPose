# encoding:utf8
"""Shared mesh-loading helpers for the Zerith server and client."""

from pathlib import Path

import numpy as np
import trimesh


def blue_z_plane_angle_deg(K, pose, to_origin=None, axis_length=0.1):
    """Return the projected local ``+Z`` axis angle in degrees.

    The axis is projected in the same way as ``Utils.draw_xyz_axis``.  The
    returned convention is the one used by the grasp client: the original
    upward direction (90 degrees in a mathematical image-plane convention)
    is zero, clockwise is positive, counter-clockwise is negative, and the
    The directed angle is first normalized to ``[-180, 180)`` and then
    folded by 180 degrees: values above 90 degrees have 180 degrees
    subtracted and values below -90 degrees have 180 degrees added.  The
    returned result is therefore in ``[-90, 90]``.

    ``pose`` maps the CAD/object frame into the camera frame.  When
    ``to_origin`` is supplied, the pose is first changed to the oriented
    bounding-box center frame, matching the pose used for ``track_vis``.
    """
    if hasattr(pose, "detach"):
        pose = pose.detach().cpu().numpy()
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if to_origin is not None:
        if hasattr(to_origin, "detach"):
            to_origin = to_origin.detach().cpu().numpy()
        to_origin = np.asarray(to_origin, dtype=np.float64).reshape(4, 4)
        pose = pose @ np.linalg.inv(to_origin)

    origin = pose[:3, 3]
    z_endpoint = origin + pose[:3, :3] @ np.array(
        [0.0, 0.0, float(axis_length)], dtype=np.float64
    )

    def project(point):
        homogeneous = K @ point
        if homogeneous[2] <= 1e-9:
            raise ValueError("Cannot project +Z axis: point is behind camera")
        return homogeneous[:2] / homogeneous[2]

    origin_uv = project(origin)
    endpoint_uv = project(z_endpoint)
    delta_uv = endpoint_uv - origin_uv
    if np.linalg.norm(delta_uv) <= 1e-12:
        raise ValueError("Projected +Z axis has zero image-plane length")

    # Image y points down, so atan2(dv, du) is clockwise-positive from the
    # rightward direction.  Add 90 degrees to make the upward direction zero.
    image_angle = np.degrees(np.arctan2(delta_uv[1], delta_uv[0]))
    angle = (image_angle + 90.0 + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return float(angle)


def load_trimesh(mesh_file):
    """Load every geometry in *mesh_file* as one validated ``Trimesh``.

    OBJ files containing more than one ``o``/``g`` section are returned as a
    ``Scene`` by some trimesh versions. FoundationPose requires a single mesh,
    so forcing/concatenating here prevents category-dependent reset failures.
    """
    path = Path(mesh_file).expanduser().resolve()
    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(
            f"Expected a Trimesh from {path}, got {type(loaded).__name__}"
        )
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh contains no usable triangles: {path}")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f"Mesh contains non-finite vertices: {path}")
    return loaded


def mask_xyxy(mask):
    """Return the tight ``[x1, y1, x2, y2]`` box of a binary mask."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64
    )


def box_iou_xyxy(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ix1, iy1 = np.maximum(a[:2], b[:2])
    ix2, iy2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def project_bounds_xyxy(pose, K, mesh_bounds, image_shape):
    """Project the eight corners of centered mesh bounds into an image box."""
    bounds = np.asarray(mesh_bounds, dtype=np.float64).reshape(2, 3)
    corners = np.asarray(
        [
            [x, y, z]
            for x in bounds[:, 0]
            for y in bounds[:, 1]
            for z in bounds[:, 2]
        ],
        dtype=np.float64,
    )
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    points = corners @ pose[:3, :3].T + pose[:3, 3]
    if np.any(points[:, 2] <= 1e-4):
        return None
    pixels_h = points @ np.asarray(K, dtype=np.float64).reshape(3, 3).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    height, width = image_shape[:2]
    box = np.asarray(
        [
            np.clip(pixels[:, 0].min(), 0, width),
            np.clip(pixels[:, 1].min(), 0, height),
            np.clip(pixels[:, 0].max(), 0, width),
            np.clip(pixels[:, 1].max(), 0, height),
        ],
        dtype=np.float64,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def select_pose_by_mask_bbox(
    poses,
    scores,
    K,
    mesh_bounds,
    mask,
    top_k=80,
    mask_weight=0.85,
    min_iou_gain=0.10,
):
    """Select a high-scoring pose whose projected bounds agree with the mask.

    Returns ``(index, ious, combined_scores)``. Candidate zero is retained
    unless another candidate improves projected-box IoU by ``min_iou_gain``;
    this keeps the FoundationPose scorer authoritative in ambiguous cases.
    """
    poses = np.asarray(poses, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = min(len(poses), len(scores), max(1, int(top_k)))
    target_box = mask_xyxy(mask)
    if count == 0 or target_box is None:
        return 0, np.zeros(0), np.zeros(0)

    ious = np.zeros(count, dtype=np.float64)
    for index in range(count):
        projected = project_bounds_xyxy(
            poses[index], K, mesh_bounds, np.asarray(mask).shape
        )
        if projected is not None:
            ious[index] = box_iou_xyxy(projected, target_box)

    score_slice = scores[:count]
    score_span = float(score_slice.max() - score_slice.min())
    if score_span > 1e-12:
        normalized_scores = (score_slice - score_slice.min()) / score_span
    else:
        normalized_scores = np.ones(count, dtype=np.float64)
    weight = float(np.clip(mask_weight, 0.0, 1.0))
    combined = weight * ious + (1.0 - weight) * normalized_scores
    selected = int(np.argmax(combined))
    if ious[selected] < ious[0] + max(0.0, float(min_iou_gain)):
        selected = 0
    return selected, ious, combined
