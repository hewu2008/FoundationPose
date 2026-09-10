#!/usr/bin/env python3
# encoding: utf-8
"""Compare mask/depth point-cloud XYZ with a saved FoundationPose pose.

This script does not run FoundationPose.  It uses only a saved segmentation
mask, its aligned depth image, and camera intrinsics to build the visible
surface point cloud.  The resulting robust point-cloud center is compared
against both reference points available from a saved FoundationPose pose:

* the original OBJ/CAD origin (``pose[:3, 3]``), and
* the oriented bounding-box (OBB) center used by ``track_vis``.

The default arguments target the current ``cat2_0`` offline result.  A mask can
be supplied explicitly; otherwise the script matches the client pose against
the server's numbered ``ob_in_cam`` files and uses the mask with the same
number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh


DEFAULT_K = np.array(
    [
        [607.62, 0.0, 329.68],
        [0.0, 608.40, 243.36],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a segmentation mask and depth point cloud to estimate XYZ, "
            "then compare it with FoundationPose."
        )
    )
    parser.add_argument(
        "--instance-dir",
        type=Path,
        default=Path("runtime/debug_zerith_client/cat2_0"),
        help="Client instance directory containing ob_in_cam/0.txt.",
    )
    parser.add_argument(
        "--pose",
        type=Path,
        default=None,
        help="FoundationPose matrix; defaults to INSTANCE_DIR/ob_in_cam/0.txt.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help=(
            "Binary segmentation mask. If omitted, match the client pose to "
            "SERVER_DEBUG_DIR/ob_in_cam and select the same numbered mask."
        ),
    )
    parser.add_argument(
        "--server-debug-dir",
        type=Path,
        default=Path("runtime/debug_zerith_server"),
        help="Server debug directory containing mask/rgb/ob_in_cam.",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=Path("assets/zerith_depth.npy"),
        help="Depth .npy corresponding to the mask.",
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        default=Path("assets/zerith_rgb.png"),
        help="RGB image used only to verify that the inferred mask is matched.",
    )
    parser.add_argument(
        "--parts-config",
        type=Path,
        default=Path("zerith/parts_config.json"),
        help="Parts configuration used to infer the mesh from cat2_0 -> cat2.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Optional mesh override for the FoundationPose OBB-center comparison.",
    )
    parser.add_argument("--fx", type=float, default=float(DEFAULT_K[0, 0]))
    parser.add_argument("--fy", type=float, default=float(DEFAULT_K[1, 1]))
    parser.add_argument("--cx", type=float, default=float(DEFAULT_K[0, 2]))
    parser.add_argument("--cy", type=float, default=float(DEFAULT_K[1, 2]))
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help=(
            "Multiplier converting depth values to meters. By default, values "
            "with median > 10 are treated as millimeters (0.001), otherwise meters."
        ),
    )
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument(
        "--trim-percent",
        type=float,
        default=2.0,
        help="Discard this percentage from each end of the mask depth distribution.",
    )
    parser.add_argument(
        "--center-method",
        choices=("median", "trimmed_mean", "percentile_bbox"),
        default="median",
        help="Point-cloud XYZ used as the main comparison result.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save all numeric results as JSON.",
    )
    return parser.parse_args()


def load_pose(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"FoundationPose matrix not found: {path}")
    pose = np.asarray(np.loadtxt(path), dtype=np.float64)
    if pose.size != 16:
        raise ValueError(f"Expected a 4x4 pose in {path}, got shape {pose.shape}")
    pose = pose.reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"Pose contains non-finite values: {path}")
    return pose


def numbered_files(directory: Path, suffix: str) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 2**31 - 1, path.name

    return sorted(directory.glob(f"*{suffix}"), key=key)


def infer_mask_from_server(
    pose_path: Path,
    pose: np.ndarray,
    server_debug_dir: Path,
) -> Path:
    """Find the numbered server mask belonging to a client pose.

    Exact/near-exact pose matching is preferred.  The mtime fallback is valid
    for the synchronous REQ/REP client: the server saves the mask before it
    returns the pose, and the next registration cannot begin until then.
    """
    server_pose_dir = server_debug_dir / "ob_in_cam"
    matches: list[Path] = []
    for candidate in numbered_files(server_pose_dir, ".txt"):
        try:
            server_pose = load_pose(candidate)
        except (OSError, ValueError):
            continue
        if np.allclose(server_pose, pose, rtol=1e-5, atol=1e-6):
            matches.append(candidate)

    if matches:
        client_time = pose_path.stat().st_mtime
        matched_pose = min(
            matches,
            key=lambda path: abs(path.stat().st_mtime - client_time),
        )
        mask_path = server_debug_dir / "mask" / f"{matched_pose.stem}.png"
        if mask_path.is_file():
            return mask_path

    mask_dir = server_debug_dir / "mask"
    masks = numbered_files(mask_dir, ".png")
    if not masks:
        raise FileNotFoundError(f"No server masks found in {mask_dir}")

    client_time = pose_path.stat().st_mtime
    preceding = [path for path in masks if path.stat().st_mtime <= client_time]
    if not preceding:
        raise RuntimeError(
            "Could not infer a mask from timestamps. Pass the correct mask with --mask."
        )
    return max(preceding, key=lambda path: path.stat().st_mtime)


def load_depth_meters(path: Path, requested_scale: float | None) -> tuple[np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Depth file not found: {path}")
    depth = np.asarray(np.load(path), dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got {depth.shape} from {path}")
    positive = depth[np.isfinite(depth) & (depth > 0)]
    if positive.size == 0:
        raise ValueError(f"Depth has no positive finite values: {path}")
    scale = requested_scale
    if scale is None:
        scale = 0.001 if float(np.median(positive)) > 10.0 else 1.0
    if scale <= 0:
        raise ValueError("--depth-scale must be positive")
    return depth * float(scale), float(scale)


def load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(f"Mask could not be read: {path}")
    if mask_image.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask_image.shape} does not match depth {expected_shape}"
        )
    mask = mask_image > 0
    if not mask.any():
        raise ValueError(f"Mask is empty: {path}")
    return mask


def mask_depth_to_points(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    min_depth: float,
    max_depth: float,
    trim_percent: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 <= trim_percent < 50.0:
        raise ValueError("--trim-percent must be in [0, 50)")
    valid = (
        mask
        & np.isfinite(depth)
        & (depth >= float(min_depth))
        & (depth <= float(max_depth))
    )
    rows, cols = np.where(valid)
    if len(rows) < 4:
        raise ValueError(f"Only {len(rows)} valid depth pixels remain inside the mask")

    z = depth[rows, cols]
    x = (cols.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (rows.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    points = np.column_stack((x, y, z))

    fraction = trim_percent / 100.0
    z_low, z_high = np.quantile(z, [fraction, 1.0 - fraction])
    keep = (z >= z_low) & (z <= z_high)
    trimmed = points[keep]
    if len(trimmed) < 4:
        raise ValueError("Depth trimming removed too many point-cloud samples")

    quantile_low = np.quantile(trimmed, fraction, axis=0)
    quantile_high = np.quantile(trimmed, 1.0 - fraction, axis=0)
    statistics = {
        "mask_pixels": int(mask.sum()),
        "valid_depth_pixels": int(len(points)),
        "trimmed_points": int(len(trimmed)),
        "depth_trim_range_m": [float(z_low), float(z_high)],
        "median_xyz_m": np.median(trimmed, axis=0),
        "trimmed_mean_xyz_m": np.mean(trimmed, axis=0),
        "percentile_bbox_center_xyz_m": 0.5 * (quantile_low + quantile_high),
        "visible_points_min_xyz_m": np.min(trimmed, axis=0),
        "visible_points_max_xyz_m": np.max(trimmed, axis=0),
    }
    return trimmed, statistics


def infer_category_id(instance_dir: Path) -> str:
    name = instance_dir.name
    prefix, separator, suffix = name.rpartition("_")
    return prefix if separator and suffix.isdigit() else name


def infer_mesh(parts_config: Path, category_id: str) -> Path | None:
    if not parts_config.is_file():
        return None
    with parts_config.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    configs = payload.get("parts", payload) if isinstance(payload, dict) else payload
    if not isinstance(configs, list):
        return None
    for config in configs:
        if str(config.get("id")) != category_id:
            continue
        mesh_path = Path(str(config["mesh_file"])).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = parts_config.parent / mesh_path
        return mesh_path.resolve()
    return None


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"No usable mesh geometry in {mesh_path}")
    return loaded


def verify_mask_rgb(mask_path: Path, rgb_path: Path, server_debug_dir: Path) -> str:
    if not rgb_path.is_file() or not mask_path.stem.isdigit():
        return "not_checked"
    server_rgb_path = server_debug_dir / "rgb" / f"{mask_path.stem}.png"
    if not server_rgb_path.is_file():
        return "not_checked"
    supplied = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    server = cv2.imread(str(server_rgb_path), cv2.IMREAD_COLOR)
    if supplied is None or server is None or supplied.shape != server.shape:
        return "mismatch"
    return "exact_match" if np.array_equal(supplied, server) else "mismatch"


def vector(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


def print_xyz(label: str, xyz: np.ndarray) -> None:
    print(
        f"{label:<36} "
        f"X={xyz[0]: .6f}  Y={xyz[1]: .6f}  Z={xyz[2]: .6f} m"
    )


def main() -> int:
    args = parse_args()
    pose_path = args.pose or args.instance_dir / "ob_in_cam" / "0.txt"
    pose = load_pose(pose_path)
    mask_path = args.mask or infer_mask_from_server(
        pose_path, pose, args.server_debug_dir
    )

    depth, depth_scale = load_depth_meters(args.depth, args.depth_scale)
    mask = load_mask(mask_path, depth.shape)
    K = np.array(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    _, stats = mask_depth_to_points(
        mask,
        depth,
        K,
        args.min_depth,
        args.max_depth,
        args.trim_percent,
    )

    method_key = {
        "median": "median_xyz_m",
        "trimmed_mean": "trimmed_mean_xyz_m",
        "percentile_bbox": "percentile_bbox_center_xyz_m",
    }[args.center_method]
    cloud_xyz = np.asarray(stats[method_key], dtype=np.float64)
    raw_xyz = pose[:3, 3].copy()

    category_id = infer_category_id(args.instance_dir)
    mesh_path = args.mesh.resolve() if args.mesh else infer_mesh(
        args.parts_config, category_id
    )
    obb_xyz = None
    if mesh_path is not None:
        mesh = load_mesh(mesh_path)
        to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
        obb_pose = pose @ np.linalg.inv(to_origin)
        obb_xyz = obb_pose[:3, 3].copy()

    rgb_check = verify_mask_rgb(mask_path, args.rgb, args.server_debug_dir)

    print("\n输入文件")
    print(f"  实例目录              : {args.instance_dir}")
    print(f"  FoundationPose 位姿   : {pose_path}")
    print(f"  分割 Mask             : {mask_path}")
    print(f"  深度图                : {args.depth}")
    print(f"  Mask/RGB 对应检查     : {rgb_check}")
    print(f"  深度换算比例          : {depth_scale:g}")
    print(
        f"  Mask/有效/裁剪后点数  : {stats['mask_pixels']} / "
        f"{stats['valid_depth_pixels']} / {stats['trimmed_points']}"
    )
    print(
        "  保留深度范围          : "
        f"[{stats['depth_trim_range_m'][0]:.6f}, "
        f"{stats['depth_trim_range_m'][1]:.6f}] m"
    )

    print("\nMask 深度点云统计")
    print_xyz("坐标中位数", np.asarray(stats["median_xyz_m"]))
    print_xyz("裁剪后均值", np.asarray(stats["trimmed_mean_xyz_m"]))
    print_xyz("百分位包围盒中心", np.asarray(stats["percentile_bbox_center_xyz_m"]))

    print("\n主要对比")
    print_xyz(f"点云中心 ({args.center_method})", cloud_xyz)
    print_xyz("FoundationPose CAD 原点", raw_xyz)
    print_xyz("差值：点云中心 - CAD 原点", cloud_xyz - raw_xyz)
    print(
        " " * 36
        + "ΔX={: .3f}  ΔY={: .3f}  ΔZ={: .3f} cm".format(
            *((cloud_xyz - raw_xyz) * 100.0)
        )
    )
    if obb_xyz is not None:
        print_xyz("FoundationPose OBB 中心", obb_xyz)
        print_xyz("差值：点云中心 - OBB 中心", cloud_xyz - obb_xyz)
        print(
            " " * 36
            + "ΔX={: .3f}  ΔY={: .3f}  ΔZ={: .3f} cm".format(
                *((cloud_xyz - obb_xyz) * 100.0)
            )
        )

    if rgb_check == "mismatch":
        print(
            "\n警告：Mask 对应的服务端 RGB 与 --rgb 不一致；"
            "请确认 Mask 和深度来自同一帧。"
        )

    if args.output_json is not None:
        result = {
            "instance_dir": str(args.instance_dir),
            "pose_path": str(pose_path),
            "mask_path": str(mask_path),
            "depth_path": str(args.depth),
            "rgb_match": rgb_check,
            "camera_intrinsics": vector(K),
            "depth_scale": depth_scale,
            "center_method": args.center_method,
            "point_cloud": {
                key: vector(value) if isinstance(value, np.ndarray) else value
                for key, value in stats.items()
            },
            "selected_point_cloud_xyz_m": vector(cloud_xyz),
            "foundationpose_cad_origin_xyz_m": vector(raw_xyz),
            "point_cloud_minus_cad_origin_m": vector(cloud_xyz - raw_xyz),
            "mesh_path": str(mesh_path) if mesh_path is not None else None,
            "foundationpose_obb_center_xyz_m": (
                vector(obb_xyz) if obb_xyz is not None else None
            ),
            "point_cloud_minus_obb_center_m": (
                vector(cloud_xyz - obb_xyz) if obb_xyz is not None else None
            ),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\nJSON 结果已保存：{args.output_json}")

    print(
        "\n说明：Mask 点云只包含相机看见的表面，尤其 Z 不等同于完整零件的"
        "体积中心；X/Y 通常更适合用于检查横向系统偏差。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
