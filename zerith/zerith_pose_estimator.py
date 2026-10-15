# encoding:utf8
from dataclasses import dataclass
from pathlib import Path

import trimesh
import numpy as np
import logging
from estimater import FoundationPose
from zerith.zerith_mesh_utils import load_trimesh, select_pose_by_mask_bbox


@dataclass
class _ObjectState:
    """Immutable-per-mesh data kept across registration requests."""

    mesh_file: str
    mesh: object
    to_origin: np.ndarray
    extents: np.ndarray
    bbox: np.ndarray
    estimator: FoundationPose


class ZerithPoseEstimator:
    def __init__(self, mesh_file, scorer, refiner, glctx, debug_dir=None):
        self._scorer = scorer
        self._refiner = refiner
        self._glctx = glctx
        self._debug_dir = debug_dir
        self._object_cache = {}
        self._initialized_meshes = set()

        state = self._get_or_create_state(mesh_file)
        self._activate_state(state)
        logging.info("Load estimator successfully")

        self.is_initialized = False

    @staticmethod
    def _mesh_cache_key(mesh_file):
        return str(Path(mesh_file).expanduser().resolve())

    def _create_state(self, mesh_file):
        mesh_file = self._mesh_cache_key(mesh_file)
        mesh = load_trimesh(mesh_file)
        logging.info("Mesh loaded and cached: %s", mesh_file)

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        logging.info("Mesh bbox: %s", bbox)

        estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self._scorer,
            refiner=self._refiner,
            debug_dir=self._debug_dir,
            debug=0,
            glctx=self._glctx,
        )
        return _ObjectState(
            mesh_file=mesh_file,
            mesh=mesh,
            to_origin=to_origin,
            extents=extents,
            bbox=bbox,
            estimator=estimator,
        )

    def _get_or_create_state(self, mesh_file):
        key = self._mesh_cache_key(mesh_file)
        state = self._object_cache.get(key)
        if state is None:
            state = self._create_state(key)
            self._object_cache[key] = state
        return state

    def _activate_state(self, state):
        self.mesh_file = state.mesh_file
        self.mesh = state.mesh
        self.to_origin = state.to_origin
        self.extents = state.extents
        self.bbox = state.bbox
        self.estimator = state.estimator
        self.is_initialized = self.mesh_file in self._initialized_meshes

    def preload_objects(self, mesh_files):
        """Build every configured mesh state before serving the first frame."""
        active_mesh = self.mesh_file
        for mesh_file in dict.fromkeys(mesh_files):
            self._get_or_create_state(mesh_file)
        self._activate_state(self._object_cache[active_mesh])
        logging.info(
            "Preloaded %d FoundationPose mesh states",
            len(self._object_cache),
        )

    @property
    def cached_mesh_files(self):
        return tuple(self._object_cache)

    def reset_object(self, mesh_file):
        key = self._mesh_cache_key(mesh_file)
        if key == self.mesh_file:
            return
        logging.info("Activate cached object %s", key)
        self._activate_state(self._get_or_create_state(key))

    def prepare_frame(self, K, rgb, depth):
        """Preprocess one RGB-D frame once for every object in the batch."""
        return self.estimator.prepare_frame(K=K, rgb=rgb, depth=depth)

    def register(
        self,
        K,
        rgb,
        depth,
        ob_mask=None,
        iteration=5,
        pose_config=None,
        frame_context=None,
    ):
        self.last_rerank = None
        pose = self.estimator.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=ob_mask,
            iteration=iteration,
            frame_context=frame_context,
        )
        rerank = (pose_config or {}).get("pose_mask_bbox_rerank", {})
        if rerank.get("enabled", False):
            candidates = getattr(self.estimator, "poses", None)
            scores = getattr(self.estimator, "scores", None)
            if candidates is not None and scores is not None:
                candidate_np = candidates.detach().cpu().numpy()
                score_np = scores.detach().cpu().numpy()
                selected, ious, combined = select_pose_by_mask_bbox(
                    candidate_np,
                    score_np,
                    K,
                    self.estimator.mesh.bounds,
                    ob_mask,
                    top_k=rerank.get("top_k", 80),
                    mask_weight=rerank.get("mask_weight", 0.85),
                    min_iou_gain=rerank.get("min_iou_gain", 0.10),
                )
                logging.info(
                    "Pose mask-box rerank selected=%d base_iou=%.3f "
                    "selected_iou=%.3f combined=%.3f",
                    selected,
                    ious[0] if len(ious) else 0.0,
                    ious[selected] if len(ious) else 0.0,
                    combined[selected] if len(combined) else 0.0,
                )
                self.last_rerank = {
                    "selected_index": int(selected),
                    "base_iou": float(ious[0]) if len(ious) else 0.0,
                    "selected_iou": (
                        float(ious[selected]) if len(ious) else 0.0
                    ),
                    "combined_score": (
                        float(combined[selected]) if len(combined) else 0.0
                    ),
                }
                if selected != 0:
                    selected_pose = candidates[selected]
                    self.estimator.pose_last = selected_pose
                    pose = (
                        selected_pose @ self.estimator.get_tf_to_centered_mesh()
                    ).detach().cpu().numpy()
        self._initialized_meshes.add(self.mesh_file)
        self.is_initialized = True
        return pose

    def track(self, K, rgb, depth, iteration=2):
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call register first.")
        return self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)
