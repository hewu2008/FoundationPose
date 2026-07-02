# encoding:utf8
import trimesh
import numpy as np
import logging
from typing import Optional
from estimater import FoundationPose

class ZerithPoseEstimator:
    def __init__(self, mesh_file, scorer, refiner, glctx, debug_dir=None):
        self.mesh_file = mesh_file
        self.mesh = trimesh.load(self.mesh_file)
        logging.info(f"Mesh loaded: {self.mesh_file}")

        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2, 3)
        logging.info(f"Mesh bbox: {self.bbox}")

        self.estimator = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=0,
            glctx=glctx
        )
        logging.info(f"Load estimator successfully")

        self.is_initialized = False

    def reset_object(self, mesh_file):
        if mesh_file != self.mesh_file:
            logging.info(f"Reset object to {mesh_file}")

            self.mesh_file = mesh_file
            self.mesh = trimesh.load(self.mesh_file)
            logging.info(f"Mesh loaded: {self.mesh_file}")

            self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
            self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2, 3)
            logging.info(f"Mesh bbox: {self.bbox}")

            self.estimator.reset_object(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh
            )

    def register(self, K, rgb, depth, ob_mask=None, iteration=5):
        pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=ob_mask, iteration=iteration)
        self.is_initialized = True
        return pose

    def track(self, K, rgb, depth, iteration=2):
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call register first.")
        return self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)
