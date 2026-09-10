import unittest

import numpy as np
import trimesh

from zerith.zerith_mesh_utils import (
    load_trimesh,
    select_pose_by_mask_bbox,
)


class ZerithMeshUtilsTest(unittest.TestCase):
    def test_multi_geometry_cat2_obj_loads_as_one_mesh(self):
        mesh = load_trimesh("assets/category_2_1.obj")
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertGreater(len(mesh.vertices), 100_000)
        np.testing.assert_allclose(
            mesh.extents,
            [0.080234, 0.115, 0.066118],
            atol=1e-6,
        )

    def test_mask_bbox_rerank_rejects_horizontal_long_axis(self):
        K = np.asarray(
            [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]
        )
        bounds = np.asarray([[-1.0, -3.0, -1.0], [1.0, 3.0, 1.0]])
        correct = np.eye(4)
        correct[2, 3] = 10.0
        horizontal = np.eye(4)
        horizontal[:3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        horizontal[2, 3] = 10.0
        mask = np.zeros((200, 200), dtype=bool)
        mask[65:135, 87:113] = True

        selected, ious, _ = select_pose_by_mask_bbox(
            np.stack([horizontal, correct]),
            np.asarray([1.0, 0.9]),
            K,
            bounds,
            mask,
            top_k=2,
            mask_weight=0.85,
            min_iou_gain=0.10,
        )

        self.assertEqual(selected, 1)
        self.assertGreater(ious[1], ious[0] + 0.5)

    def test_mask_rerank_keeps_scorer_when_gain_is_too_small(self):
        pose = np.eye(4)
        pose[2, 3] = 10.0
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        K = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )

        selected, _, _ = select_pose_by_mask_bbox(
            np.stack([pose, pose]),
            np.asarray([1.0, 0.9]),
            K,
            np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            mask,
            top_k=2,
            min_iou_gain=0.10,
        )

        self.assertEqual(selected, 0)


if __name__ == "__main__":
    unittest.main()
