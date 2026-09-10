from io import BytesIO
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from zerith.zerith_client import (
    ZerithFoundationPoseClient,
    build_instance_output_dir,
    process_labels_batch,
    save_registration_result,
)
from zerith.zerith_pose_estimator import ZerithPoseEstimator
from zerith.zerith_server import ZerithServer
from zerith.zerith_yolo_segmentation import ZerithYoloSegmentation


class _FakeDetector:
    def __init__(self):
        self.configs = {
            "cat2": {
                "id": "cat2",
                "label": "part two",
                "mesh_file": "cat2.obj",
            },
            "cat3": {
                "id": "cat3",
                "label": "part three",
                "mesh_file": "cat3.obj",
            },
            "cat4": {
                "id": "cat4",
                "label": "part four",
                "mesh_file": "cat4.obj",
            },
        }

    def get_category_config(self, category_id=None, label=None):
        if category_id in self.configs:
            return self.configs[category_id]
        return next(
            (cfg for cfg in self.configs.values() if cfg["label"] == label),
            None,
        )


class _FakeSegmentation:
    def __init__(self, masks):
        self.object_detector = _FakeDetector()
        self.masks = masks
        self.batch_calls = []
        self.single_calls = []

    def category_sort_key(self, category_id=None, label=None):
        order = {"cat2": 0, "cat3": 1, "cat4": 2}
        config = self.get_category_config(category_id, label)
        return order[config["id"]]

    def get_category_config(self, category_id=None, label=None):
        return self.object_detector.get_category_config(category_id, label)

    def segment_many(self, rgb, objects):
        self.batch_calls.append((rgb.copy(), list(objects)))
        return [mask.copy() for mask in self.masks]

    def segment(self, rgb, label, box, threshold):
        mask = self.masks[len(self.single_calls)].copy()
        self.single_calls.append((rgb.copy(), label, box, threshold))
        return mask


class _FakePoseEstimator:
    def __init__(self):
        self.reset_calls = []
        self.register_calls = []
        self.prepare_calls = []
        self.last_rerank = None
        self.to_origin = np.eye(4, dtype=np.float32)

    def prepare_frame(self, K, rgb, depth):
        context = {"prepared": object()}
        self.prepare_calls.append((K, rgb, depth, context))
        return context

    def reset_object(self, mesh_file):
        self.reset_calls.append(mesh_file)

    def register(
        self,
        K,
        rgb,
        depth,
        ob_mask,
        iteration,
        pose_config,
        frame_context=None,
    ):
        self.register_calls.append(
            {
                "K": K.copy(),
                "rgb": rgb.copy(),
                "depth": depth.copy(),
                "mask": ob_mask.copy(),
                "iteration": iteration,
                "category_id": pose_config["id"],
                "frame_context": frame_context,
            }
        )
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(np.where(ob_mask)[1].mean()) / 100.0
        return pose


class _FakeBatchClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def register_batch(self, K, rgb, depth, objects):
        self.calls.append((K, rgb, depth, objects))
        return self.response


class _FakeYoloModel:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


class _FakeYoloDetector:
    def __init__(self):
        self.names = {
            0: "medium_fluid_reservoir",
            1: "small_fluid_reservoir",
            2: "linkage_cover",
            3: "elbow_pipe",
            4: "interior_door_handle",
        }
        self.model = _FakeYoloModel()
        self.device = "cpu"
        self.calls = []

    def predict(self, image, confidence, iou_thresh):
        self.calls.append((image.copy(), confidence, iou_thresh))
        first = np.zeros(image.shape[:2], dtype=np.uint8)
        first[1:4, 1:5] = 1
        second = np.zeros(image.shape[:2], dtype=np.uint8)
        second[4:7, 3:7] = 1
        # Detector confidence/raw order is intentionally cat3 then cat2. The
        # adapter must expose the configured cat2 -> cat3 category order.
        return [
            {
                "class_id": 1,
                "class_name": "small_fluid_reservoir",
                "confidence": 0.87,
                "bbox": [3, 4, 7, 7],
                "mask": second,
            },
            {
                "class_id": 0,
                "class_name": "medium_fluid_reservoir",
                "confidence": 0.91,
                "bbox": [1, 1, 5, 4],
                "mask": first,
            },
        ]


class ZerithBatchRegistrationTest(unittest.TestCase):
    def test_client_forwards_debug_image_policy(self):
        client = object.__new__(ZerithFoundationPoseClient)
        policy = {
            "enabled": True,
            "scope": "first_input_only",
            "max_sets": 1,
            "image_keys": ["detection_boxes", "all_parts"],
        }
        with patch.object(
            client,
            "send_request",
            return_value={"status": "success"},
        ) as send_request:
            client.register_batch(
                np.eye(3),
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.ones((2, 2), dtype=np.float32),
                [{"category_id": "cat2"}],
                debug_image_policy=policy,
            )

        self.assertEqual(send_request.call_args.args, ("register_batch",))
        self.assertEqual(
            send_request.call_args.kwargs["debug_image_policy"],
            policy,
        )

    def test_server_honors_debug_image_policy_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = ZerithServer(
                _FakePoseEstimator(),
                _FakeSegmentation([]),
                save_dir=tmpdir,
            )
            policy = {
                "enabled": True,
                "scope": "first_input_only",
                "max_sets": 1,
                "image_keys": ["all_parts", "unknown", "all_parts"],
            }

            self.assertEqual(
                server._requested_batch_debug_keys(policy, input_index=0),
                ("all_parts",),
            )
            self.assertEqual(
                server._requested_batch_debug_keys(
                    {**policy, "enabled": False},
                    input_index=0,
                ),
                (),
            )
            self.assertEqual(
                server._requested_batch_debug_keys(policy, input_index=1),
                (),
            )
            server._debug_image_sets_returned = 1
            self.assertEqual(
                server._requested_batch_debug_keys(policy, input_index=0),
                (),
            )

    def test_batch_debug_images_contain_only_requested_encoded_bytes(self):
        detection_png = b"\x89PNG\r\n\x1a\ndetection"
        all_parts_png = b"\x89PNG\r\n\x1a\nparts"
        debug_images = ZerithServer._build_batch_debug_images(
            {
                "detection_id": 7,
                "detection_boxes": {"data": detection_png},
            },
            [{"detection_id": 7}],
            {"data": all_parts_png},
            ("all_parts",),
        )

        self.assertEqual(
            debug_images,
            {"sets": [{"images": {"all_parts": all_parts_png}}]},
        )
        self.assertIsInstance(
            debug_images["sets"][0]["images"]["all_parts"],
            bytes,
        )

    def test_server_category_order_is_cat2_cat3_cat4(self):
        segmentation = _FakeSegmentation([])
        with tempfile.TemporaryDirectory() as tmpdir:
            server = ZerithServer(
                _FakePoseEstimator(), segmentation, save_dir=tmpdir
            )
            shuffled = [
                segmentation.get_category_config(category_id=category_id)
                for category_id in ("cat4", "cat2", "cat3")
            ]
            ordered = sorted(shuffled, key=server._category_sort_key)
            self.assertEqual(
                [config["id"] for config in ordered],
                ["cat2", "cat3", "cat4"],
            )

    def test_client_instance_directory_uses_category_and_instance_index(self):
        self.assertEqual(
            Path(build_instance_output_dir(
                "runtime/debug_zerith_client", "cat2", 3
            )),
            Path("runtime/debug_zerith_client/cat2_3"),
        )

    def test_cat5_draws_obb_box_with_cad_xyz_axes(self):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        to_origin = np.eye(4, dtype=np.float32)
        to_origin[0, 3] = 0.25
        expected_center_pose = pose @ np.linalg.inv(to_origin)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "zerith.zerith_client.draw_posed_3d_box",
            side_effect=lambda K, img, ob_in_cam, bbox: img,
        ) as draw_box, patch(
            "zerith.zerith_client.draw_xyz_axis",
            side_effect=lambda img, ob_in_cam, **kwargs: img,
        ) as draw_axis, patch("zerith.zerith_client.imageio.imwrite"):
            saved = save_registration_result(
                K=np.eye(3),
                color=np.zeros((8, 8, 3), dtype=np.uint8),
                response={
                    "status": "success",
                    "pose": pose,
                    "blue_z_plane_angle_deg": 0.0,
                },
                mesh_bbox=np.ones((2, 3), dtype=np.float32),
                to_origin=to_origin,
                label_output_dir=tmpdir,
                category_id="cat5",
            )

        self.assertTrue(saved)
        np.testing.assert_allclose(draw_box.call_args.args[2], expected_center_pose)
        np.testing.assert_allclose(draw_axis.call_args.args[1], pose)

    def test_non_cat5_keeps_obb_xyz_axes(self):
        pose = np.eye(4, dtype=np.float32)
        to_origin = np.eye(4, dtype=np.float32)
        to_origin[1, 3] = 0.5
        expected_center_pose = pose @ np.linalg.inv(to_origin)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "zerith.zerith_client.draw_posed_3d_box",
            side_effect=lambda K, img, ob_in_cam, bbox: img,
        ), patch(
            "zerith.zerith_client.draw_xyz_axis",
            side_effect=lambda img, ob_in_cam, **kwargs: img,
        ) as draw_axis, patch("zerith.zerith_client.imageio.imwrite"):
            saved = save_registration_result(
                K=np.eye(3),
                color=np.zeros((8, 8, 3), dtype=np.uint8),
                response={
                    "status": "success",
                    "pose": pose,
                    "blue_z_plane_angle_deg": 0.0,
                },
                mesh_bbox=np.ones((2, 3), dtype=np.float32),
                to_origin=to_origin,
                label_output_dir=tmpdir,
                category_id="cat6",
            )

        self.assertTrue(saved)
        np.testing.assert_allclose(
            draw_axis.call_args.args[1], expected_center_pose
        )

    def test_yolo_detects_and_segments_once_with_exact_instance_masks(self):
        detector = _FakeYoloDetector()
        segmentation = ZerithYoloSegmentation(
            parts_config="zerith/parts_config.json",
            detector=detector,
            confidence=0.6,
            iou_threshold=0.7,
            device="cpu",
        )
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]

        boxes = segmentation.detect_part(Image.fromarray(rgb))
        objects = [
            {
                "category_id": box["category_id"],
                "box": [box[key] for key in ("x1", "y1", "x2", "y2")],
                "detection_id": box["detection_id"],
                "mask_id": box["mask_id"],
            }
            for box in boxes
        ]
        masks = segmentation.segment_many(rgb, objects)

        self.assertEqual(len(detector.calls), 1)
        self.assertEqual(detector.calls[0][1:], (0.6, 0.7))
        # Zerith uses RGB; Demo_Detector expects OpenCV BGR.
        self.assertEqual(detector.calls[0][0][0, 0].tolist(), [0, 0, 255])
        self.assertEqual(
            [box["category_id"] for box in boxes],
            ["cat2", "cat3"],
        )
        self.assertEqual(len(masks), 2)
        self.assertEqual(int(masks[0].sum()), 12)
        self.assertEqual(int(masks[1].sum()), 12)
        self.assertFalse(np.array_equal(masks[0], masks[1]))

    def test_server_segments_once_and_returns_configured_category_order(self):
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        depth = np.ones((12, 16), dtype=np.float32)
        K = np.eye(3, dtype=np.float64)
        left = np.zeros((12, 16), dtype=np.uint8)
        left[2:6, 1:5] = 1
        right = np.zeros((12, 16), dtype=np.uint8)
        right[4:10, 9:15] = 1
        # Deliberately supply cat3 before cat2. Masks remain paired with their
        # original request and must follow the object during server sorting.
        segmentation = _FakeSegmentation([right, left])
        estimator = _FakePoseEstimator()
        objects = [
            {
                "category_id": "cat3",
                "label": "part three",
                "box": [9, 4, 15, 10],
                "iteration": 4,
            },
            {
                "category_id": "cat2",
                "label": "part two",
                "box": [1, 2, 5, 6],
                "iteration": 5,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            server = ZerithServer(estimator, segmentation, save_dir=tmpdir)
            with self.assertLogs(level="INFO") as captured_logs:
                response = server._handle_register_batch(
                    {"K": K, "rgb": rgb, "depth": depth, "objects": objects}
                )

            self.assertEqual(response["status"], "success")
            self.assertEqual(response["successful"], 2)
            self.assertEqual(len(segmentation.batch_calls), 1)
            self.assertEqual(
                [call["category_id"] for call in estimator.register_calls],
                ["cat2", "cat3"],
            )
            self.assertTrue(
                np.array_equal(estimator.register_calls[0]["mask"], left > 0)
            )
            self.assertTrue(
                np.array_equal(estimator.register_calls[1]["mask"], right > 0)
            )
            self.assertEqual(len(estimator.prepare_calls), 1)
            self.assertIs(
                estimator.register_calls[0]["frame_context"],
                estimator.prepare_calls[0][3],
            )
            self.assertIs(
                estimator.register_calls[1]["frame_context"],
                estimator.prepare_calls[0][3],
            )
            self.assertEqual(
                [result["request_index"] for result in response["results"]],
                [1, 0],
            )
            self.assertEqual(
                [result["category_id"] for result in response["results"]],
                ["cat2", "cat3"],
            )
            self.assertTrue((Path(tmpdir) / "mask" / "0.png").is_file())
            self.assertTrue((Path(tmpdir) / "mask" / "1.png").is_file())
            for index, expected_mask in enumerate((left > 0, right > 0)):
                depth_image = np.asarray(Image.open(
                    Path(tmpdir) / "depth" / f"{index}.png"
                ))
                self.assertEqual(depth_image.dtype, np.uint16)
                self.assertTrue(np.all(depth_image[expected_mask] == 1000))
                self.assertTrue(np.all(depth_image[~expected_mask] == 0))
            all_parts_overlay = np.asarray(Image.open(
                Path(tmpdir) / "mask_overlay" / "all_parts.png"
            ))
            expected_green = np.array([0, 255, 0], dtype=np.uint8)
            self.assertTrue(np.all(
                all_parts_overlay[left > 0] == expected_green
            ))
            self.assertTrue(np.all(
                all_parts_overlay[right > 0] == expected_green
            ))
            self.assertTrue(np.all(
                all_parts_overlay[~((left > 0) | (right > 0))] == 0
            ))
            self.assertTrue(
                (Path(tmpdir) / "ob_in_cam" / "0.txt").is_file()
            )
            self.assertTrue(
                (Path(tmpdir) / "ob_in_cam" / "1.txt").is_file()
            )
            event_output = "\n".join(captured_logs.output)
            for event_name in (
                "BATCH_START",
                "FRAME_PREPROCESS_DONE",
                "POSE_START",
                "POSE_DONE",
                "POSE_SAVED",
                "BATCH_DONE",
            ):
                self.assertIn(f"EVENT={event_name}", event_output)
            self.assertIn("pose_4x4=[[", event_output)

    def test_all_parts_overlay_accumulates_sequential_single_registers(self):
        rgb = np.full((8, 10, 3), 17, dtype=np.uint8)
        depth = np.ones((8, 10), dtype=np.float32)
        first = np.zeros((8, 10), dtype=np.uint8)
        first[1:4, 1:4] = 1
        second = np.zeros((8, 10), dtype=np.uint8)
        second[4:7, 6:9] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            segmentation = _FakeSegmentation([first, second])
            server = ZerithServer(
                _FakePoseEstimator(),
                segmentation,
                save_dir=tmpdir,
            )
            responses = [
                server._handle_register({
                    "K": np.eye(3),
                    "rgb": rgb,
                    "depth": depth,
                    "category_id": category_id,
                })
                for category_id in ("cat2", "cat3")
            ]
            overlay = np.asarray(Image.open(
                Path(tmpdir) / "mask_overlay" / "all_parts.png"
            ))

        self.assertEqual(
            [response["status"] for response in responses],
            ["success", "success"],
        )
        self.assertEqual(len(segmentation.single_calls), 2)
        union = (first > 0) | (second > 0)
        self.assertTrue(np.all(overlay[union] == [0, 255, 0]))
        self.assertTrue(np.all(overlay[~union] == 17))

    def test_pose_estimator_caches_each_mesh_state_once(self):
        class _Mesh:
            def __init__(self, offset):
                self.vertices = np.array([
                    [offset, 0.0, 0.0],
                    [offset + 1.0, 0.0, 0.0],
                    [offset, 1.0, 0.0],
                ])
                self.vertex_normals = np.ones((3, 3), dtype=np.float32)

        meshes = {
            str(Path("/tmp/cat2.obj").resolve()): _Mesh(0.0),
            str(Path("/tmp/cat3.obj").resolve()): _Mesh(2.0),
        }

        with patch(
            "zerith.zerith_pose_estimator.load_trimesh",
            side_effect=lambda path: meshes[str(Path(path).resolve())],
        ) as load_mesh, patch(
            "zerith.zerith_pose_estimator.trimesh.bounds.oriented_bounds",
            return_value=(np.eye(4), np.ones(3)),
        ), patch(
            "zerith.zerith_pose_estimator.FoundationPose",
            side_effect=lambda **kwargs: object(),
        ) as foundation_pose:
            estimator = ZerithPoseEstimator(
                "/tmp/cat2.obj",
                scorer=object(),
                refiner=object(),
                glctx=object(),
                debug_dir="/tmp",
            )
            cat2_estimator = estimator.estimator
            estimator.preload_objects(["/tmp/cat2.obj", "/tmp/cat3.obj"])
            estimator.reset_object("/tmp/cat3.obj")
            cat3_estimator = estimator.estimator
            estimator.reset_object("/tmp/cat2.obj")

            self.assertEqual(load_mesh.call_count, 2)
            self.assertEqual(foundation_pose.call_count, 2)
            self.assertIs(estimator.estimator, cat2_estimator)
            self.assertIsNot(cat3_estimator, cat2_estimator)
            self.assertEqual(len(estimator.cached_mesh_files), 2)

    def test_server_yolo_detection_and_registration_reuse_exact_masks(self):
        detector = _FakeYoloDetector()
        segmentation = ZerithYoloSegmentation(
            parts_config="zerith/parts_config.json",
            detector=detector,
            confidence=0.6,
            iou_threshold=0.7,
            device="cpu",
        )
        estimator = _FakePoseEstimator()
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            server = ZerithServer(estimator, segmentation, save_dir=tmpdir)
            detection = server._handle_detection({"rgb": rgb})
            self.assertEqual(detection["status"], "success")
            self.assertNotIn("debug_images", detection)

            objects = []
            for box in detection["boxes"]:
                objects.append({
                    "label": box["label"],
                    "category_id": box["category_id"],
                    "box": [box[key] for key in ("x1", "y1", "x2", "y2")],
                    "detection_id": box["detection_id"],
                    "mask_id": box["mask_id"],
                    "iteration": 5,
                })
            registration = server._handle_register_batch({
                "K": np.eye(3),
                "rgb": rgb,
                "depth": depth,
                "objects": objects,
                "debug_image_policy": {
                    "enabled": True,
                    "scope": "first_input_only",
                    "max_sets": 1,
                    "image_keys": ["detection_boxes", "all_parts"],
                },
            })

            self.assertEqual(registration["status"], "success")
            self.assertEqual(registration["successful"], 2)
            self.assertEqual(len(detector.calls), 1)
            self.assertEqual(
                [call["category_id"] for call in estimator.register_calls],
                ["cat2", "cat3"],
            )
            self.assertEqual(
                [result["pose"].shape for result in registration["results"]],
                [(4, 4), (4, 4)],
            )
            self.assertEqual(registration["detection_id"], 0)
            debug_images = registration["debug_images"]
            self.assertEqual(len(debug_images["sets"]), 1)
            images = debug_images["sets"][0]["images"]
            self.assertEqual(set(images), {"detection_boxes", "all_parts"})
            for image_bytes in images.values():
                self.assertIsInstance(image_bytes, bytes)
                self.assertTrue(image_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
                with Image.open(BytesIO(image_bytes)) as image:
                    self.assertEqual(image.size, (8, 8))

            self.assertEqual(
                images["detection_boxes"],
                (Path(tmpdir) / "detections" / "0_boxes.png").read_bytes(),
            )
            self.assertEqual(
                images["all_parts"],
                (
                    Path(tmpdir) / "mask_overlay" / "all_parts.png"
                ).read_bytes(),
            )

            transported = pickle.loads(pickle.dumps(registration))
            self.assertEqual(
                transported["debug_images"]["sets"][0]["images"]
                ["detection_boxes"],
                images["detection_boxes"],
            )

            next_detection = server._handle_detection({"rgb": rgb})
            next_objects = []
            for box in next_detection["boxes"]:
                next_objects.append({
                    "label": box["label"],
                    "category_id": box["category_id"],
                    "box": [
                        box[key] for key in ("x1", "y1", "x2", "y2")
                    ],
                    "detection_id": box["detection_id"],
                    "mask_id": box["mask_id"],
                    "iteration": 5,
                })
            repeated_registration = server._handle_register_batch({
                "K": np.eye(3),
                "rgb": rgb,
                "depth": depth,
                "objects": next_objects,
                "debug_image_policy": {
                    "enabled": True,
                    "scope": "first_input_only",
                    "max_sets": 1,
                    "image_keys": ["detection_boxes", "all_parts"],
                },
            })
            self.assertEqual(repeated_registration["status"], "success")
            self.assertNotIn("debug_images", repeated_registration)

    def test_client_uploads_rgbd_once_and_saves_all_results(self):
        pose_a = np.eye(4, dtype=np.float32)
        pose_b = np.eye(4, dtype=np.float32)
        pose_b[0, 3] = 0.25
        response = {
            "status": "success",
            "results": [
                {
                    "status": "success",
                    "pose": pose_a,
                    "mask_box": [1, 2, 5, 6],
                    "debug_index": 10,
                    "elapsed_seconds": 0.1,
                },
                {
                    "status": "success",
                    "pose": pose_b,
                    "mask_box": [9, 4, 15, 10],
                    "debug_index": 11,
                    "elapsed_seconds": 0.2,
                },
            ],
            "timing": {
                "segmentation_seconds": 0.05,
                "registration_seconds": 0.30,
            },
        }
        client = _FakeBatchClient(response)
        K = np.eye(3)
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        depth = np.ones((12, 16), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            items = []
            for index, category_id in enumerate(("cat2", "cat3")):
                items.append({
                    "label": category_id,
                    "category_id": category_id,
                    "box": [index, 0, index + 2, 2],
                    "detection_id": 23,
                    "mask_id": f"23:{index}",
                    "mesh_bbox": None,
                    "to_origin": None,
                    "label_output_dir": str(Path(tmpdir) / category_id),
                })

            successes = process_labels_batch(
                client,
                K,
                rgb,
                depth,
                items,
                register_iterations=5,
            )

            self.assertEqual(successes, [True, True])
            self.assertEqual(len(client.calls), 1)
            self.assertIs(client.calls[0][1], rgb)
            self.assertIs(client.calls[0][2], depth)
            sent_objects = client.calls[0][3]
            self.assertEqual(
                [item["mask_id"] for item in sent_objects],
                ["23:0", "23:1"],
            )
            saved_a = np.loadtxt(Path(tmpdir) / "cat2" / "ob_in_cam/0.txt")
            saved_b = np.loadtxt(Path(tmpdir) / "cat3" / "ob_in_cam/0.txt")
            self.assertTrue(np.allclose(saved_a, pose_a))
            self.assertTrue(np.allclose(saved_b, pose_b))


if __name__ == "__main__":
    unittest.main()
