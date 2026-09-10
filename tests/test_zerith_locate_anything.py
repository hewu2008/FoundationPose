import unittest
from pathlib import Path

from zerith.zerith_locate_anything import LocateAnythingWorker


def make_box(worker, category_id, coords, source):
    config = next(c for c in worker._category_configs if c["id"] == category_id)
    x1, y1, x2, y2 = coords
    return {
        "label": config["label"],
        "category_id": category_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "area_ratio": (x2 - x1) * (y2 - y1) / 10000,
        "source": source,
    }


class LocateAnythingPostprocessTest(unittest.TestCase):
    def setUp(self):
        # Post-processing helpers do not require loading the 3B model.
        self.worker = object.__new__(LocateAnythingWorker)

    def test_each_category_has_the_expected_mesh_file(self):
        expected = {
            "cat1": "assets/category_1.obj",
            "cat2": "assets/category_2_1.obj",
            "cat3": "assets/category_3.obj",
            "cat4": "assets/category_4.obj",
        }
        actual = {
            config["id"]: f"assets/{Path(config['mesh_file']).name}"
            for config in self.worker._category_configs
        }
        self.assertEqual(actual, expected)

    def test_parse_keeps_all_boxes_after_one_ref(self):
        answer = (
            "<ref>yellow boot</ref>"
            "<box><10><20><110><220></box>"
            "<box><300><400><500><700></box>"
            "<ref>white reservoir</ref>"
            "<box><600><100><900><300></box>"
        )
        boxes = self.worker.parse_boxes(answer, 1000, 500)
        self.assertEqual(len(boxes), 3)
        self.assertEqual([b["label"] for b in boxes], [
            "yellow boot", "yellow boot", "white reservoir"
        ])
        self.assertEqual(boxes[1]["y2"], 350)

    def test_joint_vote_resolves_cross_class_proposals(self):
        boxes = [
            make_box(self.worker, "cat1", (10, 10, 90, 90), "joint"),
            make_box(self.worker, "cat1", (12, 12, 91, 91), "per_class"),
            make_box(self.worker, "cat3", (11, 11, 92, 92), "per_class"),
        ]
        resolved = self.worker._resolve_category_conflicts(boxes, 0.55)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["category_id"], "cat1")
        self.assertEqual(resolved[0]["support"], 3)

    def test_ambiguous_per_class_tie_is_not_guessed_by_area(self):
        boxes = [
            make_box(self.worker, "cat1", (10, 10, 90, 90), "per_class"),
            make_box(self.worker, "cat3", (9, 9, 93, 93), "per_class"),
        ]
        self.assertEqual(self.worker._resolve_category_conflicts(boxes, 0.55), [])

    def test_overlapping_tile_grid_covers_image(self):
        windows = self.worker._tile_windows(640, 480, 2, 0.40)
        self.assertEqual(len(windows), 4)
        self.assertIn((0, 0, 400, 300), windows)
        self.assertIn((240, 180, 640, 480), windows)


if __name__ == "__main__":
    unittest.main()
