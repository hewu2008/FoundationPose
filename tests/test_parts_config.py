import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from zerith.zerith_locate_anything import (
    LocateAnythingWorker,
    load_part_configs,
)


class PartsConfigTest(unittest.TestCase):
    def test_default_config_has_one_existing_mesh_per_part(self):
        parts = load_part_configs()
        self.assertEqual([part["id"] for part in parts], ["cat4", "cat1", "cat3", "cat2"])
        self.assertTrue(all(Path(part["mesh_file"]).is_file() for part in parts))

    def test_category_count_comes_only_from_json(self):
        mesh = Path("assets/category_1.obj").resolve()
        payload = {
            "parts": [
                {
                    "id": f"part{i}",
                    "label": f"custom part {i}",
                    "mesh_file": str(mesh),
                    "area_range": [0.01, 0.20],
                    "aspect_range": [0.2, 4.0],
                }
                for i in range(2)
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parts = load_part_configs(path)

        self.assertEqual(len(parts), 2)
        worker = object.__new__(LocateAnythingWorker)
        worker._category_configs = parts
        raw = [{"label": "custom part 0", "x1": 10, "y1": 10, "x2": 40, "y2": 40}]
        boxes = worker._filter_boxes(raw, image_area=10000)
        boxes = worker._postprocess_classes(boxes, Image.new("RGB", (100, 100)))
        self.assertEqual(boxes[0]["category_id"], "part0")
        self.assertEqual(boxes[0]["mesh_file"], str(mesh))


if __name__ == "__main__":
    unittest.main()
