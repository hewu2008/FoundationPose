import unittest

from PIL import Image, ImageDraw

from zerith.zerith_locate_anything import LocateAnythingWorker


class LocateAnythingAppearanceTest(unittest.TestCase):
    def setUp(self):
        self.worker = object.__new__(LocateAnythingWorker)
        self.configs = {
            config["id"]: config for config in self.worker._category_configs
        }

    def _candidate(self, category_id="cat3"):
        return {
            "label": self.configs[category_id]["label"],
            "category_id": category_id,
            "x1": 10,
            "y1": 10,
            "x2": 90,
            "y2": 90,
            "area_ratio": 0.03,
        }

    def _classified_id(self, dark_width):
        image = Image.new("RGB", (100, 100), (220, 220, 220))
        # The candidate is 80x80.  This 40-pixel-high end band covers either
        # 6.25% (shadow) or 8.125% (a substantial black cap) of the proposal.
        ImageDraw.Draw(image).rectangle(
            (10, 10, 10 + dark_width - 1, 49), fill=(40, 40, 40)
        )
        result = self.worker._postprocess_classes(
            [self._candidate()], image
        )
        self.assertEqual(len(result), 1)
        return result[0]["category_id"]

    def test_small_dark_shadow_does_not_turn_cat2_into_cat3(self):
        self.assertEqual(self._classified_id(dark_width=10), "cat2")

    def test_substantial_black_cap_stays_cat3(self):
        self.assertEqual(self._classified_id(dark_width=13), "cat3")

    def test_loose_cat3_box_uses_dark_to_white_ratio(self):
        self.assertTrue(
            self.worker._has_black_cap(
                {"dark_frac": 0.056, "white_frac": 0.272}
            )
        )

    def test_cat2_shadow_fails_dark_to_white_ratio(self):
        self.assertFalse(
            self.worker._has_black_cap(
                {"dark_frac": 0.061, "white_frac": 0.548}
            )
        )

    def test_top_expansion_recovers_black_cap_just_above_tight_box(self):
        image = Image.new("RGB", (100, 100), (220, 220, 220))
        # The detector starts the box at y=20 while the black cap occupies the
        # immediately preceding strip.  cat2/cat3 config expands appearance
        # inspection upward without changing the box used by SAM.
        ImageDraw.Draw(image).rectangle((25, 6, 74, 19), fill=(40, 40, 40))
        candidate = self._candidate(category_id="cat2")
        candidate.update({"x1": 10, "y1": 20, "x2": 90, "y2": 90})

        result = self.worker._postprocess_classes([candidate], image)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category_id"], "cat3")
        self.assertEqual(result[0]["appearance_top_expand"], 0.20)

    def test_dedicated_cat1_candidate_is_not_demoted_by_compact_area(self):
        image = Image.new("RGB", (100, 100), (220, 220, 220))
        candidate = self._candidate(category_id="cat1")
        candidate["source"] = "supplement"
        result = self.worker._postprocess_classes([candidate], image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category_id"], "cat1")

    def test_loose_cat1_derived_cat2_does_not_duplicate_native_cat2(self):
        loose = self._candidate(category_id="cat2")
        loose.update(
            {"x1": 10, "y1": 20, "x2": 70, "y2": 80, "refined_from": "cat1"}
        )
        native = self._candidate(category_id="cat2")
        native.update({"x1": 35, "y1": 20, "x2": 85, "y2": 80})
        result = self.worker._suppress_reclassified_cat2_duplicates(
            [loose, native]
        )
        self.assertEqual(result, [native])

    def test_cat1_nms_prefers_tight_dedicated_proposal(self):
        merged = self._candidate(category_id="cat1")
        merged.update(
            {
                "x1": 10,
                "y1": 10,
                "x2": 90,
                "y2": 90,
                "area_ratio": 0.064,
                "source": "supplement",
            }
        )
        tight = self._candidate(category_id="cat1")
        tight.update(
            {
                "x1": 25,
                "y1": 10,
                "x2": 90,
                "y2": 90,
                "area_ratio": 0.052,
                "source": "supplement",
            }
        )
        self.assertEqual(self.worker._nms([merged, tight], 0.5), [tight])

    def test_merged_box_check_counts_duplicate_child_proposals_once(self):
        parent = self._candidate(category_id="cat1")
        parent.update(
            {"x1": 10, "y1": 10, "x2": 90, "y2": 90, "area_ratio": 0.09}
        )
        child = self._candidate(category_id="cat2")
        child.update(
            {"x1": 15, "y1": 15, "x2": 45, "y2": 45, "area_ratio": 0.02}
        )
        duplicate = self._candidate(category_id="cat3")
        duplicate.update(
            {"x1": 16, "y1": 16, "x2": 46, "y2": 46, "area_ratio": 0.02}
        )
        result = self.worker._suppress_merged_boxes(
            [parent, child, duplicate]
        )
        self.assertIn(parent, result)

    def test_merged_box_check_removes_parent_over_two_distinct_children(self):
        parent = self._candidate(category_id="cat1")
        parent.update(
            {"x1": 10, "y1": 10, "x2": 90, "y2": 90, "area_ratio": 0.09}
        )
        left = self._candidate(category_id="cat2")
        left.update(
            {"x1": 15, "y1": 20, "x2": 40, "y2": 55, "area_ratio": 0.02}
        )
        right = self._candidate(category_id="cat3")
        right.update(
            {"x1": 60, "y1": 45, "x2": 85, "y2": 80, "area_ratio": 0.02}
        )
        result = self.worker._suppress_merged_boxes([parent, left, right])
        self.assertNotIn(parent, result)


if __name__ == "__main__":
    unittest.main()
