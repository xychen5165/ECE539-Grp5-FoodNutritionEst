"""Check containment and pale-food support without running SAM."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from appearance_selector import select_appearance


class AppearanceSelectorTest(unittest.TestCase):
    def test_preserves_pale_food_hole_and_rejects_external_fixture(self):
        image = np.full((200, 240, 3), 70, dtype=np.uint8)
        image[30:170, 45:195] = 210
        # A pale food region, explicitly separated from plate by SAM.
        image[80:110, 100:140] = 200
        plate = np.zeros(image.shape[:2], dtype=bool)
        plate[30:170, 45:195] = True
        pale = np.zeros_like(plate)
        pale[80:110, 100:140] = True
        plate[pale] = False
        external = np.zeros_like(plate)
        external[5:20, 5:25] = True
        # SAM masks can be Fortran-contiguous; OpenCV needs a C-contiguous
        # destination for drawing the hull.
        candidates = [{'segmentation': np.asfortranarray(mask)} for mask in [plate, pale, external]]
        food, info = select_appearance(candidates, image)
        self.assertEqual(info['plate_candidate_index'], 0)
        self.assertTrue(food[pale].all())
        self.assertFalse(food[external].any())
        self.assertFalse(food[50:60, 60:70].any())

    def test_missing_plate_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, 'No geometrically plausible'):
            select_appearance([], np.full((100, 100, 3), 200, dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
