import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from B_nutrition_estimation.prepare_input_comparison import geometry, load_rows
from B_nutrition_estimation.prepare_crop_comparison import crop_from_mask
from B_nutrition_estimation.train_input_comparison import DualView, SingleView, standardize


class InputComparisonTests(unittest.TestCase):
    def test_natural_crop_uses_mask_bbox_margin_and_clamps(self):
        rgb = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:6, 4:12] = True
        crop, box = crop_from_mask(rgb, mask, margin_fraction=0.10)
        self.assertEqual(box, [3, 1, 13, 7])
        np.testing.assert_array_equal(crop, rgb[1:7, 3:13])

        edge = np.zeros((10, 20), dtype=bool)
        edge[0:2, 0:3] = True
        _, edge_box = crop_from_mask(rgb, edge, margin_fraction=0.10)
        self.assertEqual(edge_box[:2], [0, 0])

    def test_natural_crop_rejects_degenerate_masks(self):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            crop_from_mask(rgb, np.zeros((4, 5), dtype=bool))
        with self.assertRaises(ValueError):
            crop_from_mask(rgb, np.ones((4, 5), dtype=bool))

    def test_selection_reads_only_train_and_dev(self):
        rows = load_rows(limit=2)
        self.assertEqual([row["split"] for row in rows], ["train", "train", "dev", "dev"])
        self.assertEqual(len({row["dish_id"] for row in rows}), 4)

    def test_geometry_has_declared_order_and_values(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:6, 4:12] = True
        values = geometry(mask)
        np.testing.assert_allclose(values[:7], [0.16, 0.4, 0.4, 0.16, 1.0,
                                                 7.5 / 20, 3.5 / 10], rtol=0, atol=1e-6)
        self.assertAlmostEqual(values[7], np.log(2), places=6)

    def test_standardization_fits_training_only(self):
        train = np.array([[0, 10], [2, 14]], dtype=np.float32)
        dev = np.array([[100, 100]], dtype=np.float32)
        train_z, dev_z, mean, std = standardize(train, dev)
        np.testing.assert_allclose(mean, [1, 12])
        np.testing.assert_allclose(train_z.mean(axis=0), [0, 0], atol=1e-6)
        self.assertTrue((dev_z > 10).all())

    def test_models_have_expected_interfaces_and_matched_capacity(self):
        batch = 3
        original = torch.randn(batch, 512)
        masked = torch.randn(batch, 512)
        geo = torch.randn(batch, 8)
        b0, b2, b3 = SingleView(), DualView(False), DualView(True)
        self.assertEqual(tuple(b0(original, None, None).shape), (batch, 5))
        self.assertEqual(tuple(b2(original, masked, None).shape), (batch, 5))
        self.assertEqual(tuple(b3(original, masked, geo).shape), (batch, 5))
        counts = [sum(p.numel() for p in model.parameters()) for model in (b0, b2, b3)]
        self.assertEqual(counts, [66309, 66309, 66349])
        self.assertTrue((b3(original, masked, geo) > 0).all())


if __name__ == "__main__":
    unittest.main()
