import math
import unittest
from dataclasses import replace

import numpy as np

from seacache_style.manifest import build_manifest
from seacache_style.paired_metrics import pair_metrics, validate_pair_manifests


class PairedMetricTest(unittest.TestCase):
    def test_identical_and_known_difference(self):
        reference = np.zeros((8, 8, 3), dtype=np.float32)
        mse, psnr, ssim = pair_metrics(reference, reference.copy())
        self.assertEqual(mse, 0.0)
        self.assertTrue(math.isinf(psnr))
        self.assertEqual(ssim, 1.0)
        candidate = np.full((8, 8, 3), 0.5, dtype=np.float32)
        mse, psnr, ssim = pair_metrics(reference, candidate)
        self.assertAlmostEqual(mse, 0.25)
        self.assertAlmostEqual(psnr, 10 * math.log10(4.0))
        self.assertLess(ssim, 1.0)

    def test_manifest_missing_class_seed_and_duplicate_fail(self):
        values = build_manifest(
            samples_per_class=1,
            base_seed=1,
            split_name="toy",
            world_size=1,
            batch_size=2,
            num_classes=3,
        )
        with self.assertRaises(ValueError):
            validate_pair_manifests(values, values[:-1])
        changed_class = list(values)
        changed_class[0] = replace(changed_class[0], class_id=99)
        with self.assertRaises(ValueError):
            validate_pair_manifests(values, changed_class)
        changed_seed = list(values)
        changed_seed[0] = replace(changed_seed[0], seed=99)
        with self.assertRaises(ValueError):
            validate_pair_manifests(values, changed_seed)
        with self.assertRaises(ValueError):
            validate_pair_manifests(values, list(values) + [values[0]])

    def test_shape_and_dtype_fail(self):
        value = np.zeros((8, 8, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            pair_metrics(value, np.zeros((9, 8, 3), dtype=np.float32))
        with self.assertRaises(TypeError):
            pair_metrics(value, np.zeros((8, 8, 3), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
