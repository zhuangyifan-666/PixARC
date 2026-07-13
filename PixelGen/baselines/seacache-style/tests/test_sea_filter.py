import importlib.util
import unittest
from pathlib import Path

import torch

from seacache_style.sea_filter import (
    apply_sea_from_ab,
    coefficients_from_time,
    rel_l1,
)


def _load_official():
    root = Path(__file__).resolve().parents[4]
    path = root / "baselines" / "SeaCache" / "FLUX" / "util_seacache.py"
    spec = importlib.util.spec_from_file_location("audited_seacache_util", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SeaFilterParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = _load_official()

    def test_float32_rectangular_full_fft_parity(self):
        torch.manual_seed(7)
        value = torch.randn(2, 3, 5, 4)
        expected = self.official.apply_sea_from_ab(
            value, 0.37, 0.63, dims=(-2, -3), norm_mode="mean", real=False
        )
        actual = apply_sea_from_ab(
            value, 0.37, 0.63, dims=(-2, -3), norm_mode="mean", real=False
        )
        self.assertEqual(actual.shape, value.shape)
        self.assertEqual(actual.dtype, value.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bfloat16_dtype_restore_and_parity(self):
        torch.manual_seed(8)
        value = torch.randn(1, 4, 6, 3).to(torch.bfloat16)
        expected = self.official.apply_sea_from_ab(value, 0.8, 0.2, dims=(-2, -3))
        actual = apply_sea_from_ab(value, 0.8, 0.2, dims=(-2, -3))
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_peak_and_real_paths_match_official(self):
        torch.manual_seed(9)
        value = torch.randn(2, 3, 7, 2)
        for norm_mode, real in [("peak", False), ("mean", True)]:
            with self.subTest(norm_mode=norm_mode, real=real):
                expected = self.official.apply_sea_from_ab(
                    value, 0.2, 0.8, dims=(-3, -2), norm_mode=norm_mode, real=real
                )
                actual = apply_sea_from_ab(
                    value, 0.2, 0.8, dims=(-3, -2), norm_mode=norm_mode, real=real
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_time_mapping_and_nonuniform_batch(self):
        self.assertEqual(coefficients_from_time(torch.tensor([0.25, 0.25])), (0.25, 0.75))
        self.assertEqual(coefficients_from_time(0.0), (1e-6, 1.0 - 1e-6))
        with self.assertRaises(ValueError):
            coefficients_from_time(torch.tensor([0.2, 0.3]))

    def test_relative_l1_matches_official(self):
        current = torch.tensor([[1.0, 3.0]])
        previous = torch.tensor([[2.0, 2.0]])
        self.assertEqual(rel_l1(current, previous), self.official.rel_l1(current, previous))


if __name__ == "__main__":
    unittest.main()

