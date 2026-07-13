import pytest
import torch

from dicache_style.pixelgen_lightning import _require_finite_tensor


def test_finite_output_guard_accepts_finite_and_rejects_nan_inf():
    _require_finite_tensor(torch.tensor([0.0, 1.0]), stage="fixture")
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(FloatingPointError, match="non-finite"):
            _require_finite_tensor(torch.tensor([value]), stage="fixture")
