import math

import numpy as np
import pytest

from taylorseer_style.manifest import ManifestRecord
from taylorseer_style.paired_metrics import pair_metrics, validate_pair_manifests


def _record(sample_id=0, class_id=0, seed=10):
    return ManifestRecord(sample_id, class_id, seed, 0, sample_id, "toy", "pixarc-imagenet-c2i-v1", f"0:{sample_id}", 0)


def test_identical_and_known_difference_metrics():
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    mse, psnr, ssim = pair_metrics(reference, reference.copy())
    assert mse == 0 and math.isinf(psnr) and ssim == pytest.approx(1.0)
    candidate = reference.copy()
    candidate[0, 0, :] = 1.0
    mse, psnr, ssim = pair_metrics(reference, candidate)
    assert mse == pytest.approx(1 / 64) and math.isfinite(psnr) and ssim < 1


@pytest.mark.parametrize("candidate,match", [([], "sample IDs differ"), ([_record(class_id=1)], "class mismatch"), ([_record(seed=11)], "seed mismatch")])
def test_pair_manifest_failures(candidate, match):
    with pytest.raises(ValueError, match=match):
        validate_pair_manifests([_record()], candidate)


def test_duplicate_manifest_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_pair_manifests([_record(), _record()], [_record()])


def test_lpips_is_optional_dependency():
    pytest.importorskip("lpips", reason="LPIPS optional dependency/weights are not installed by tests")

