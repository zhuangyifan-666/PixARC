import math

import numpy as np
import pytest

from dicache_style.manifest import ManifestRecord
from dicache_style.paired_metrics import pair_metrics, validate_pair_manifests


def record(sample_id=0, class_id=0, seed=10):
    return ManifestRecord(sample_id, class_id, seed, 0, sample_id, "x", "pixarc-imagenet-c2i-v1", f"0:{sample_id}", 0)


def test_identical_and_known_difference():
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    mse, psnr, ssim = pair_metrics(reference, reference.copy())
    assert mse == 0 and math.isinf(psnr) and ssim == 1
    candidate = np.ones_like(reference)
    mse, psnr, _ = pair_metrics(reference, candidate)
    assert mse == 1 and psnr == 0


@pytest.mark.parametrize("candidate", [[record(class_id=2)], [record(seed=20)], [record(1)]])
def test_manifest_mismatch_fails(candidate):
    with pytest.raises(ValueError):
        validate_pair_manifests([record()], candidate)
