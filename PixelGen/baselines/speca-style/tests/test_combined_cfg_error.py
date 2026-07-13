from __future__ import annotations

import pytest
import torch

from speca_style.error_metrics import BatchGlobalMetricAccumulator, error_value


@pytest.mark.parametrize("metric", ["l1", "l2", "relative_l1", "relative_l2", "cosine_similarity"])
def test_combined_2b_metric_uses_both_halves(metric):
    pred = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    exact = torch.tensor([[[2.0, 2.0]], [[1.0, 5.0]]])
    accumulator = BatchGlobalMetricAccumulator(metric=metric)
    accumulator.update(pred, exact)
    assert accumulator.finalize() == pytest.approx(
        error_value(pred, exact, metric=metric), rel=1e-6, abs=1e-7
    )


def test_only_validating_half_would_change_result():
    pred = torch.tensor([[[0.0]], [[100.0]]])
    exact = torch.tensor([[[0.0]], [[1.0]]])
    combined = error_value(pred, exact, metric="l1")
    unconditional_only = error_value(pred[:1], exact[:1], metric="l1")
    assert combined != unconditional_only

