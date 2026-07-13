from __future__ import annotations

import pytest
import torch

from speca_style.error_metrics import BatchGlobalMetricAccumulator, error_value


@pytest.mark.parametrize("metric", ["l1", "l2", "relative_l1", "relative_l2", "cosine_similarity"])
def test_cond_uncond_streaming_equals_cat(metric):
    pred_cond = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    exact_cond = torch.tensor([[[2.0, 2.0], [1.0, 5.0]]])
    pred_uncond = torch.tensor([[[5.0, 1.0], [3.0, 3.0]]])
    exact_uncond = torch.tensor([[[4.0, 2.0], [2.0, 1.0]]])
    accumulator = BatchGlobalMetricAccumulator(metric=metric)
    accumulator.update(pred_cond, exact_cond)
    accumulator.update(pred_uncond, exact_uncond)
    expected = error_value(
        torch.cat([pred_cond, pred_uncond]),
        torch.cat([exact_cond, exact_uncond]),
        metric=metric,
    )
    assert accumulator.finalize() == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_scalar_average_is_not_used_for_unequal_payload_sizes():
    pred_a, exact_a = torch.zeros(1, 1, 1), torch.ones(1, 1, 1)
    pred_b, exact_b = torch.ones(1, 10, 1), torch.ones(1, 10, 1)
    accumulator = BatchGlobalMetricAccumulator(metric="l1")
    accumulator.update(pred_a, exact_a)
    accumulator.update(pred_b, exact_b)
    assert accumulator.finalize() == pytest.approx(1 / 11)
    assert accumulator.finalize() != pytest.approx(0.5)
