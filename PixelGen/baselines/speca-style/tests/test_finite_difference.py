import pytest
import torch

from speca_style.finite_difference import factor_count_after_exact, taylor_forecast, update_factors


def test_signed_gaps_and_orders():
    old = [torch.tensor(4.0), torch.tensor(2.0)]
    positive = update_factors(old, torch.tensor(9.0), coordinate=3, previous_coordinate=2, max_order=2)
    assert [float(value) for value in positive] == [9.0, 5.0, 3.0]
    negative = update_factors(old, torch.tensor(1.0), coordinate=1, previous_coordinate=2, max_order=2)
    assert [float(value) for value in negative] == [1.0, 3.0, -1.0]
    assert float(taylor_forecast(positive, coordinate=4, anchor_coordinate=3)) == pytest.approx(15.5)


def test_first_anchor_dtype_and_count():
    value = torch.ones(2, 3, dtype=torch.float16)
    factors = update_factors([], value, coordinate=8, previous_coordinate=None, max_order=4, cache_dtype="fp32")
    assert len(factors) == 1 and factors[0].dtype == torch.float32
    assert [factor_count_after_exact(count, 4) for count in range(7)] == [0, 1, 2, 3, 4, 5, 5]


def test_zero_gap_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        update_factors([torch.tensor(1.0)], torch.tensor(2.0), coordinate=1, previous_coordinate=1, max_order=1)

