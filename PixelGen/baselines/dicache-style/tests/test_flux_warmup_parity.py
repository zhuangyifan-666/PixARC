import pytest

from dicache_style.gate import flux_direct_full_reason


@pytest.mark.parametrize(
    "total,ratio,warm_last",
    [(30, 0.2, 6), (99, 0.2, 19), (30, 0.0, 0), (99, 0.0, 0)],
)
def test_flux_inclusive_boundary(total, ratio, warm_last):
    assert flux_direct_full_reason(call_index=warm_last, total_calls=total, ret_ratio=ratio, force_last_full=True) == "flux_inclusive_warmup"
    if warm_last + 1 < total - 1:
        assert flux_direct_full_reason(call_index=warm_last + 1, total_calls=total, ret_ratio=ratio, force_last_full=True) is None
    assert flux_direct_full_reason(call_index=total - 1, total_calls=total, ret_ratio=ratio, force_last_full=True) is not None


def test_ret_ratio_one_is_all_full():
    assert all(flux_direct_full_reason(call_index=i, total_calls=30, ret_ratio=1.0, force_last_full=True) for i in range(30))
