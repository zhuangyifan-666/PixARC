import pytest
import torch

from dicache_style.pixelgen_sampler import combined_cfg_sample_ids


def test_combined_cfg_ids_preserve_unconditional_then_conditional_order():
    assert combined_cfg_sample_ids([4, 9], 2) == (4, 9, 4, 9)
    assert combined_cfg_sample_ids(torch.tensor([4, 9, 4, 9]), 2) == (4, 9, 4, 9)
    with pytest.raises(ValueError):
        combined_cfg_sample_ids([4, 9, 9, 4], 2)


def test_one_combined_stream_not_split_branches():
    from dicache_style.runtime import DiCacheRuntime

    runtime = DiCacheRuntime(mode="instrumented_full", rel_l1_thresh=None)
    runtime.begin_trajectory(
        total_nfe=1,
        stream_total_calls={"combined_cfg": 1},
        trajectory_id="cfg",
        sample_ids=[2],
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    assert set(runtime.trajectory.streams) == {"combined_cfg"}
