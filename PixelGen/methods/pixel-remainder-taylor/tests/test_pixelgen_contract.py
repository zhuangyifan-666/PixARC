from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from pixel_remainder_taylor.controller import plan_segment
from pixel_remainder_taylor.pixelgen_sampler import (
    PixelRemainderTaylorHeunSampler,
    real_batch_sample_ids,
)
from pixel_remainder_taylor.runtime import PixelRemainderRuntime
from pixel_remainder_taylor.scheduler import expected_network_forward_count


def test_shared_core_resolves_in_pixelgen_package():
    plan = plan_segment(
        [
            torch.ones(2, 3, 16, 16),
            torch.zeros(2, 3, 16, 16),
            torch.full((2, 3, 16, 16), 0.01),
        ],
        feature_available_order_min=1,
        nfe_index=2,
        total_nfe=99,
        tau=1.0,
        max_taylor_span=3,
    )
    assert plan.selected_span == 3


def test_combined_cfg_real_ids_are_not_duplicated_for_controller():
    assert real_batch_sample_ids([3, 4], 2) == (3, 4)
    assert real_batch_sample_ids([3, 4, 3, 4], 2) == (3, 4)
    with pytest.raises(ValueError):
        real_batch_sample_ids([3, 4, 4, 3], 2)
    with pytest.raises(TypeError):
        real_batch_sample_ids([3.5], 1)
    with pytest.raises(ValueError):
        real_batch_sample_ids([], 0)


def test_pixelgen_no_extra_forward_contract():
    calls = sum(1 for _nfe in range(99))
    assert calls == expected_network_forward_count(
        model_family="pixelgen", sampler="heun", num_steps=50
    ) == 99


def test_guided_velocity_uses_sampler_epsilon_and_preserves_cfg_order():
    sampler = object.__new__(PixelRemainderTaylorHeunSampler)
    sampler.t_eps = 0.25
    sampler.guidance = 2.0
    sampler.guidance_interval_min = 0.1
    sampler.guidance_interval_max = 0.9
    sampler.guidance_fn = lambda velocity, scale: (
        velocity[:1] + scale * (velocity[1:] - velocity[:1])
    )
    cfg_x = torch.zeros(2, 1, 1, 1)
    raw = torch.tensor([[[[1.0]]], [[[3.0]]]])
    cfg_t = torch.tensor([0.9, 0.9])

    guided = sampler._guided_velocity(raw, cfg_x, cfg_t, torch.tensor([0.5]))

    # 1-t=0.1 is clamped to t_eps=0.25, so [uncond, cond]=[4,12].
    assert torch.equal(guided, torch.tensor([[[[20.0]]]]))


def test_deepcopy_runtime_is_empty_and_independent():
    runtime = PixelRemainderRuntime(
        mode="pixel_remainder_taylor", tau=0.02, max_taylor_span=3
    )
    clone = copy.deepcopy(runtime)
    assert clone is not runtime
    assert clone.pixel_history.available_order == -1
    assert not clone.active


def test_runtime_is_not_persistent_module_state():
    module = nn.Linear(2, 2)
    runtime = PixelRemainderRuntime(
        mode="pixel_remainder_taylor", tau=0.02, max_taylor_span=3
    )
    object.__setattr__(module, "pixel_remainder_runtime", runtime)
    assert not any("pixel" in key or "remainder" in key for key in module.state_dict())
