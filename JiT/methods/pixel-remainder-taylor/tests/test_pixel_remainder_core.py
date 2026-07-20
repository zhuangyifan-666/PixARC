from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from pixel_remainder_taylor.config import validate_method_config
from pixel_remainder_taylor.controller import plan_segment, split_bands
from pixel_remainder_taylor.finite_difference import (
    nonuniform_lagrange_weights,
    nonuniform_polynomial_forecast,
    taylor_forecast,
)
from pixel_remainder_taylor.runtime import PixelRemainderRuntime
from pixel_remainder_taylor.scheduler import (
    FULL,
    TAYLOR,
    DynamicSegmentScheduler,
    FixedParityScheduler,
    expected_network_forward_count,
)
from pixel_remainder_taylor.state import ModuleTaylorState, PixelHistory


def _factor(value: float, batch: int = 2) -> torch.Tensor:
    return torch.full((batch, 3, 16, 16), value, dtype=torch.float32)


def test_pixel_factors_exact_only():
    history = PixelHistory()
    history.update_exact(_factor(1.0), coordinate=8)
    history.update_exact(_factor(2.0), coordinate=7)
    snapshot = [value.clone() for value in history.anchor_values]
    _ = history.forecast(6, order_override=1)
    assert history.exact_update_count == 2
    assert all(
        torch.equal(left, right)
        for left, right in zip(snapshot, history.anchor_values, strict=True)
    )


def test_remainder_formula_order1():
    plan = plan_segment(
        [_factor(2.0), _factor(0.0), _factor(2.0)],
        feature_available_order_min=1,
        nfe_index=0,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
    )
    assert plan.risk[1][3] == pytest.approx(3**2 / 2)


def test_remainder_formula_order2():
    plan = plan_segment(
        [_factor(3.0), _factor(0.0), _factor(0.0), _factor(3.0)],
        feature_available_order_min=2,
        nfe_index=0,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
    )
    assert plan.risk[2][3] == pytest.approx(3**3 / 6)


def test_low_high_split_reconstruction():
    value = torch.randn(2, 3, 16, 16)
    low, high = split_bands(value)
    assert torch.allclose(low + high, value)


def test_progress_gates_high_frequency():
    grid = torch.arange(16).view(1, 1, 16, 1) + torch.arange(16).view(1, 1, 1, 16)
    checker = (grid.remainder(2) * 2 - 1).float().repeat(1, 3, 1, 1)
    p0 = 2.0 + checker
    factors = [p0, torch.zeros_like(p0), checker]
    early = plan_segment(
        factors,
        feature_available_order_min=1,
        nfe_index=0,
        total_nfe=99,
        tau=1.0,
        max_taylor_span=1,
    )
    late = plan_segment(
        factors,
        feature_available_order_min=1,
        nfe_index=98,
        total_nfe=99,
        tau=1.0,
        max_taylor_span=1,
    )
    assert early.risk[1][1] == pytest.approx(0.0, abs=1e-7)
    assert late.risk[1][1] > early.risk[1][1]


def test_safe_h_monotonic_in_tau():
    factors = [_factor(2.0), _factor(0.0), _factor(0.2)]
    conservative = plan_segment(
        factors,
        feature_available_order_min=1,
        nfe_index=30,
        total_nfe=99,
        tau=0.05,
        max_taylor_span=3,
    )
    aggressive = plan_segment(
        factors,
        feature_available_order_min=1,
        nfe_index=30,
        total_nfe=99,
        tau=0.5,
        max_taylor_span=3,
    )
    assert aggressive.safe_h[1] >= conservative.safe_h[1]


def test_safe_h_bounded_by_cap():
    plan = plan_segment(
        [_factor(2.0), _factor(0.0), _factor(0.001)],
        feature_available_order_min=1,
        nfe_index=1,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
    )
    assert 0 <= plan.selected_span <= 3


def test_safe_h_is_capped_by_remaining_trajectory():
    factors = [_factor(2.0), _factor(0.0), _factor(0.001)]
    penultimate = plan_segment(
        factors,
        feature_available_order_min=1,
        nfe_index=97,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
        available_future_nfe=1,
    )
    final = plan_segment(
        factors,
        feature_available_order_min=1,
        nfe_index=98,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
        available_future_nfe=0,
    )
    assert penultimate.selected_span == 1
    assert final.selected_span == 0
    assert final.selected_order is None


def test_tie_prefers_lower_order():
    plan = plan_segment(
        [_factor(2.0), _factor(0.0), _factor(0.001), _factor(0.001)],
        feature_available_order_min=2,
        nfe_index=1,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
    )
    assert plan.safe_h[1] == plan.safe_h[2] == 3
    assert plan.selected_order == 1


def test_order2_requires_mature_histories():
    factors = [_factor(2.0), _factor(0.0), _factor(10.0), _factor(0.001)]
    immature = plan_segment(
        factors,
        feature_available_order_min=1,
        nfe_index=1,
        total_nfe=99,
        tau=1.0,
        max_taylor_span=3,
    )
    assert immature.safe_h[2] == 0
    assert 2 not in immature.risk


def test_dynamic_schedule_counts():
    scheduler = DynamicSegmentScheduler()
    scheduler.reset(8)
    actions = []
    for index in range(8):
        decision = scheduler.decide(
            macro_step_index=index,
            solver_stage="predictor",
            continuous_t=0.0,
            t_next=0.1,
        )
        actions.append(decision.action)
        if index == 2:
            scheduler.plan_next_segment(
                anchor_q=5,
                selected_order=1,
                selected_span=2,
                risk_table={},
            )
        elif decision.action == FULL and index >= 3:
            scheduler.plan_next_segment(
                anchor_q=7 - index,
                selected_order=None,
                selected_span=0,
                risk_table={},
            )
    assert actions == [FULL, FULL, FULL, TAYLOR, TAYLOR, FULL, FULL, FULL]


def test_diagnostic_return_replaces_taylor_before_stream_execution():
    runtime = PixelRemainderRuntime(
        mode="pixel_remainder_taylor", tau=0.02, max_taylor_span=3
    )
    runtime.begin_trajectory(
        total_nfe=4,
        expected_streams={"combined_cfg"},
        trajectory_id="diagnostic",
        sample_ids=[0],
    )
    runtime.scheduler.nfe_index = 3
    runtime.scheduler.full_count = 3
    runtime.scheduler.plan_next_segment(
        anchor_q=1, selected_order=1, selected_span=1, risk_table={}
    )
    decision = runtime.begin_nfe(
        macro_step_index=1,
        solver_stage="corrector",
        continuous_t=0.5,
        t_next=0.5,
    )
    assert decision.action == TAYLOR
    replacement = runtime.force_current_full("diagnostic_return")
    assert replacement.action == FULL
    assert replacement.full_reason == "diagnostic_return"
    assert runtime.scheduler.full_count == 4
    assert runtime.scheduler.taylor_count == 0
    runtime.reset()


def test_no_extra_forward_contract_jit():
    calls = 0
    for _nfe in range(99):
        calls += 2  # conditional and unconditional B-sized calls
    assert calls == expected_network_forward_count(
        model_family="jit", sampler="heun", num_steps=50
    ) == 198


def test_no_extra_forward_contract_pixelgen():
    calls = 0
    for _nfe in range(99):
        calls += 1  # one combined 2B CFG call
    assert calls == expected_network_forward_count(
        model_family="pixelgen", sampler="heun", num_steps=50
    ) == 99


def test_cfg_real_batch_reduction():
    real = [_factor(2.0, batch=3), _factor(0.0, batch=3), _factor(0.2, batch=3)]
    duplicated = [torch.cat([value, value]) for value in real]
    real_plan = plan_segment(
        real,
        feature_available_order_min=1,
        nfe_index=10,
        total_nfe=99,
        tau=1.0,
        max_taylor_span=3,
    )
    duplicate_plan = plan_segment(
        duplicated,
        feature_available_order_min=1,
        nfe_index=10,
        total_nfe=99,
        tau=1.0,
        max_taylor_span=3,
    )
    assert real_plan.risk == duplicate_plan.risk
    assert real[0].shape[0] == 3


def test_state_dict_clean():
    module = nn.Linear(2, 2)
    runtime = PixelRemainderRuntime(
        mode="pixel_remainder_taylor",
        tau=0.02,
        max_taylor_span=3,
    )
    object.__setattr__(module, "pixel_remainder_runtime", runtime)
    before = set(module.state_dict())
    clone = copy.deepcopy(module)
    assert set(clone.state_dict()) == before
    assert clone.pixel_remainder_runtime is not runtime
    assert clone.pixel_remainder_runtime.pixel_history.available_order == -1


def test_nonfinite_forces_full():
    factors = [_factor(1.0), _factor(0.0), _factor(float("nan"))]
    plan = plan_segment(
        factors,
        feature_available_order_min=2,
        nfe_index=20,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
    )
    assert plan.nonfinite
    assert plan.selected_span == 0
    assert plan.selected_order is None


def test_fixed_schedule_parity_mode():
    scheduler = FixedParityScheduler(interval=3, order=2)
    scheduler.reset(99)
    actions = [
        scheduler.decide(
            macro_step_index=index // 2,
            solver_stage="predictor",
            continuous_t=0.0,
            t_next=0.0,
        ).action
        for index in range(99)
    ]
    assert actions[:8] == [FULL, FULL, TAYLOR, TAYLOR, FULL, TAYLOR, TAYLOR, FULL]
    assert actions.count(FULL) == 34
    assert actions.count(TAYLOR) == 65


def test_order_override_preserves_higher_factors():
    state = ModuleTaylorState()
    for coordinate, value in zip((5, 4, 3), (1.0, 2.0, 4.0)):
        state.update_exact(
            torch.tensor([value]),
            coordinate=coordinate,
            max_order=2,
            cache_dtype="inherit",
        )
    snapshot = [factor.clone() for factor in state.cached_tensors()]
    first = state.forecast(2, order_override=1)
    second = state.forecast(2, order_override=2)
    assert not torch.equal(first, second)
    assert len(state.cached_tensors()) == 3
    assert all(
        torch.equal(a, b)
        for a, b in zip(snapshot, state.cached_tensors(), strict=True)
    )


def test_nonuniform_exact_anchor_quadratic_forecast():
    state = ModuleTaylorState()
    for coordinate in (98, 97, 94):
        value = torch.tensor([float((coordinate - 100) ** 2)])
        state.update_exact(
            value,
            coordinate=coordinate,
            max_order=2,
            cache_dtype="inherit",
        )
    assert state.forecast(93, order_override=2).item() == pytest.approx(49.0)
    assert state.forecast(93, order_override=1).item() == pytest.approx(45.0)


def test_bf16_large_coordinate_extrapolation_accumulates_in_fp32():
    coordinates = (1000, 1004, 1008, 1012)
    values = [
        torch.tensor([value], dtype=torch.bfloat16)
        for value in (0.0, 0.064, 0.512, 1.728)
    ]
    weights = nonuniform_lagrange_weights(coordinates, 1036)
    expected_fp32 = sum(
        value.float() * weight
        for value, weight in zip(values, weights, strict=True)
    ).to(torch.bfloat16)
    forecast = nonuniform_polynomial_forecast(
        coordinates, values, coordinate=1036, order_override=3
    )
    assert torch.equal(forecast, expected_fp32)


def test_config_rejects_tai_and_null_tau():
    fixed = {
        "mode": "pixel_remainder_taylor",
        "max_taylor_span": 3,
        "stored_feature_order": 2,
        "pixel_max_order": 3,
        "warmup_full_nfe": 3,
        "pool_kernel": 8,
        "batch_reduction": "mean",
        "cache_dtype": "inherit",
        "trace_mode": "full",
    }
    with pytest.raises(ValueError):
        validate_method_config({**fixed, "tau": None})
    with pytest.raises(ValueError):
        validate_method_config({**fixed, "tai": 0.1, "tau": 0.1})
    with pytest.raises(ValueError, match="unknown method keys"):
        validate_method_config({**fixed, "tau": 0.1, "lambda": 1.0})
