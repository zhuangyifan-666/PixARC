from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from pixel_remainder_taylor.config import validate_method_config
from pixel_remainder_taylor.controller import plan_segment, split_bands
from pixel_remainder_taylor.finite_difference import (
    UnsafeInterpolationError,
    nonuniform_lagrange_weights,
    nonuniform_polynomial_forecast,
    taylor_forecast,
)
from pixel_remainder_taylor.jit_denoiser import PixelRemainderTaylorDenoiser
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
    assert decision.action == FULL
    assert decision.full_reason == "unsafe_forecast_preflight:RuntimeError"
    assert runtime.scheduler.full_count == 4
    assert runtime.scheduler.taylor_count == 0
    runtime.reset()


def test_no_extra_forward_contract_jit():
    class CountingNet(nn.Module):
        def __init__(self, runtime):
            super().__init__()
            object.__setattr__(self, "runtime", runtime)
            self.calls = 0

        def forward_taylor(self, z, t, labels, *, stream_id):
            self.calls += 1
            shaped_t = t.reshape(-1, 1, 1, 1).to(z)
            result = self.runtime.branch(
                stream_id=stream_id,
                layer_idx=0,
                module_name="dummy",
                exact_fn=lambda: z + (1.0 - shaped_t),
            )
            self.runtime.mark_stream_complete(stream_id)
            return result

    model = object.__new__(PixelRemainderTaylorDenoiser)
    nn.Module.__init__(model)
    runtime = PixelRemainderRuntime(
        mode="instrumented_full", tau=0.0, max_taylor_span=3
    )
    object.__setattr__(model, "pixel_remainder_runtime", runtime)
    model.net = CountingNet(runtime)
    model.img_size = 8
    model.num_classes = 10
    model.noise_scale = 1.0
    model.t_eps = 1.0e-6
    model.method = "heun"
    model.steps = 50
    model.cfg_scale = 3.0
    model.cfg_interval = (0.1, 1.0)
    object.__setattr__(model, "_network_forward_count", 0)
    object.__setattr__(model, "_trajectory_serial", 0)
    object.__setattr__(model, "_last_pixel_remainder_summary", None)
    labels = torch.tensor([1, 2])
    output = model.generate(labels, noise=torch.zeros(2, 3, 8, 8), sample_ids=[7, 8])
    assert output.shape == (2, 3, 8, 8)
    assert model.net.calls == expected_network_forward_count(
        model_family="jit", sampler="heun", num_steps=50
    ) == 198
    summary = model._last_pixel_remainder_summary
    assert summary["total_nfe"] == 99
    assert summary["network_forward_count"] == 198
    assert summary["stage_statistics"] == {
        "corrector": {"full_nfe": 49, "taylor_nfe": 0},
        "final_euler": {"full_nfe": 1, "taylor_nfe": 0},
        "predictor": {"full_nfe": 49, "taylor_nfe": 0},
    }
    assert not runtime.active


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


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_nonuniform_decreasing_grid_exact_polynomials(degree: int):
    coordinates = [10, 7, 3, -2]
    values = [torch.tensor([float(q**degree)]) for q in coordinates]
    forecast = nonuniform_polynomial_forecast(
        coordinates,
        values,
        coordinate=-4,
        order_override=degree,
    )
    assert forecast.item() == pytest.approx(float((-4) ** degree), abs=1e-4)


def test_required_nonuniform_quadratic_counterexample():
    coordinates = [10, 7, 3]
    values = [torch.tensor([float(q * q)]) for q in coordinates]
    result = nonuniform_polynomial_forecast(
        coordinates, values, coordinate=2, order_override=2
    )
    assert result.item() == pytest.approx(4.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_forecast_returns_storage_dtype_after_fp32_sum(dtype):
    coordinates = [10, 7, 3]
    values = [torch.tensor([float(q * q)], dtype=dtype) for q in coordinates]
    weights = nonuniform_lagrange_weights(coordinates, 2)
    expected = sum(
        value.float() * weight
        for value, weight in zip(values, weights, strict=True)
    ).to(dtype)
    result = nonuniform_polynomial_forecast(
        coordinates, values, coordinate=2, order_override=2
    )
    assert result.dtype == dtype
    assert torch.equal(result, expected)


def test_invalid_and_ill_conditioned_interpolation_fail_closed():
    with pytest.raises(ValueError, match="distinct"):
        nonuniform_lagrange_weights([3, 3], 2)
    with pytest.raises(ValueError, match="finite"):
        nonuniform_lagrange_weights([3, float("nan")], 2)
    with pytest.raises(UnsafeInterpolationError):
        nonuniform_lagrange_weights([0.0, 1.0e-12, 1.0], 10.0)


def test_runtime_backend_is_mode_specific():
    dynamic = PixelRemainderRuntime(
        mode="pixel_remainder_taylor", tau=0.01, max_taylor_span=3
    )
    fixed = PixelRemainderRuntime(
        mode="fixed_schedule_parity",
        tau=0.0,
        max_taylor_span=3,
        debug_fixed_interval=3,
        debug_fixed_order=2,
    )
    assert dynamic.predictor_backend == "nonuniform_polynomial"
    assert fixed.predictor_backend == "legacy_recursive"


def _run_trace_mode(trace_mode: str):
    runtime = PixelRemainderRuntime(
        mode="pixel_remainder_taylor",
        tau=1.0,
        max_taylor_span=3,
        trace_mode=trace_mode,
    )
    runtime.begin_trajectory(
        total_nfe=9,
        expected_streams={"combined_cfg"},
        trajectory_id="trace-equivalence",
        sample_ids=[11, 12],
    )
    actions = []
    outputs = []
    for index in range(9):
        decision = runtime.begin_nfe(
            macro_step_index=index // 2,
            solver_stage="predictor" if index % 2 == 0 else "corrector",
            continuous_t=index / 9,
            t_next=(index + 1) / 9,
        )
        actions.append(decision.action)
        value = runtime.branch(
            stream_id="combined_cfg",
            layer_idx=0,
            module_name="gate",
            exact_fn=lambda q=decision.q: torch.full((2, 4), float(q * q)),
        )
        outputs.append(value.clone())
        runtime.mark_stream_complete("combined_cfg")
        state = _factor(2.0 + decision.q * 0.01)
        runtime.end_nfe(
            current_state=state,
            t=0.5,
            guided_velocity=torch.zeros_like(state),
        )
    return actions, outputs, runtime.end_trajectory()


def test_full_and_summary_trace_do_not_change_actions_outputs_or_counts():
    full_actions, full_outputs, full = _run_trace_mode("full")
    summary_actions, summary_outputs, summary = _run_trace_mode("summary")
    assert full_actions == summary_actions
    assert all(
        torch.equal(left, right)
        for left, right in zip(full_outputs, summary_outputs, strict=True)
    )
    for key in (
        "total_nfe", "full_nfe", "taylor_nfe", "order1_taylor_nfe",
        "order2_taylor_nfe", "span_histogram", "stage_statistics",
    ):
        assert full[key] == summary[key]
    assert len(full["nfe_trace"]) == 9
    assert "nfe_trace" not in summary


def test_vectorized_dynamic_risk_matches_direct_reference():
    coordinates = [10, 7, 3, 0]
    grid = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1)
    anchors = [
        (1.0 + q * 0.01 + grid * (q + 1) * 1.0e-4).repeat(2, 3, 1, 16)
        for q in coordinates
    ]
    plan = plan_segment(
        anchors,
        pixel_coordinates=coordinates,
        feature_available_order_min=2,
        nfe_index=40,
        total_nfe=99,
        tau=10.0,
        max_taylor_span=3,
    )
    progress = 40 / 98
    base_low, base_high = (
        component.abs().mean(dim=(1, 2, 3))
        for component in split_bands(anchors[-1])
    )
    for order in (1, 2):
        for horizon in range(1, 4):
            protected = nonuniform_polynomial_forecast(
                coordinates,
                anchors,
                coordinate=-horizon,
                order_override=order + 1,
            )
            selected = nonuniform_polynomial_forecast(
                coordinates,
                anchors,
                coordinate=-horizon,
                order_override=order,
            )
            omitted_low, omitted_high = (
                component.abs().mean(dim=(1, 2, 3))
                for component in split_bands(protected - selected)
            )
            direct = torch.maximum(
                omitted_low / (base_low + 1.0e-6),
                progress * omitted_high / (base_high + 1.0e-6),
            )
            assert plan.risk[order][horizon] == pytest.approx(float(direct.mean()))
            assert plan.risk_max[order][horizon] == pytest.approx(float(direct.max()))


def test_exception_during_jit_sampling_clears_runtime_state():
    class FailingNet(nn.Module):
        def forward_taylor(self, *args, **kwargs):
            raise RuntimeError("synthetic failure")

    model = object.__new__(PixelRemainderTaylorDenoiser)
    nn.Module.__init__(model)
    runtime = PixelRemainderRuntime(
        mode="instrumented_full", tau=0.0, max_taylor_span=3
    )
    object.__setattr__(model, "pixel_remainder_runtime", runtime)
    model.net = FailingNet()
    model.img_size = 8
    model.num_classes = 10
    model.noise_scale = 1.0
    model.t_eps = 1.0e-6
    model.method = "heun"
    model.steps = 50
    model.cfg_scale = 3.0
    model.cfg_interval = (0.1, 1.0)
    object.__setattr__(model, "_network_forward_count", 0)
    object.__setattr__(model, "_trajectory_serial", 0)
    object.__setattr__(model, "_last_pixel_remainder_summary", None)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        model.generate(
            torch.tensor([1]), noise=torch.zeros(1, 3, 8, 8), sample_ids=[1]
        )
    assert not runtime.active
    assert runtime.pixel_history.available_order == -1
    assert runtime.streams == {}


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
