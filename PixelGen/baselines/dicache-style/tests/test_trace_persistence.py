import importlib.util
import sys
import types

if importlib.util.find_spec("lightning") is None:
    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    pytorch.Callback = type("Callback", (), {})
    lightning.pytorch = pytorch
    sys.modules["lightning"] = lightning
    sys.modules["lightning.pytorch"] = pytorch

from dicache_style.pixelgen_io import _scalar_summary
from dicache_style.trace import aggregate_trace_rows


def test_io_preserves_scalar_only_stream_and_shadow_series():
    trace = [
        {
            "solver_stage": "predictor",
            "delta_y": 0.1,
            "actual_full_residual_change": 0.2,
            "zero_order_relative_error": 0.4,
            "dcta_relative_error": 0.2,
        }
    ]
    value = _scalar_summary(
        {
            "trajectory_id": "t",
            "sample_ids": [7],
            "real_batch_size": 1,
            "stream_trace": trace,
            "shadow_scalar_series": trace,
        }
    )
    assert value["sample_ids"] == [7]
    assert value["stream_trace"] == trace
    assert value["shadow_scalar_series"] == trace


def test_io_rejects_bad_sample_ids_and_nulls_nonfinite_diagnostics():
    value = _scalar_summary(
        {
            "trajectory_id": "t",
            "sample_ids": [7],
            "real_batch_size": 1,
            "mean_delta_y": float("inf"),
            "stream_trace": [{"delta_y": float("nan"), "gamma": float("inf")}],
        }
    )
    assert value["mean_delta_y"] is None
    assert value["stream_trace"] == [{"delta_y": None, "gamma": None}]

    import pytest

    with pytest.raises(ValueError, match="sample_ids"):
        _scalar_summary(
            {"trajectory_id": "t", "sample_ids": [7.5], "real_batch_size": 1}
        )


def test_trace_aggregate_reports_spearman_improvement_and_stages():
    events = [
        {
            "solver_stage": "predictor",
            "delta_y": 0.1,
            "actual_full_residual_change": 0.2,
            "zero_order_relative_error": 0.4,
            "dcta_relative_error": 0.2,
        },
        {
            "solver_stage": "corrector",
            "delta_y": 0.2,
            "actual_full_residual_change": 0.4,
            "zero_order_relative_error": 0.6,
            "dcta_relative_error": 0.3,
        },
    ]
    report = aggregate_trace_rows(
        [
            {
                "trajectory_id": "t",
                "direct_full_count": 1,
                "resumed_full_count": 1,
                "reuse_count": 0,
                "cfg_action_disagreement_rate": 0.0,
                "cache_bytes": 4,
                "stream_trace": events,
            }
        ]
    )
    shadow = report["shadow_diagnostics"]
    assert shadow["probe_full_spearman"] == 1.0
    assert shadow["dcta_relative_improvement"] == 0.5
    assert shadow["dcta_better_fraction"] == 1.0
    assert set(shadow["by_solver_stage"]) == {"corrector", "predictor"}
