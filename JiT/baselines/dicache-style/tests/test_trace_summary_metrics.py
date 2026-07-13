import pytest

from dicache_style.trace import aggregate_trace_rows


def test_trace_aggregate_exposes_required_scheduler_timing_and_memory_metrics():
    row = {
        "trajectory_id": "t",
        "direct_full_count": 2,
        "resumed_full_count": 1,
        "reuse_count": 3,
        "probe_depth": 1,
        "probe_count": 4,
        "dcta_count": 2,
        "zero_order_fallback_count": 1,
        "gamma_clip_min_count": 1,
        "gamma_clip_max_count": 2,
        "gamma_nonfinite_count": 0,
        "mean_gamma": 1.2,
        "p95_gamma": 1.4,
        "mean_accumulated_error": 0.1,
        "p95_accumulated_error": 0.3,
        "mean_refresh_gap": 2.0,
        "p95_refresh_gap": 3.0,
        "max_refresh_gap": 4,
        "probe_time_ms": 2.0,
        "gate_time_ms": 1.0,
        "scalar_sync_time_ms": 1.0,
        "dcta_time_ms": 1.0,
        "suffix_time_ms": 3.0,
        "cache_io_time_ms": 2.0,
        "cache_bytes": 100,
        "cache_tensor_count": 12,
        "peak_memory_allocated": 200,
        "peak_memory_reserved": 300,
        "call_count_valid": True,
    }
    report = aggregate_trace_rows([row])
    assert report["full_ratio"] == pytest.approx(0.5)
    assert report["reuse_ratio"] == pytest.approx(0.5)
    assert report["dcta_per_reuse_ratio"] == pytest.approx(2 / 3)
    assert report["zero_order_per_reuse_ratio"] == pytest.approx(1 / 3)
    assert report["gamma_clip_max_per_dcta_ratio"] == 1.0
    assert report["probe_runtime_ratio_of_component_host_time"] == pytest.approx(0.4)
    assert report["p95_refresh_gap_over_trajectory_p95"] == 3.0
    assert report["max_peak_memory_reserved"] == 300
    assert report["all_call_counts_valid"] is True


def test_trace_means_are_weighted_by_underlying_event_counts():
    base = {
        "direct_full_count": 1,
        "resumed_full_count": 0,
        "reuse_count": 0,
        "call_count_valid": True,
    }
    rows = [
        {
            **base,
            "trajectory_id": "short",
            "gamma_value_count": 1,
            "gamma_value_sum": 1.0,
            "accumulated_error_value_count": 1,
            "accumulated_error_value_sum": 1.0,
            "refresh_gap_value_count": 1,
            "refresh_gap_value_sum": 1.0,
        },
        {
            **base,
            "trajectory_id": "long",
            "gamma_value_count": 3,
            "gamma_value_sum": 6.0,
            "accumulated_error_value_count": 3,
            "accumulated_error_value_sum": 6.0,
            "refresh_gap_value_count": 3,
            "refresh_gap_value_sum": 6.0,
        },
    ]
    report = aggregate_trace_rows(rows)
    assert report["mean_gamma"] == pytest.approx(1.75)
    assert report["mean_accumulated_error"] == pytest.approx(1.75)
    assert report["mean_refresh_gap"] == pytest.approx(1.75)
    assert report["gamma_value_count"] == 4
    assert report["aggregation_semantics"]["p95_fields"].startswith("P95 over")


def test_trace_aggregate_persists_single_candidate_identity():
    row = {
        "trajectory_id": "identity",
        "direct_full_count": 1,
        "resumed_full_count": 0,
        "reuse_count": 1,
        "call_count_valid": True,
        "profile": "flux_image_released",
        "probe_depth": 1,
        "error_choice": "delta_y",
        "rel_l1_thresh": 0.125,
        "gamma_nonfinite_policy": "force_full",
        "config_hash": "1" * 64,
        "dicache_config_hash": "2" * 64,
        "manifest_sha256": "3" * 64,
        "checkpoint_path": "/immutable/checkpoint.pt",
        "checkpoint_size": 123,
        "checkpoint_sha256": "4" * 64,
        "method": "dicache",
    }
    report = aggregate_trace_rows([row])
    expected = {
        "profile_values": ["flux_image_released"],
        "probe_depth_values": [1],
        "error_choice_values": ["delta_y"],
        "rel_l1_thresh_values": [0.125],
        "gamma_nonfinite_policy_values": ["force_full"],
        "config_hash_values": ["1" * 64],
        "dicache_config_hash_values": ["2" * 64],
        "manifest_sha256_values": ["3" * 64],
        "checkpoint_path_values": ["/immutable/checkpoint.pt"],
        "checkpoint_size_values": [123],
        "checkpoint_sha256_values": ["4" * 64],
        "method_values": ["dicache"],
    }
    for field, value in expected.items():
        assert report[field] == value
