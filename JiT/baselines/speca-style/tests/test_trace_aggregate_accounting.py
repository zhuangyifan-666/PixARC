from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_aggregator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_speca_trace.py"
    spec = importlib.util.spec_from_file_location("jit_speca_trace_aggregator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inclusive_cache_io_is_not_added_to_predictor_and_history_again():
    module = _load_aggregator_module()
    verification, measured = module.diagnostic_time_totals(
        {
            "verification_block_time_ms": [1.0],
            "error_reduction_time_ms": [2.0],
            "scalar_sync_time_ms": [3.0],
            "predictor_time_ms": [10.0],
            "history_update_time_ms": [20.0],
            "cache_io_time_ms": [30.0],
            "scheduler_time_ms": [4.0],
        }
    )
    assert verification == 6.0
    assert measured == 40.0


def test_trajectory_id_collision_is_fail_closed():
    module = _load_aggregator_module()
    registry = {}
    first = {
        "trajectory_id": "shard-0-group-0:2",
        "batch_group_id": "0:2",
        "trajectory_total_nfe": 99,
        "trajectory_full_nfe": 12,
    }
    same_group_sample = {**first, "sample_id": 8, "position_in_batch": 1}
    other_group = {**first, "batch_group_id": "1:2"}
    changed_summary = {**first, "trajectory_full_nfe": 13}

    assert module.register_trajectory(registry, first) is True
    assert module.register_trajectory(registry, same_group_sample) is False
    with pytest.raises(ValueError, match="trajectory_id collision"):
        module.register_trajectory(registry, other_group)
    with pytest.raises(ValueError, match="trajectory_id collision"):
        module.register_trajectory(registry, changed_summary)
