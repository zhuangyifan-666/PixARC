from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_aggregator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_speca_trace.py"
    spec = importlib.util.spec_from_file_location(
        "pixelgen_speca_trace_accounting", path
    )
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
