from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from speca_style.pixelgen_lightning import (
    batch_group_id_from_metadata,
    trajectory_id_from_metadata,
)


def _load_aggregator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_speca_trace.py"
    spec = importlib.util.spec_from_file_location("pixelgen_speca_trace_aggregator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_group_identity_is_global_and_resume_stable():
    first = ({"sample_id": 0, "batch_group_id": "0:17"},)
    resumed = {"sample_id": [0], "batch_group_id": ["0:17"]}
    other_shard = ({"sample_id": 1, "batch_group_id": "1:17"},)

    assert batch_group_id_from_metadata(first, batch_size=1, batch_idx=0) == "0:17"
    assert trajectory_id_from_metadata(first, batch_size=1, batch_idx=0) == (
        "manifest-group:0:17"
    )
    assert trajectory_id_from_metadata(resumed, batch_size=1, batch_idx=0) == (
        "manifest-group:0:17"
    )
    assert trajectory_id_from_metadata(other_shard, batch_size=1, batch_idx=0) == (
        "manifest-group:1:17"
    )


def test_manifest_group_identity_rejects_mixed_or_missing_groups():
    with pytest.raises(ValueError, match="mixes manifest batch groups"):
        trajectory_id_from_metadata(
            (
                {"sample_id": 0, "batch_group_id": "0:0"},
                {"sample_id": 4, "batch_group_id": "0:1"},
            ),
            batch_size=2,
            batch_idx=3,
        )
    with pytest.raises(ValueError, match="missing or non-string"):
        trajectory_id_from_metadata(
            ({"sample_id": 0},), batch_size=1, batch_idx=0
        )


def test_aggregator_accepts_only_identical_same_group_repeats():
    module = _load_aggregator_module()
    registry = {}
    first = {
        "trajectory_id": "manifest-group:0:2",
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
