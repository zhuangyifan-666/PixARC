from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from dicache_style.image_io import atomic_write_png, validate_outputs
from dicache_style.manifest import build_manifest
from dicache_style.metadata import DICACHE_CONFIG_FIELDS


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path):
    records = build_manifest(
        samples_per_class=1,
        base_seed=10,
        split_name="fixture",
        world_size=1,
        batch_size=1,
        num_classes=1,
    )
    sample_dir = tmp_path / "samples"
    atomic_write_png(
        np.zeros((4, 4, 3), dtype=np.uint8),
        sample_dir / "000000.png",
        resolution=4,
    )
    config = yaml.safe_load(
        (ROOT / "configs" / "pixelgen_xl_256_instrumented_full.yaml").read_text(
            encoding="utf-8"
        )
    )
    record = records[0]
    metadata = {
        0: {
            "sample_id": 0,
            "class_id": record.class_id,
            "seed": record.seed,
            "batch_group_id": record.batch_group_id,
            "position_in_batch": record.position_in_batch,
            "status": "ok",
            "config_hash": "config",
            "dicache_config_hash": "dicache",
            "release_gate_sha256": "unreleased",
            "checkpoint_path": "/checkpoint",
            "checkpoint_size": 1,
            "manifest_sha256": "manifest",
            "method": "instrumented_full",
            "real_batch_size": 1,
            "effective_cfg_batch_size": 2,
            "trajectory_id": "group-0",
            "trajectory_sample_ids": [0],
            "trajectory_call_count_valid": True,
            "trajectory_total_nfe": 1,
            "trajectory_total_stream_calls": 1,
            "trajectory_direct_full_count": 1,
            "trajectory_resumed_full_count": 0,
            "trajectory_reuse_count": 0,
            "trajectory_network_forward_count": 1,
            "trajectory_expected_network_forward_count": 1,
            "trajectory_mean_delta_y": None,
            "trajectory_stream_trace": [{"delta_y": None, "gamma": None}],
            **{
                field: config["dicache"][field]
                for field in DICACHE_CONFIG_FIELDS
            },
        }
    }
    return sample_dir, records, metadata


def test_validate_outputs_accepts_bound_finite_trajectory(tmp_path: Path):
    sample_dir, records, metadata = _fixture(tmp_path)
    report = validate_outputs(
        sample_dir,
        records,
        metadata=metadata,
        expected_count=1,
        expected_per_class=1,
        expected_num_classes=1,
        resolution=4,
    )
    assert report["profile"] == "flux_image_released"
    assert report["release_gate_sha256"] == "unreleased"


def test_validate_outputs_requires_release_prefix_identity(tmp_path: Path):
    sample_dir, records, metadata = _fixture(tmp_path)
    metadata[0].pop("release_gate_sha256")
    with pytest.raises(ValueError, match="missing release_gate_sha256"):
        validate_outputs(
            sample_dir,
            records,
            metadata=metadata,
            expected_count=1,
            expected_per_class=1,
            expected_num_classes=1,
            resolution=4,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trajectory_sample_ids", [1], "sample IDs mismatch"),
        ("trajectory_call_count_valid", False, "call_count_valid"),
        ("trajectory_expected_network_forward_count", 2, "forward counts differ"),
        ("trajectory_stream_trace", [{"gamma": float("nan")}], "non-finite"),
    ],
)
def test_validate_outputs_rejects_corrupt_trajectory(
    tmp_path: Path, field: str, value: object, message: str
):
    sample_dir, records, metadata = _fixture(tmp_path)
    bad = deepcopy(metadata)
    bad[0][field] = value
    with pytest.raises(ValueError, match=message):
        validate_outputs(
            sample_dir,
            records,
            metadata=bad,
            expected_count=1,
            expected_per_class=1,
            expected_num_classes=1,
            resolution=4,
        )
