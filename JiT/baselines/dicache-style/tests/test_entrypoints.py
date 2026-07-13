from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
import yaml

from dicache_style.image_io import atomic_write_png, validate_outputs
from dicache_style.manifest import build_manifest
from dicache_style.metadata import DICACHE_CONFIG_FIELDS


ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    value["CUDA_VISIBLE_DEVICES"] = ""
    value["PYTHONPATH"] = str(ROOT)
    return value


def test_generate_preflight_is_cpu_only_and_maps_all_fields():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_shard.py"),
            "--config",
            str(ROOT / "configs" / "jit_b16_256_instrumented_full.yaml"),
            "--preflight",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_environment(),
    )
    report = json.loads(result.stdout)
    assert report["cuda_touched"] is False
    assert report["expected_nfe"] == 99
    assert report["expected_network_forwards"] == 198
    assert report["constructor_kwargs"]["mode"] == "instrumented_full"


def test_unresolved_main_config_fails_before_cuda_guard():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_shard.py"),
            "--config",
            str(ROOT / "configs" / "jit_b16_256_dicache.yaml"),
            "--preflight",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_environment(),
    )
    assert result.returncode != 0
    assert "search parameters must be materialized" in result.stderr


def test_generate_preflight_rejects_non_ema1_config(tmp_path: Path):
    config = yaml.safe_load(
        (ROOT / "configs" / "jit_b16_256_instrumented_full.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["model"]["ema"] = "model_ema2"
    path = tmp_path / "ema2.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_shard.py"),
            "--config",
            str(path),
            "--preflight",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_environment(),
    )
    assert result.returncode != 0
    assert "model.ema=model_ema1" in result.stderr


def test_output_validation_uses_dicache_metadata_fields(tmp_path: Path):
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
    with (ROOT / "configs" / "jit_b16_256_instrumented_full.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
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
            "trajectory_total_stream_calls": 2,
            "trajectory_direct_full_count": 2,
            "trajectory_resumed_full_count": 0,
            "trajectory_reuse_count": 0,
            "trajectory_network_forward_count": 2,
            "trajectory_expected_network_forward_count": 2,
            "trajectory_mean_delta_y": None,
            "trajectory_stream_trace": [{"delta_y": None, "gamma": None}],
            **{field: config["dicache"][field] for field in DICACHE_CONFIG_FIELDS},
        }
    }
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

    missing_gate_identity = deepcopy(metadata)
    missing_gate_identity[0].pop("release_gate_sha256")
    with pytest.raises(ValueError, match="missing release_gate_sha256"):
        validate_outputs(
            sample_dir,
            records,
            metadata=missing_gate_identity,
            expected_count=1,
            expected_per_class=1,
            expected_num_classes=1,
            resolution=4,
        )

    for field, bad_value, match in (
        ("trajectory_sample_ids", [99], "sample IDs mismatch"),
        ("trajectory_call_count_valid", False, "call_count_valid"),
        ("trajectory_network_forward_count", 1, "forward counts differ"),
        ("trajectory_mean_delta_y", float("inf"), "non-finite"),
    ):
        bad_metadata = deepcopy(metadata)
        bad_metadata[0][field] = bad_value
        with pytest.raises(ValueError, match=match):
            validate_outputs(
                sample_dir,
                records,
                metadata=bad_metadata,
                expected_count=1,
                expected_per_class=1,
                expected_num_classes=1,
                resolution=4,
            )
