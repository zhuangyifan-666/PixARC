from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from dicache_style.image_io import atomic_write_png, resumable_batch_groups
from dicache_style.manifest import build_manifest
from dicache_style.metadata import (
    UNRELEASED_RELEASE_GATE,
    archived_release_gate_sha256,
    validate_archived_release_gate,
)


def test_archived_release_gate_digest_and_proxy_sentinel(tmp_path: Path):
    assert archived_release_gate_sha256(tmp_path) == UNRELEASED_RELEASE_GATE
    assert (
        validate_archived_release_gate(tmp_path, UNRELEASED_RELEASE_GATE)
        == UNRELEASED_RELEASE_GATE
    )

    gate = tmp_path / "release_gate.json"
    gate.write_bytes(b'{"status":"released"}\n')
    digest = hashlib.sha256(gate.read_bytes()).hexdigest()
    assert archived_release_gate_sha256(tmp_path) == digest
    assert validate_archived_release_gate(tmp_path, digest) == digest
    with pytest.raises(ValueError, match="unreleased run identity"):
        validate_archived_release_gate(tmp_path, UNRELEASED_RELEASE_GATE)

    gate.write_bytes(b'{"status":"tampered"}\n')
    with pytest.raises(ValueError, match="differs from run identity"):
        validate_archived_release_gate(tmp_path, digest)


def test_resume_binds_existing_rows_to_release_gate_digest(tmp_path: Path):
    records = build_manifest(
        samples_per_class=1,
        base_seed=7,
        split_name="fixture",
        world_size=1,
        batch_size=1,
        num_classes=1,
    )
    record = records[0]
    sample_dir = tmp_path / "samples"
    atomic_write_png(
        np.zeros((4, 4, 3), dtype=np.uint8),
        sample_dir / "000000.png",
        resolution=4,
    )
    metadata = {
        record.sample_id: {
            "class_id": record.class_id,
            "seed": record.seed,
            "batch_group_id": record.batch_group_id,
            "position_in_batch": record.position_in_batch,
            "manifest_sha256": "manifest",
            "config_hash": "config",
            "dicache_config_hash": "dicache",
            "release_gate_sha256": "a" * 64,
            "checkpoint_path": "/checkpoint",
            "checkpoint_size": 1,
            "method": "dicache",
            "real_batch_size": 1,
            "effective_cfg_batch_size": 2,
        }
    }
    pending, skipped = resumable_batch_groups(
        records,
        0,
        sample_dir,
        metadata,
        manifest_sha256="manifest",
        config_hash="config",
        dicache_config_hash="dicache",
        release_gate_sha256="a" * 64,
        checkpoint_path="/checkpoint",
        checkpoint_size=1,
        method="dicache",
        resolution=4,
    )
    assert pending == []
    assert skipped == [record.batch_group_id]

    with pytest.raises(RuntimeError, match="release_gate_sha256"):
        resumable_batch_groups(
            records,
            0,
            sample_dir,
            metadata,
            manifest_sha256="manifest",
            config_hash="config",
            dicache_config_hash="dicache",
            release_gate_sha256="b" * 64,
            checkpoint_path="/checkpoint",
            checkpoint_size=1,
            method="dicache",
            resolution=4,
        )
