from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_dicache_trace.py"


def test_entrypoint_carries_top_level_candidate_identity_into_report(tmp_path: Path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    identity = {
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
    row = {
        "trajectory_id": "trajectory-0",
        "trajectory_direct_full_count": 1,
        "trajectory_resumed_full_count": 1,
        "trajectory_reuse_count": 1,
        "trajectory_dcta_count": 1,
        "trajectory_call_count_valid": True,
        **identity,
    }
    (metadata / "rank_0.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = tmp_path / "trace.json"
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)})
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metadata-dir",
            str(metadata),
            "--world-size",
            "1",
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    for field, value in identity.items():
        assert report[f"{field}_values"] == [value]

    mismatched = {**row, "checkpoint_sha256": "5" * 64}
    (metadata / "rank_0.jsonl").write_text(
        "\n".join(json.dumps(value, sort_keys=True) for value in (row, mismatched))
        + "\n",
        encoding="utf-8",
    )
    rejected = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metadata-dir",
            str(metadata),
            "--world-size",
            "1",
            "--output-json",
            str(tmp_path / "mixed.json"),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert rejected.returncode != 0
    assert "mixed candidate identity" in rejected.stderr
