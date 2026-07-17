from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _selection_decision(path: Path, threshold: float, policy: str) -> Path:
    paired = path.parent / "paired.json"
    trace = path.parent / "trace.json"
    benchmark = path.parent / "benchmark.json"
    manifest_sha = "a" * 64
    checkpoint_sha = "c" * 64
    checkpoint_path = "/immutable/checkpoint.pt"
    candidate = {
        "mode": "dicache", "profile": "flux_image_released",
        "probe_depth": 1, "error_choice": "delta_y",
        "rel_l1_thresh": threshold, "gamma_nonfinite_policy": policy,
    }
    benchmark_config = {"dicache": candidate}
    benchmark_config_path = path.parent / "benchmark_candidate.yaml"
    benchmark_config_path.write_text(yaml.safe_dump(benchmark_config), encoding="utf-8")
    config_hash = hashlib.sha256(json.dumps(
        benchmark_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    dicache_hash = hashlib.sha256(json.dumps(
        candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    _write_json(paired, {
        "sample_count": 8000,
        "reference_manifest_sha256": manifest_sha,
        "candidate_manifest_sha256": manifest_sha,
        "candidate_dicache_config": candidate,
        "candidate_input_config_hash": config_hash,
        "candidate_dicache_config_hash": dicache_hash,
        "candidate_method": "dicache",
        "candidate_run_identity": {
            "manifest_sha256": manifest_sha,
            "checkpoint_path": checkpoint_path,
            "checkpoint_size": 123,
            "checkpoint_sha256": checkpoint_sha,
        },
        "aggregate_mse": 0.01, "psnr_from_aggregate_mse": 20.0,
        "per_image_psnr": {key: 20.0 for key in ("mean", "median", "p90", "p95", "p99")},
        "ssim": {key: 0.9 for key in ("mean", "median", "p90", "p95", "p99")},
        "exact_pair_count": 0,
        "nan_counts": {"psnr": 0, "ssim": 0, "lpips": 0},
        "inf_counts": {"psnr": 0, "ssim": 0, "lpips": 0},
        "ssim_protocol": {"channel_axis": -1, "data_range": 1.0, "win_size": 7},
        "lpips": {
            **{key: 0.1 for key in ("mean", "median", "p90", "p95", "p99")},
            "max": 0.2, "value_count": 8000, "backbone": "alex",
            "spatial": False, "package_version": "fixture",
        },
    })
    _write_json(trace, {
        "trajectory_count": 8000, "profile_values": ["flux_image_released"],
        "probe_depth_values": [1], "error_choice_values": ["delta_y"],
        "rel_l1_thresh_values": [threshold],
        "gamma_nonfinite_policy_values": [policy],
        "config_hash_values": [config_hash],
        "dicache_config_hash_values": [dicache_hash],
        "manifest_sha256_values": [manifest_sha],
        "checkpoint_path_values": [checkpoint_path],
        "checkpoint_size_values": [123],
        "checkpoint_sha256_values": [checkpoint_sha],
        "method_values": ["dicache"],
        "all_call_counts_valid": True, "direct_full_count": 1,
        "resumed_full_count": 0, "reuse_count": 1, "dcta_count": 1,
    })
    _write_json(benchmark, {
        "protocol": {
            "batch_size": 32, "compile_mode": "matched_eager",
            "input_config": str(benchmark_config_path.resolve()),
            "input_config_hash": config_hash,
            "dicache_config_hash": dicache_hash,
            "manifest_sha256": manifest_sha,
            "checkpoint": checkpoint_path, "checkpoint_size": 123,
            "checkpoint_sha256": checkpoint_sha, "candidate_mode": "dicache",
        },
        "full": {"raw_ms_per_image": [2.0] * 30, "warmup_batches": 10, "measured_batches": 30},
        "dicache": {"raw_ms_per_image": [1.0] * 30, "warmup_batches": 10, "measured_batches": 30},
    })
    evidence = {
        name: {"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        for name, source in (("paired", paired), ("trace", trace), ("benchmark", benchmark))
    }
    decision = path.parent / "decision.json"
    _write_json(decision, {
        "schema_version": "pixarc-dicache-selection-decision-v1", "passed": True,
        "status": "selected", "model_family": "JiT",
        "profile": "flux_image_released", "rel_l1_thresh": threshold,
        "gamma_nonfinite_policy": policy, "selection_rule": "quality_floor",
        "validation_sample_count": 8000, "final_50k_used_for_selection": False,
        "evidence": evidence,
    })
    return decision


def _selection(
    path: Path, *, status: str, threshold: float, policy: str, probe_depth: int = 1
) -> None:
    decision = _selection_decision(path, threshold, policy) if status == "selected" else None
    path.write_text(
        json.dumps(
            {
                "schema_version": "pixarc-dicache-selection-v1",
                "passed": True,
                "status": status,
                "model_family": "JiT",
                "profile": "flux_image_released",
                "probe_depth": probe_depth,
                "batch_size": 32,
                "rel_l1_thresh": threshold,
                "gamma_nonfinite_policy": policy,
                "final_50k_used_for_selection": False,
                "decision": (
                    {
                        "path": str(decision.resolve()),
                        "sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
                        "schema_version": "pixarc-dicache-selection-decision-v1",
                        "passed": True,
                    }
                    if decision is not None else None
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_materializer_writes_absolute_checkpoint_and_resolves_placeholders(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    selection = tmp_path / "selection.json"
    _selection(selection, status="selected", threshold=0.25, policy="force_full")
    output = tmp_path / "resolved.yaml"
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)})
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_dicache_config.py"),
            "--input",
            str(ROOT / "configs" / "jit_b16_256_dicache.yaml"),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--rel-l1-thresh",
            "0.25",
            "--gamma-nonfinite-policy",
            "force_full",
            "--selection-report",
            str(selection),
        ],
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["model"]["checkpoint"] == str(checkpoint.resolve())
    assert config["dicache"]["rel_l1_thresh"] == 0.25
    assert config["dicache"]["gamma_nonfinite_policy"] == "force_full"
    assert config["selection_provenance"]["status"] == "selected"
    assert config["selection_provenance"]["threshold_selected_before_final_50k"] is True


def test_materializer_supports_shadow_probe_depth_only(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    selection = tmp_path / "selection.json"
    _selection(
        selection, status="provisional", threshold=0.25,
        policy="force_full", probe_depth=3,
    )
    output = tmp_path / "shadow.yaml"
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)})
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_dicache_config.py"),
            "--input",
            str(ROOT / "configs" / "jit_b16_256_probe_shadow_full.yaml"),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--rel-l1-thresh",
            "0.25",
            "--gamma-nonfinite-policy",
            "force_full",
            "--probe-depth",
            "3",
            "--selection-report",
            str(selection),
        ],
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["dicache"]["probe_depth"] == 3
    assert config["selection_provenance"]["status"] == "provisional"
    assert config["selection_provenance"]["threshold_selected_before_final_50k"] is False
