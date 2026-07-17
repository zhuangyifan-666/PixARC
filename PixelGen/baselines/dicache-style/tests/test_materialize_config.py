from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_dicache_config.py"
CONFIGS = ROOT / "configs"


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
            "batch_size": 4, "compile_mode": "matched_eager",
            "input_config": str(benchmark_config_path.resolve()),
            "input_config_hash": config_hash,
            "dicache_config_hash": dicache_hash,
            "manifest_sha256": manifest_sha,
            "checkpoint": checkpoint_path, "checkpoint_size": 123,
            "checkpoint_sha256": checkpoint_sha, "config_mode": "dicache",
            "dicache": candidate,
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
        "status": "selected", "model_family": "PixelGen",
        "profile": "flux_image_released", "rel_l1_thresh": threshold,
        "gamma_nonfinite_policy": policy, "selection_rule": "quality_floor",
        "validation_sample_count": 8000, "final_50k_used_for_selection": False,
        "evidence": evidence,
    })
    return decision


def _run_materializer(
    tmp_path: Path, base_name: str, *extra: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "materialized.yaml"
    base = yaml.safe_load((CONFIGS / base_name).read_text(encoding="utf-8"))
    threshold = base["dicache"]["rel_l1_thresh"]
    policy = base["dicache"]["gamma_nonfinite_policy"]
    probe_depth = base["dicache"]["probe_depth"]
    for index, value in enumerate(extra):
        if value == "--threshold":
            threshold = float(extra[index + 1])
        elif value == "--gamma-nonfinite-policy":
            policy = extra[index + 1]
        elif value == "--probe-depth":
            probe_depth = int(extra[index + 1])
    selection = tmp_path / "selection.json"
    selected = base["dicache"]["mode"] == "dicache"
    decision = _selection_decision(selection, threshold, policy) if selected else None
    selection.write_text(
        json.dumps(
            {
                "schema_version": "pixarc-dicache-selection-v1",
                "passed": True,
                "status": (
                    "selected" if selected else "provisional"
                ),
                "model_family": "PixelGen",
                "profile": "flux_image_released",
                "probe_depth": probe_depth,
                "batch_size": 4,
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
    command = [
        sys.executable,
        str(SCRIPT),
        "--base",
        str(CONFIGS / base_name),
        "--checkpoint",
        str(checkpoint),
        "--selection-report",
        str(selection),
        *extra,
        "--output",
        str(output),
    ]
    return subprocess.run(command, capture_output=True, text=True), output


def test_compile_mode_is_synchronized_across_all_constructor_surfaces(tmp_path):
    completed, output = _run_materializer(
        tmp_path,
        "pixelgen_xl_256_dicache.yaml",
        "--threshold",
        "0.1",
        "--gamma-nonfinite-policy",
        "official_propagate",
        "--compile-mode",
        "blockwise",
    )
    assert completed.returncode == 0, completed.stderr
    value = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert value["runtime"]["compile_mode"] == "blockwise"
    assert value["model"]["compile_mode"] == "blockwise"
    assert value["model"]["denoiser"]["init_args"]["compile_mode"] == "blockwise"
    assert value["selection_provenance"]["status"] == "selected"
    assert json.loads(completed.stdout)["compile_mode"] == "blockwise"


def test_upstream_compile_is_restricted_to_upstream_full(tmp_path):
    completed, output = _run_materializer(
        tmp_path,
        "pixelgen_xl_256_dicache.yaml",
        "--threshold",
        "0.1",
        "--gamma-nonfinite-policy",
        "official_propagate",
        "--compile-mode",
        "upstream",
    )
    assert completed.returncode != 0
    assert not output.exists()
    assert "valid only for upstream_full" in completed.stderr


def test_upstream_full_can_materialize_upstream_compile(tmp_path):
    completed, output = _run_materializer(
        tmp_path,
        "pixelgen_xl_256_upstream_full.yaml",
        "--compile-mode",
        "upstream",
    )
    assert completed.returncode == 0, completed.stderr
    value = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert value["runtime"]["compile_mode"] == "upstream"
    assert value["model"]["compile_mode"] == "upstream"
    assert value["model"]["denoiser"]["init_args"]["compile_mode"] == "upstream"


@pytest.mark.parametrize(
    ("base_name", "depth"),
    [
        ("pixelgen_xl_256_probe_shadow_full.yaml", 3),
        ("pixelgen_xl_256_probe_only_ablation.yaml", 2),
    ],
)
def test_probe_depth_ablation_is_synchronized(tmp_path, base_name, depth):
    completed, output = _run_materializer(
        tmp_path,
        base_name,
        "--threshold",
        "0.1",
        "--probe-depth",
        str(depth),
    )
    assert completed.returncode == 0, completed.stderr
    value = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert value["dicache"]["probe_depth"] == depth
    assert value["model"]["denoiser"]["init_args"]["dicache_probe_depth"] == depth


def test_main_dicache_rejects_probe_depth_ablation(tmp_path):
    completed, output = _run_materializer(
        tmp_path,
        "pixelgen_xl_256_dicache.yaml",
        "--threshold",
        "0.1",
        "--gamma-nonfinite-policy",
        "official_propagate",
        "--probe-depth",
        "2",
    )
    assert completed.returncode != 0
    assert not output.exists()
    assert "restricted to probe_shadow_full or probe_only_ablation" in completed.stderr


def test_probe_depth_ablation_rejects_undeclared_depth(tmp_path):
    completed, output = _run_materializer(
        tmp_path,
        "pixelgen_xl_256_probe_shadow_full.yaml",
        "--threshold",
        "0.1",
        "--probe-depth",
        "4",
    )
    assert completed.returncode != 0
    assert not output.exists()
    assert "invalid choice: 4" in completed.stderr
