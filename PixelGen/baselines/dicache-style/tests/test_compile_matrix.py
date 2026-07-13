from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch
import pytest
import yaml

from dicache_style.latency import tensor_fingerprint
from dicache_style.pixelgen_benchmark import (
    COMPILE_MATRIX_ROWS,
    validate_compile_role_config,
)
from dicache_style.source_identity import (
    release_source_bindings,
    require_source_identity_current,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_compile_matrix.py"
SPEC = importlib.util.spec_from_file_location("pixelgen_compile_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)
SOURCE = release_source_bindings(
    ROOT, ROOT.parents[2] / "third-party" / "PixelGen"
)


def test_tensor_fingerprint_is_byte_exact_and_device_independent():
    original = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.uint8)
    identical = original.clone()
    changed = original.clone()
    changed[0, 0, 0, 0] = 4
    first = tensor_fingerprint(original)
    second = tensor_fingerprint(identical)
    third = tensor_fingerprint(changed)
    assert first == second
    assert first["sha256"] != third["sha256"]
    assert first["shape"] == [1, 1, 2, 2]
    assert first["finite"] is True
    assert tensor_fingerprint(torch.tensor(1.0))["byte_count"] == 4
    assert tensor_fingerprint(torch.ones(2, dtype=torch.bfloat16))[
        "byte_count"
    ] == 4


def test_compile_role_validation_rejects_cross_surface_mismatch():
    source = ROOT / "configs" / "pixelgen_xl_256_upstream_full.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["runtime"]["compile_mode"] = "upstream"
    config["model"]["compile_mode"] = "upstream"
    config["model"]["denoiser"]["init_args"]["compile_mode"] = "upstream"
    assert validate_compile_role_config(config, "upstream_whole_model") == (
        "upstream_full",
        "upstream",
        "full",
    )
    config["model"]["compile_mode"] = "blockwise"
    try:
        validate_compile_role_config(config, "upstream_whole_model")
    except ValueError as exc:
        assert "config surfaces disagree" in str(exc)
    else:
        raise AssertionError("cross-surface compile mismatch was accepted")


def _row(role: str, digest: str) -> dict[str, object]:
    config_mode, compile_mode, _selector = COMPILE_MATRIX_ROWS[role]
    expected = 7
    protocol = {
        "input_config": f"/{role}.yaml",
        "input_config_sha256": f"config-{role}",
        "input_config_hash": f"canonical-{role}",
        "model": "PixelGen-JiT",
        "model_config_hash": "model",
        "checkpoint": "/checkpoint.ckpt",
        "checkpoint_size": 123,
        "checkpoint_sha256": "checkpoint-sha",
        "ema": "ema_denoiser",
        "sampler": "exact_heun",
        "sampler_config_hash": "sampler",
        "steps": 4,
        "cfg_scale": 2.25,
        "guidance_interval": [0.1, 0.9],
        "timeshift": 2.0,
        "sample_ids": [0],
        "seeds": [1234],
        "class_ids": [0],
        "manifest_sha256": "manifest",
        "noise_scale": 1.0,
        "cfg_execution": "single combined [unconditional, conditional] 2B forward",
        "expected_network_forward_count": expected,
        "batch_size": 1,
        "effective_cfg_batch_size": 2,
        "dtype": "bf16-mixed",
        "compile_mode": compile_mode,
        "config_mode": config_mode,
        "dicache_config_hash": (
            "candidate-dicache" if config_mode == "dicache" else f"{config_mode}-dicache"
        ),
    }
    measurement = {
        "mean_ms_per_image": 2.0,
        "median_ms_per_image": 2.0,
        "p95_ms_per_image": 2.1,
        "first_execution_upper_bound_seconds": 3.0,
        "peak_memory_allocated": 100,
        "peak_memory_reserved": 120,
        "first_execution_peak_memory_allocated": 110,
        "first_execution_peak_memory_reserved": 130,
        "graph_break_count": 1,
        "recompile_guard_failure_count": 2,
        "dynamo_counter_delta": {"graph_break.test": 1},
        "final_output": {
            "sha256": digest,
            "shape": [1, 3, 4, 4],
            "dtype": "torch.uint8",
            "byte_count": 48,
            "finite": True,
        },
    }
    validation = {
        "sample_finite": True,
        "decoded_image_finite": True,
        "summary": {
            "network_forward_count": expected,
            "call_count_valid": True,
        },
        "runtime_active_after": False,
        "cache_bytes_after": 0,
        "cache_tensor_count_after": 0,
    }
    return {
        "schema_version": "pixarc-dicache-compile-row-v1",
        "role": role,
        "passed": True,
        "source": copy.deepcopy(SOURCE),
        "protocol": protocol,
        "measurement": measurement,
        "validation": validation,
    }


def test_matrix_artifact_passes_only_all_exact_and_operational_gates():
    reports = {
        role: _row(role, "full" if role.endswith("full") or role.startswith("upstream") else "candidate")
        for role in COMPILE_MATRIX_ROWS
    }
    report = MATRIX.assemble_matrix(reports)
    assert report["passed"] is True
    assert len(report["matrix"]) == 5
    for row in report["matrix"].values():
        assert "graph_break_count" in row
        assert "recompile_guard_failure_count" in row
        assert "first_execution_upper_bound_seconds" in row

    mismatched = copy.deepcopy(reports)
    mismatched["blockwise_dicache"]["measurement"]["final_output"][
        "sha256"
    ] = "different"
    failed = MATRIX.assemble_matrix(mismatched)
    assert failed["passed"] is False
    assert failed["correctness"][
        "matched_eager_vs_blockwise_dicache"
    ]["passed"] is False


def test_matrix_rejects_row_source_identity_drift():
    reports = {
        role: _row(
            role,
            "full"
            if role.endswith("full") or role.startswith("upstream")
            else "candidate",
        )
        for role in COMPILE_MATRIX_ROWS
    }
    reports["blockwise_dicache"]["source"]["upstream"]["sha256"] = "0" * 64
    report = MATRIX.assemble_matrix(reports, expected_source=SOURCE)
    assert report["passed"] is False
    assert set(report["source_mismatches"]) == {"blockwise_dicache"}
    assert report["source"] == SOURCE


def test_evidence_source_snapshot_rejects_generation_time_drift(tmp_path: Path):
    baseline = tmp_path / "baseline"
    port_package = baseline / "dicache_style"
    port_package.mkdir(parents=True)
    executable = port_package / "worker.py"
    executable.write_text("VERSION = 1\n", encoding="utf-8")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
    captured = release_source_bindings(baseline, upstream)
    executable.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during fixture evidence"):
        require_source_identity_current(
            captured,
            baseline,
            upstream,
            context="fixture evidence",
        )


def test_compile_row_and_aggregator_snapshot_before_work_and_verify_before_write():
    latency_path = ROOT / "dicache_style" / "latency.py"
    latency_source = latency_path.read_text(encoding="utf-8")
    matrix_source = SCRIPT.read_text(encoding="utf-8")
    assert latency_source.index("source = release_source_bindings(") < latency_source.index(
        "factory = load_factory(arguments.runner_factory)"
    )
    assert latency_source.rindex("require_source_identity_current(") < latency_source.index(
        "atomic_write_json(arguments.output_json, result)"
    )
    matrix_capture = matrix_source.index("matrix_source = release_source_bindings(")
    assert matrix_capture < matrix_source.index("for role in ROLE_ORDER:", matrix_capture)
    assert matrix_source.rindex("require_source_identity_current(") < matrix_source.rindex(
        "atomic_write_json(output_json, report)"
    )


def test_matrix_row_nonfinite_or_bad_count_is_fail_closed():
    reports = {
        role: _row(role, "full" if role.endswith("full") or role.startswith("upstream") else "candidate")
        for role in COMPILE_MATRIX_ROWS
    }
    reports["matched_eager_dicache"]["validation"]["sample_finite"] = False
    reports["blockwise_full"]["validation"]["summary"][
        "network_forward_count"
    ] = 6
    report = MATRIX.assemble_matrix(reports)
    assert report["passed"] is False
    assert report["matrix"]["matched_eager_dicache"]["row_gate"][
        "checks"
    ]["raw_sample_finite"] is False
    assert report["matrix"]["blockwise_full"]["row_gate"]["checks"][
        "network_forward_count_matches_sampler"
    ] is False
