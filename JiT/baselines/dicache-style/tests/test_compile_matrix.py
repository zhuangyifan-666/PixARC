from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from dicache_style.jit_benchmark import validate_runner_config
from dicache_style.latency import _dynamo_delta, tensor_correctness, tensor_error_metrics
from dicache_style.manifest import build_manifest, write_manifest
from dicache_style.source_identity import (
    release_source_bindings,
    require_source_identity_current,
)


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "run_compile_mode_benchmark.py"
MATRIX_PATH = ROOT / "scripts" / "run_compile_matrix.py"
WRAPPER_PATH = ROOT / "scripts" / "run_compile_matrix.sh"


def _load_script(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


WORKER = _load_script("jit_compile_worker", WORKER_PATH)
MATRIX = _load_script("jit_compile_matrix", MATRIX_PATH)
SOURCE = release_source_bindings(ROOT, ROOT.parents[2] / "third-party" / "JiT")


def test_raw_tensor_correctness_is_finite_dtype_aware_and_tolerance_bound():
    reference = torch.tensor([1.0, -2.0], dtype=torch.float32)
    candidate = torch.tensor([1.125, -1.875], dtype=torch.float32)
    metrics = tensor_error_metrics(candidate, reference)
    assert metrics["max_absolute_error"] == pytest.approx(0.125)
    assert metrics["mean_absolute_error"] == pytest.approx(0.125)
    assert tensor_correctness(candidate, reference, atol=0.125, rtol=0.0)[
        "passed"
    ] is True
    assert tensor_correctness(candidate, reference, atol=0.0, rtol=0.0)[
        "passed"
    ] is False
    wrong_dtype = tensor_correctness(
        candidate.double(), reference, atol=1.0, rtol=1.0
    )
    assert wrong_dtype["dtype_matches"] is False
    assert wrong_dtype["passed"] is False
    nonfinite = tensor_correctness(
        torch.tensor([float("nan")]), torch.zeros(1), atol=0.0, rtol=0.0
    )
    assert nonfinite["candidate_finite"] is False
    assert nonfinite["max_absolute_error"] is None
    assert nonfinite["passed"] is False


def test_dynamo_delta_reports_graph_breaks_and_guard_recompiles():
    report = _dynamo_delta(
        ({"graph_break.reason": 2, "stats.calls_captured": 3}, 1),
        (
            {
                "graph_break.reason": 5,
                "stats.calls_captured": 4,
                "stats.unique_graphs_recompiled": 2,
            },
            5,
        ),
    )
    assert report["graph_break_count"] == 3
    assert report["recompile_guard_failure_count"] == 4
    assert report["guard_recompile_count"] == 4


def _measurement(runtime_mode: str) -> dict[str, object]:
    expected_nfe = 7
    expected_forwards = 2 * expected_nfe
    direct = expected_forwards if runtime_mode != "dicache" else 2
    reuse = 0 if runtime_mode != "dicache" else expected_forwards - direct
    return {
        "passed": True,
        "mode": runtime_mode,
        "total_nfe": expected_nfe,
        "total_stream_calls": expected_forwards,
        "network_forward_count": expected_forwards,
        "expected_network_forward_count": expected_forwards,
        "direct_full_count": direct,
        "resumed_full_count": 0,
        "reuse_count": reuse,
        "call_count_valid": True,
        "mean_ms_per_image": 2.0,
        "median_ms_per_image": 2.0,
        "p95_ms_per_image": 2.1,
        "first_execution_wall_seconds": 3.0,
        "first_execution_cuda_event_ms_per_image": 2.5,
        "first_execution_peak_memory_allocated": 110,
        "first_execution_peak_memory_reserved": 130,
        "peak_memory_allocated": 100,
        "peak_memory_reserved": 120,
        "graph_break_count": 1,
        "recompile_guard_failure_count": 2,
        "guard_recompile_count": 2,
        "dynamo_counter_delta": {"graph_break.reason": 1},
        "measured_batches": 3,
        "raw_ms_per_image": [1.9, 2.0, 2.1],
        "runtime_summary_last": {"mode": runtime_mode},
        "runtime_lifecycle": {
            "runtime_active_after": False,
            "cache_bytes_after": 0,
            "cache_tensor_count_after": 0,
        },
        "invariants": {"all_worker_invariants": True},
    }


def _protocol(mode: str) -> dict[str, object]:
    contract = MATRIX.MODE_CONTRACTS[mode]
    return {
        "input_config": "/config.yaml",
        "input_config_hash": "config",
        "model": "JiT-B/16",
        "model_config_hash": "model",
        "checkpoint": "/checkpoint.pth",
        "checkpoint_size": 123,
        "checkpoint_sha256": "checkpoint-sha",
        "ema": "EMA1",
        "sampler": "heun",
        "exact_heun": True,
        "sampler_config_hash": "sampler",
        "dicache_config_hash": "dicache",
        "steps": 4,
        "cfg_scale": 3.0,
        "guidance_interval": [0.1, 1.0],
        "sample_ids": list(range(32)),
        "seeds": list(range(123, 155)),
        "class_ids": [7] * 32,
        "manifest_sha256": "manifest-sha",
        "rel_l1_thresh": 0.1,
        "noise_scale": 1.0,
        "initial_noise_protocol": "CPU generator then GPU copy",
        "cfg_execution": "separate conditional streams",
        "compile_mode": contract["compile_mode"],
        "full_mode": contract["full_mode"],
        "compile_scope": contract["compile_scope"],
        "batch_size": 32,
        "effective_cfg_batch_size": 64,
        "dtype": "bfloat16",
        "expected_nfe": 7,
        "expected_network_forward_count": 14,
        "torch_logs": "graph_breaks,recompiles",
    }


def _mode_report(mode: str) -> dict[str, object]:
    contract = MATRIX.MODE_CONTRACTS[mode]
    roles = {"full": _measurement(contract["full_mode"])}
    if mode != "upstream":
        roles["candidate"] = _measurement("dicache")
    return {
        "schema_version": "pixarc-jit-compile-mode-worker-v1",
        "passed": True,
        "source": copy.deepcopy(SOURCE),
        "mode": mode,
        "compile_mode": contract["compile_mode"],
        "full_mode": contract["full_mode"],
        "expected_nfe": 7,
        "expected_network_forward_count": 14,
        "protocol": _protocol(mode),
        "roles": roles,
    }


def _outputs() -> dict[str, dict[str, torch.Tensor]]:
    full = torch.tensor([1.0, -1.0])
    candidate = torch.tensor([0.75, -0.5])
    return {
        "upstream": {"full": full.clone()},
        "matched_eager": {
            "full": full.clone(),
            "candidate": candidate.clone(),
        },
        "blockwise": {
            "full": full.clone(),
            "candidate": candidate.clone(),
        },
    }


def test_worker_role_gate_uses_configuration_derived_dual_stream_counts():
    measurement = _measurement("instrumented_full")
    checks = WORKER.validate_role_measurement(
        measurement,
        expected_mode="instrumented_full",
        expected_nfe=7,
        expected_forwards=14,
        require_all_full=True,
        runtime_state=measurement["runtime_lifecycle"],
    )
    assert all(checks.values())


def test_matrix_passes_only_all_modes_identity_and_raw_correctness():
    reports = {mode: _mode_report(mode) for mode in MATRIX.MODE_ORDER}
    outputs = _outputs()
    result = MATRIX.assemble_matrix(reports, outputs, atol=0.0, rtol=0.0)
    assert result["passed"] is True
    assert set(result["matrix"]) == set(MATRIX.MODE_ORDER)
    assert result["correctness"]["upstream_vs_matched_eager_full"][
        "exact_equal"
    ] is True

    changed = _outputs()
    changed["blockwise"]["candidate"][0] += 1.0
    failed = MATRIX.assemble_matrix(reports, changed, atol=0.0, rtol=0.0)
    assert failed["passed"] is False
    assert failed["correctness"][
        "matched_eager_vs_blockwise_candidate"
    ]["passed"] is False

    mismatched = copy.deepcopy(reports)
    mismatched["blockwise"]["protocol"]["sample_ids"] = [99]
    failed_identity = MATRIX.assemble_matrix(
        mismatched, outputs, atol=0.0, rtol=0.0
    )
    assert failed_identity["passed"] is False
    assert "sample_ids" in failed_identity["identity_mismatches"]["blockwise"]


def test_matrix_rejects_worker_source_identity_drift():
    reports = {mode: _mode_report(mode) for mode in MATRIX.MODE_ORDER}
    reports["blockwise"]["source"]["port"]["sha256"] = "0" * 64
    result = MATRIX.assemble_matrix(
        reports,
        _outputs(),
        atol=0.0,
        rtol=0.0,
        expected_source=SOURCE,
    )
    assert result["passed"] is False
    assert set(result["source_mismatches"]) == {"blockwise"}
    assert result["source"] == SOURCE


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


def test_compile_worker_and_aggregator_snapshot_before_work_and_verify_before_write():
    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    matrix_source = MATRIX_PATH.read_text(encoding="utf-8")
    assert "source = release_source_bindings(BASELINE_ROOT, UPSTREAM_ROOT)" in worker_source
    assert worker_source.index("source = release_source_bindings") < worker_source.index(
        "build_benchmark_spec(runner)"
    )
    assert worker_source.rindex("require_source_identity_current(") < worker_source.index(
        "atomic_write_json(arguments.output_json, report)"
    )
    matrix_capture = matrix_source.index("matrix_source = release_source_bindings(")
    assert matrix_capture < matrix_source.index("for mode in MODE_ORDER:", matrix_capture)
    assert matrix_source.rindex("require_source_identity_current(") < matrix_source.rindex(
        "atomic_write_json(output_json, report)"
    )


def test_runner_override_routes_only_legal_full_modes(tmp_path: Path):
    config = yaml.safe_load(
        (ROOT / "configs" / "jit_b16_256_dicache.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["dicache"]["rel_l1_thresh"] = 0.1
    config["dicache"]["gamma_nonfinite_policy"] = "force_full"
    manifest = tmp_path / "benchmark.jsonl"
    records = build_manifest(
        samples_per_class=32,
        base_seed=123,
        split_name="benchmark",
        world_size=1,
        batch_size=32,
        num_classes=1,
    )
    write_manifest(
        records,
        manifest,
        base_seed=123,
        world_size=1,
        batch_size=32,
        generator_device="cpu",
    )
    runner = {
        "batch_size": 32,
        "sample_ids": [record.sample_id for record in records],
        "seeds": [record.seed for record in records],
        "class_ids": [record.class_id for record in records],
        "manifest": str(manifest),
        "batch_group_id": records[0].batch_group_id,
        "compile_mode_override": "upstream",
        "full_mode_override": "upstream_full",
    }
    _dicache, runtime, *_rest = validate_runner_config(runner, config)
    assert runtime["compile_mode"] == "upstream"
    assert runtime["benchmark_full_mode"] == "upstream_full"
    runner["full_mode_override"] = "instrumented_full"
    with pytest.raises(ValueError, match="upstream_full"):
        validate_runner_config(runner, config)
    config["model"]["ema"] = "model_ema2"
    runner["full_mode_override"] = "upstream_full"
    with pytest.raises(ValueError, match="model.ema=model_ema1"):
        validate_runner_config(runner, config)


def test_deferred_compile_scripts_never_probe_or_execute_a_gpu_in_cpu_tests():
    combined = WORKER_PATH.read_text(encoding="utf-8") + MATRIX_PATH.read_text(
        encoding="utf-8"
    )
    assert "nvidia-smi" not in combined
    for forbidden in ("= 99", "== 99", "= 198", "== 198"):
        assert forbidden not in combined


def test_compile_matrix_wrapper_is_fail_closed_and_never_signals_processes():
    source = WRAPPER_PATH.read_text(encoding="utf-8")
    assert "DICACHE_GPU_TESTS_ALLOWED" in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "--query-compute-apps=pid" in source
    assert "--query-gpu=utilization.gpu,memory.used" in source
    assert "run_compile_matrix.py" in source
    assert "exec python" in source
    for forbidden in ("pkill", "kill -", "killall"):
        assert forbidden not in source
    subprocess.run(["bash", "-n", str(WRAPPER_PATH)], check=True)
