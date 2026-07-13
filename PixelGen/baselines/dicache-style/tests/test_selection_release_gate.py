from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from dicache_style.source_identity import release_source_bindings


ROOT = Path(__file__).resolve().parents[1]
MODEL_FAMILY = ROOT.parents[1].name
RECORD = ROOT / "scripts" / "record_selection.py"
DECISION = ROOT / "scripts" / "record_selection_decision.py"
SMOKE = ROOT / "scripts" / "record_smoke_gate.py"
GATE = ROOT / "scripts" / "release_gate.py"
LAUNCHER = ROOT / "scripts" / "launch_4gpu_50k.sh"
GENERATE = ROOT / "scripts" / "generate_shard.py"
SMOKE_SPEC = importlib.util.spec_from_file_location("pixelgen_smoke_gate", SMOKE)
assert SMOKE_SPEC is not None and SMOKE_SPEC.loader is not None
SMOKE_MODULE = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(SMOKE_MODULE)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_report(tmp_path: Path) -> Path:
    threshold = 0.125
    policy = "force_full"
    manifest_sha = "a" * 64
    checkpoint_sha = "c" * 64
    checkpoint_path = "/immutable/checkpoint.pt"
    checkpoint_size = 123
    candidate_config = {
        "mode": "dicache",
        "profile": "flux_image_released",
        "probe_depth": 1,
        "error_choice": "delta_y",
        "rel_l1_thresh": threshold,
        "gamma_nonfinite_policy": policy,
    }
    config = {"dicache": candidate_config}
    config_path = tmp_path / "benchmark_candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_hash = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    dicache_hash = hashlib.sha256(
        json.dumps(
            candidate_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    paired = tmp_path / "paired_8k.json"
    trace = tmp_path / "trace_8k.json"
    benchmark = tmp_path / "benchmark_8k.json"
    _write_json(
        paired,
        {
            "sample_count": 8000,
            "reference_manifest_sha256": manifest_sha,
            "candidate_manifest_sha256": manifest_sha,
            "candidate_dicache_config": candidate_config,
            "candidate_input_config_hash": config_hash,
            "candidate_dicache_config_hash": dicache_hash,
            "candidate_method": "dicache",
            "candidate_run_identity": {
                "manifest_sha256": manifest_sha,
                "checkpoint_path": checkpoint_path,
                "checkpoint_size": checkpoint_size,
                "checkpoint_sha256": checkpoint_sha,
            },
            "aggregate_mse": 0.01,
            "psnr_from_aggregate_mse": 20.0,
            "per_image_psnr": {
                "mean": 20.0, "median": 20.0, "p90": 21.0,
                "p95": 22.0, "p99": 23.0,
            },
            "ssim": {
                "mean": 0.9, "median": 0.9, "p90": 0.92,
                "p95": 0.93, "p99": 0.94,
            },
            "exact_pair_count": 0,
            "nan_counts": {"psnr": 0, "ssim": 0, "lpips": 0},
            "inf_counts": {"psnr": 0, "ssim": 0, "lpips": 0},
            "ssim_protocol": {
                "channel_axis": -1, "data_range": 1.0, "win_size": 7,
            },
            "lpips": {
                "mean": 0.1, "median": 0.1, "p90": 0.12,
                "p95": 0.13, "p99": 0.14, "max": 0.2,
                "value_count": 8000, "backbone": "alex", "spatial": False,
                "package_version": "fixture",
            },
        },
    )
    _write_json(
        trace,
        {
            "trajectory_count": 8000,
            "profile_values": ["flux_image_released"],
            "probe_depth_values": [1],
            "error_choice_values": ["delta_y"],
            "rel_l1_thresh_values": [threshold],
            "gamma_nonfinite_policy_values": [policy],
            "config_hash_values": [config_hash],
            "dicache_config_hash_values": [dicache_hash],
            "manifest_sha256_values": [manifest_sha],
            "checkpoint_path_values": [checkpoint_path],
            "checkpoint_size_values": [checkpoint_size],
            "checkpoint_sha256_values": [checkpoint_sha],
            "method_values": ["dicache"],
            "all_call_counts_valid": True,
            "direct_full_count": 1,
            "resumed_full_count": 1,
            "reuse_count": 1,
            "dcta_count": 1,
        },
    )
    protocol = {
        "batch_size": 1,
        "compile_mode": "matched_eager",
        "input_config": str(config_path.resolve()),
        "input_config_hash": config_hash,
        "dicache_config_hash": dicache_hash,
        "manifest_sha256": manifest_sha,
        "checkpoint": checkpoint_path,
        "checkpoint_size": checkpoint_size,
        "checkpoint_sha256": checkpoint_sha,
    }
    if MODEL_FAMILY == "PixelGen":
        protocol["dicache"] = candidate_config
        protocol["config_mode"] = "dicache"
    else:
        protocol["candidate_mode"] = "dicache"
    _write_json(
        benchmark,
        {
            "protocol": protocol,
            "full": {
                "raw_ms_per_image": [2.0] * 30,
                "warmup_batches": 10,
                "measured_batches": 30,
            },
            "dicache": {
                "raw_ms_per_image": [1.0] * 30,
                "warmup_batches": 10,
                "measured_batches": 30,
            },
        },
    )
    output = tmp_path / "selection_decision.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(DECISION),
            "--model-family",
            MODEL_FAMILY,
            "--rel-l1-thresh",
            str(threshold),
            "--gamma-nonfinite-policy",
            policy,
            "--selection-rule",
            "quality_floor_then_minimum_latency",
            "--paired-report",
            str(paired),
            "--trace-report",
            str(trace),
            "--benchmark-report",
            str(benchmark),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    return output


def _record_selection(path: Path, *, status: str = "selected") -> dict[str, object]:
    decision = _decision_report(path.parent)
    command = [
        sys.executable,
        str(RECORD),
        "--model-family",
        MODEL_FAMILY,
        "--status",
        status,
        "--rel-l1-thresh",
        "0.125",
        "--gamma-nonfinite-policy",
        "force_full",
        "--decision-report",
        str(decision),
        "--output",
        str(path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, env=_environment()
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(path.read_text(encoding="utf-8"))


def _revalidate_decision(
    decision_path: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RECORD),
            "--model-family",
            MODEL_FAMILY,
            "--status",
            "selected",
            "--rel-l1-thresh",
            "0.125",
            "--gamma-nonfinite-policy",
            "force_full",
            "--decision-report",
            str(decision_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )


def _smoke_report(tmp_path: Path, full_config: Path, candidate_config: Path) -> Path:
    count = 1
    nfe = 99
    forwards = 198 if MODEL_FAMILY == "JiT" else 99
    parity = tmp_path / "resume_parity.json"
    png = tmp_path / "png_parity.json"
    validation = tmp_path / "candidate_validation.json"
    summary = tmp_path / "candidate_summary.json"
    metadata = tmp_path / "candidate_metadata.jsonl"
    upstream_config = tmp_path / "upstream.yaml"
    upstream = yaml.safe_load(full_config.read_text(encoding="utf-8"))
    upstream["dicache"]["mode"] = "upstream_full"
    upstream_config.write_text(yaml.safe_dump(upstream), encoding="utf-8")
    smoke_candidate = tmp_path / "smoke_candidate.yaml"
    smoke_candidate.write_text(candidate_config.read_text(encoding="utf-8"), encoding="utf-8")
    parity_value = {
        "schema_version": f"pixarc-{MODEL_FAMILY.lower()}-dicache-resume-parity-v1",
        "passed": True,
        "source": release_source_bindings(
            ROOT, ROOT.parents[2] / "third-party" / MODEL_FAMILY
        ),
        "nine_resume_invariants": {f"invariant_{index}": True for index in range(9)},
        "operational_invariants": {"finite": True, "runtime_reset": True},
        "expected_nfe": nfe,
        (
            "expected_network_forwards"
            if MODEL_FAMILY == "JiT"
            else "expected_combined_forwards"
        ): forwards,
    }
    _write_json(parity, parity_value)
    _write_json(
        png,
        {
            "schema_version": "pixarc-image-tree-parity-v1",
            "sample_count": count,
            "exact": True,
            "differing_image_count": 0,
        },
    )
    _write_json(
        validation,
        {
            "sample_count": count,
            "resolution": 256,
            "mode": "RGB",
            "dtype": "uint8",
            "identity_validation": "passed",
        },
    )
    prefix = "sum_" if MODEL_FAMILY == "JiT" else ""
    summary_value = {
        "trajectory_count": count,
        f"{prefix}total_nfe": nfe,
        f"{prefix}total_stream_calls": forwards,
        f"{prefix}network_forward_count": forwards,
        f"{prefix}direct_full_count": 1,
        f"{prefix}resumed_full_count": 0,
        f"{prefix}reuse_count": 1,
        f"{prefix}probe_count": 1,
        f"{prefix}dcta_count": 1,
    }
    if MODEL_FAMILY == "JiT":
        summary_value["all_call_counts_valid"] = True
    _write_json(summary, summary_value)
    metadata.write_text(
        json.dumps(
            {
                "trajectory_call_count_valid": True,
                "trajectory_total_nfe": nfe,
                "trajectory_total_stream_calls": forwards,
                "trajectory_network_forward_count": forwards,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "smoke_gate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SMOKE),
            "--model-family",
            MODEL_FAMILY,
            "--expected-count",
            str(count),
            "--resume-parity",
            str(parity),
            "--png-parity",
            str(png),
            "--candidate-validation",
            str(validation),
            "--candidate-summary",
            str(summary),
            "--candidate-metadata",
            str(metadata),
            "--upstream-config",
            str(upstream_config),
            "--full-config",
            str(full_config),
            "--candidate-config",
            str(smoke_candidate),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    return output


def _compile_report(
    full_config: Path, candidate_config: Path, manifest: Path
) -> dict[str, object]:
    source = release_source_bindings(
        ROOT, ROOT.parents[2] / "third-party" / MODEL_FAMILY
    )
    correctness = {
        "upstream_vs_matched_eager_full": {"passed": True},
        "matched_eager_vs_blockwise_full": {"passed": True},
    }
    if MODEL_FAMILY == "JiT":
        modes = {
            "upstream": ("upstream", "upstream_full", ("full",)),
            "matched_eager": (
                "matched_eager", "instrumented_full", ("full", "candidate")
            ),
            "blockwise": (
                "blockwise", "instrumented_full", ("full", "candidate")
            ),
        }
        candidate = yaml.safe_load(candidate_config.read_text(encoding="utf-8"))
        candidate_hash = hashlib.sha256(
            json.dumps(
                candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        raw_modes: dict[str, object] = {}
        mode_gates: dict[str, object] = {}
        matrix: dict[str, object] = {}
        for mode, (compile_mode, full_mode, roles) in modes.items():
            raw_modes[mode] = {
                "passed": True,
                "source": source,
                "mode": mode,
                "compile_mode": compile_mode,
                "full_mode": full_mode,
                "protocol": {
                    "compile_mode": compile_mode,
                    "input_config_hash": candidate_hash,
                    "batch_size": 1,
                },
                "roles": {role: {"passed": True} for role in roles},
            }
            mode_gates[mode] = {
                "passed": True,
                "checks": {"fixture_complete": True},
                "role_gates": {
                    role: {
                        "passed": True,
                        "checks": {"fixture_complete": True},
                    }
                    for role in roles
                },
            }
            matrix[mode] = {role: {} for role in roles}
        correctness["matched_eager_vs_blockwise_candidate"] = {"passed": True}
        return {
            "schema_version": "pixarc-jit-compile-matrix-v1",
            "model_family": "JiT",
            "passed": True,
            "source": source,
            "source_mismatches": {},
            "protocol": {
                "mode_order": list(modes),
                "role_order": {
                    mode: list(contract[2]) for mode, contract in modes.items()
                },
            },
            "identity": {
                "input_config_hash": candidate_hash,
                "checkpoint_sha256": _sha256(
                    Path(candidate["model"]["checkpoint"])
                ),
                "manifest_sha256": _sha256(manifest),
            },
            "identity_mismatches": {},
            "correctness": correctness,
            "mode_gates": mode_gates,
            "matrix": matrix,
            "modes": raw_modes,
        }

    rows = {
        "upstream_whole_model": ("upstream", "upstream_full", full_config),
        "matched_eager_full": ("matched_eager", "instrumented_full", full_config),
        "matched_eager_dicache": ("matched_eager", "dicache", candidate_config),
        "blockwise_full": ("blockwise", "instrumented_full", full_config),
        "blockwise_dicache": ("blockwise", "dicache", candidate_config),
    }
    matrix = {}
    source_rows = {}
    for role, (compile_mode, config_mode, config) in rows.items():
        config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
        matrix[role] = {
            "input_config": str(config.resolve()),
            "input_config_sha256": _sha256(config),
            "input_config_hash": hashlib.sha256(
                json.dumps(
                    config_value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "compile_mode": compile_mode,
            "config_mode": config_mode,
            "dicache_config_hash": hashlib.sha256(
                json.dumps(
                    config_value["dicache"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "row_gate": {
                "passed": True,
                "checks": {"fixture_complete": True},
            },
        }
        source_rows[role] = {"source": source}
    correctness["matched_eager_vs_blockwise_dicache"] = {"passed": True}
    return {
        "schema_version": "pixarc-dicache-compile-matrix-v1",
        "model_family": "PixelGen",
        "passed": True,
        "source": source,
        "source_mismatches": {},
        "protocol": {"row_order": list(rows)},
        "identity": {
            "batch_size": 1,
            "effective_cfg_batch_size": 2,
            "checkpoint_sha256": _sha256(
                Path(yaml.safe_load(candidate_config.read_text(encoding="utf-8"))["checkpoint"])
            ),
            "manifest_sha256": _sha256(manifest),
        },
        "identity_mismatches": {},
        "correctness": correctness,
        "matrix": matrix,
        "rows": source_rows,
    }


def _release_fixture(tmp_path: Path) -> dict[str, Path]:
    selection_path = tmp_path / "selection.json"
    selection = _record_selection(selection_path)
    selected_provenance = {
        **selection,
        "selection_report_sha256": _sha256(selection_path),
        "selection_report_name": selection_path.name,
        "threshold_selected_before_final_50k": True,
        "gamma_policy_preregistered": True,
        "checkpoint_resolved_absolute": True,
    }
    provisional_provenance = {
        "schema_version": "pixarc-dicache-selection-v1",
        "passed": True,
        "status": "provisional",
        "model_family": MODEL_FAMILY,
        "profile": "flux_image_released",
        "probe_depth": 1,
        "batch_size": 1,
        "rel_l1_thresh": None,
        "gamma_nonfinite_policy": "force_full",
        "final_50k_used_for_selection": False,
        "threshold_selected_before_final_50k": False,
        "gamma_policy_preregistered": False,
    }
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    base = {
        "schema_version": "pixarc-dicache-config-v1",
        "runtime": {"batch_size": 1, "compile_mode": "matched_eager"},
    }
    if MODEL_FAMILY == "JiT":
        base["model"] = {
            "variant": "JiT-B/16",
            "checkpoint": str(checkpoint.resolve()),
            "ema": "model_ema1",
            "args": {"hidden_size": 768},
        }
        base["sampling"] = {
            "method": "heun", "steps": 50, "exact_heun": True,
            "cfg_scale": 3.0, "guidance_interval": [0.1, 1.0],
            "dtype": "bfloat16",
        }
    else:
        base["runtime"].update(
            {"effective_cfg_batch_size": 2, "precision": "bf16-mixed"}
        )
        base["trainer"] = {"precision": "bf16-mixed"}
        base["data"] = {"pred_batch_size": 1}
        base["checkpoint"] = str(checkpoint.resolve())
        base["model"] = {
            "vae": {"class_path": "PixelAE", "init_args": {}},
            "denoiser": {"class_path": "PixelGenJiT", "init_args": {"depth": 28}},
            "conditioner": {"class_path": "LabelConditioner", "init_args": {}},
            "diffusion_sampler": {
                "class_path": "ExactHeun", "init_args": {"num_steps": 50, "guidance": 2.25}
            },
            "ema_tracker": {"class_path": "SimpleEMA", "init_args": {"decay": 0.9999}},
        }
    full_path = tmp_path / "full.yaml"
    candidate_path = tmp_path / "candidate.yaml"
    full = {
        **base,
        "dicache": {
            "mode": "instrumented_full",
            "profile": "flux_image_released",
            "probe_depth": 1,
            "rel_l1_thresh": None,
            "gamma_nonfinite_policy": "force_full",
        },
        "selection_provenance": provisional_provenance,
    }
    candidate = {
        **base,
        "dicache": {
            "mode": "dicache",
            "profile": "flux_image_released",
            "probe_depth": 1,
            "rel_l1_thresh": 0.125,
            "gamma_nonfinite_policy": "force_full",
        },
        "selection_provenance": selected_provenance,
    }
    full_path.write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"sample_id":0}\n', encoding="utf-8")
    _write_json(
        Path(f"{manifest}.meta.json"),
        {"manifest_sha256": _sha256(manifest), "record_count": 1},
    )
    smoke = _smoke_report(tmp_path, full_path, candidate_path)
    smoke_value = json.loads(smoke.read_text(encoding="utf-8"))
    parity = Path(smoke_value["artifacts"]["resume_parity"]["path"])
    paths = {
        "full": full_path,
        "candidate": candidate_path,
        "manifest": manifest,
        "selection": selection_path,
        "parity": parity,
        "smoke": smoke,
        "compile": tmp_path / "compile.json",
        "gate": tmp_path / "release_gate.json",
    }
    _write_json(
        paths["compile"], _compile_report(full_path, candidate_path, manifest)
    )
    return paths


def _create_gate(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GATE),
            "create",
            "--model-family",
            MODEL_FAMILY,
            "--full-config",
            str(paths["full"]),
            "--candidate-config",
            str(paths["candidate"]),
            "--manifest",
            str(paths["manifest"]),
            "--selection-report",
            str(paths["selection"]),
            "--parity-report",
            str(paths["parity"]),
            "--smoke-report",
            str(paths["smoke"]),
            "--compile-report",
            str(paths["compile"]),
            "--output",
            str(paths["gate"]),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )


def _verify_gate(paths: dict[str, Path], config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GATE),
            "verify",
            "--model-family",
            MODEL_FAMILY,
            "--gate",
            str(paths["gate"]),
            "--config",
            str(config),
            "--manifest",
            str(paths["manifest"]),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )


def test_selected_report_is_explicit_and_final_50k_independent(tmp_path: Path):
    report = _record_selection(tmp_path / "selected.json")
    assert report["schema_version"] == "pixarc-dicache-selection-v1"
    assert report["passed"] is True
    assert report["status"] == "selected"
    assert report["model_family"] == MODEL_FAMILY
    assert report["profile"] == "flux_image_released"
    assert report["probe_depth"] == report["batch_size"] == 1
    assert report["final_50k_used_for_selection"] is False
    assert report["decision"]["passed"] is True


def test_selected_report_rejects_missing_threshold_and_policy(tmp_path: Path):
    output = tmp_path / "invalid.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RECORD),
            "--model-family",
            MODEL_FAMILY,
            "--status",
            "selected",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode != 0
    assert not output.exists()


def test_provisional_report_records_diagnostic_probe_depth(tmp_path: Path):
    output = tmp_path / "provisional.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RECORD),
            "--model-family",
            MODEL_FAMILY,
            "--status",
            "provisional",
            "--probe-depth",
            "3",
            "--rel-l1-thresh",
            "0.125",
            "--gamma-nonfinite-policy",
            "force_full",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "provisional"
    assert report["probe_depth"] == 3
    assert report["decision"] is None


def test_selection_decision_rejects_benchmark_candidate_mismatch(tmp_path: Path):
    decision_path = _decision_report(tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    benchmark_path = Path(decision["evidence"]["benchmark"]["path"])
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if MODEL_FAMILY == "PixelGen":
        benchmark["protocol"]["dicache"]["rel_l1_thresh"] = 0.5
    else:
        config_path = Path(benchmark["protocol"]["input_config"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["dicache"]["rel_l1_thresh"] = 0.5
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        benchmark["protocol"]["input_config_hash"] = hashlib.sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
    _write_json(benchmark_path, benchmark)
    decision["evidence"]["benchmark"]["sha256"] = _sha256(benchmark_path)
    _write_json(decision_path, decision)
    output = tmp_path / "selection.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RECORD),
            "--model-family",
            MODEL_FAMILY,
            "--status",
            "selected",
            "--rel-l1-thresh",
            "0.125",
            "--gamma-nonfinite-policy",
            "force_full",
            "--decision-report",
            str(decision_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode != 0
    assert not output.exists()
    assert "benchmark candidate threshold differs" in completed.stderr


def test_selection_decision_rejects_trace_gamma_policy_mismatch(tmp_path: Path):
    decision_path = _decision_report(tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    trace_path = Path(decision["evidence"]["trace"]["path"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["gamma_nonfinite_policy_values"] = ["official_propagate"]
    _write_json(trace_path, trace)
    decision["evidence"]["trace"]["sha256"] = _sha256(trace_path)
    _write_json(decision_path, decision)

    completed = _revalidate_decision(decision_path, tmp_path / "selection.json")
    assert completed.returncode != 0
    assert "trace gamma policy mismatch" in completed.stderr


def test_selection_decision_rejects_cross_evidence_identity_mismatch(
    tmp_path: Path,
):
    decision_path = _decision_report(tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    trace_path = Path(decision["evidence"]["trace"]["path"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["checkpoint_sha256_values"] = ["d" * 64]
    _write_json(trace_path, trace)
    decision["evidence"]["trace"]["sha256"] = _sha256(trace_path)
    _write_json(decision_path, decision)

    completed = _revalidate_decision(decision_path, tmp_path / "selection.json")
    assert completed.returncode != 0
    assert "candidate identity differs" in completed.stderr


def test_release_gate_allows_only_its_full_and_candidate_configs(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    created = _create_gate(paths)
    assert created.returncode == 0, created.stderr
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    assert gate["source"] == release_source_bindings(
        ROOT, ROOT.parents[2] / "third-party" / MODEL_FAMILY
    )
    for role in ("full", "candidate"):
        verified = _verify_gate(paths, paths[role])
        assert verified.returncode == 0, verified.stderr
        assert json.loads(verified.stdout)["passed"] is True
    other = tmp_path / "other.yaml"
    other.write_text(paths["full"].read_text(encoding="utf-8") + "# changed\n")
    rejected = _verify_gate(paths, other)
    assert rejected.returncode != 0
    assert "neither release-gated" in rejected.stderr


def test_worker_reverifies_exact_archived_gate_and_rejects_digest_drift(
    tmp_path: Path,
):
    paths = _release_fixture(tmp_path)
    created = _create_gate(paths)
    assert created.returncode == 0, created.stderr
    output_root = tmp_path / "worker-output"
    output_root.mkdir()
    archived = {
        "gate": output_root / "release_gate.json",
        "config": output_root / "config_resolved.yaml",
        "manifest": output_root / "input_manifest.jsonl",
        "sidecar": output_root / "input_manifest.jsonl.meta.json",
    }
    archived["gate"].write_bytes(paths["gate"].read_bytes())
    archived["config"].write_bytes(paths["candidate"].read_bytes())
    archived["manifest"].write_bytes(paths["manifest"].read_bytes())
    archived["sidecar"].write_bytes(
        Path(f"{paths['manifest']}.meta.json").read_bytes()
    )
    digest = _sha256(archived["gate"])

    def verify(expected_digest: str, config: Path = archived["config"]):
        return subprocess.run(
            [
                sys.executable,
                str(GATE),
                "worker-verify",
                "--model-family",
                MODEL_FAMILY,
                "--gate",
                str(archived["gate"]),
                "--expected-gate-sha256",
                expected_digest,
                "--config",
                str(config),
                "--manifest",
                str(archived["manifest"]),
                "--output-root",
                str(output_root),
            ],
            capture_output=True,
            text=True,
            env=_environment(),
        )

    passed = verify(digest)
    assert passed.returncode == 0, passed.stderr
    report = json.loads(passed.stdout)
    assert report["release_gate_sha256"] == digest
    assert report["checkpoint_sha256"] == _sha256(Path(report["checkpoint_path"]))
    rejected_digest = verify("0" * 64)
    assert rejected_digest.returncode != 0
    assert "release gate SHA-256 mismatch" in rejected_digest.stderr
    rejected_path = verify(digest, paths["candidate"])
    assert rejected_path.returncode != 0
    assert "exact output-root archive" in rejected_path.stderr


def test_direct_final_worker_cli_cannot_bypass_release_gate(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    output_root = tmp_path / "direct-worker"
    output_root.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATE),
            "--config",
            str(paths["candidate"]),
            "--manifest",
            str(paths["manifest"]),
            "--shard-id",
            "0",
            "--world-size",
            "1",
            "--output-root",
            str(output_root),
            "--acknowledge-gpu-job",
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode != 0
    assert "final shard workers require --release-gate" in completed.stderr


def test_release_gate_detects_bound_evidence_tampering(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    created = _create_gate(paths)
    assert created.returncode == 0, created.stderr
    _write_json(
        paths["parity"],
        {"schema_version": "pixarc-dicache-parity-v1", "passed": False},
    )
    rejected = _verify_gate(paths, paths["candidate"])
    assert rejected.returncode != 0
    assert "SHA-256 changed" in rejected.stderr


def test_release_gate_rehashes_checkpoint_on_verify(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    created = _create_gate(paths)
    assert created.returncode == 0, created.stderr
    config = yaml.safe_load(paths["candidate"].read_text(encoding="utf-8"))
    checkpoint_value = (
        config["model"]["checkpoint"]
        if MODEL_FAMILY == "JiT"
        else config["checkpoint"]
    )
    checkpoint = Path(checkpoint_value)
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"X" * len(original))
    rejected = _verify_gate(paths, paths["candidate"])
    assert rejected.returncode != 0
    assert "checkpoint SHA-256 changed" in rejected.stderr


def test_release_gate_revalidates_smoke_gate_inputs(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    created = _create_gate(paths)
    assert created.returncode == 0, created.stderr
    smoke = json.loads(paths["smoke"].read_text(encoding="utf-8"))
    summary = Path(smoke["artifacts"]["candidate_summary"]["path"])
    _write_json(summary, {"trajectory_count": 0})
    rejected = _verify_gate(paths, paths["candidate"])
    assert rejected.returncode != 0
    assert "candidate_summary SHA-256 changed" in rejected.stderr


def test_smoke_gate_rejects_nested_parity_source_identity(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    smoke = json.loads(paths["smoke"].read_text(encoding="utf-8"))
    parity_path = Path(smoke["artifacts"]["resume_parity"]["path"])
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["source"]["port"]["sha256"] = "0" * 64
    _write_json(parity_path, parity)
    smoke["artifacts"]["resume_parity"]["sha256"] = _sha256(parity_path)
    with pytest.raises(ValueError, match="parity source identity differs"):
        SMOKE_MODULE.validate_smoke_gate(
            smoke, expected_model_family=MODEL_FAMILY
        )


def test_release_gate_rejects_nested_compile_source_identity(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    report = json.loads(paths["compile"].read_text(encoding="utf-8"))
    if MODEL_FAMILY == "JiT":
        report["modes"]["blockwise"]["source"]["port"]["sha256"] = "0" * 64
        expected = "compile worker blockwise source identity mismatch"
    else:
        report["rows"]["blockwise_dicache"]["source"]["port"]["sha256"] = "0" * 64
        expected = "compile row blockwise_dicache source identity mismatch"
    _write_json(paths["compile"], report)
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert expected in rejected.stderr


def test_release_gate_revalidates_selection_decision_inputs(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    decision_path = Path(selection["decision"]["path"])
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    trace = Path(decision["evidence"]["trace"]["path"])
    _write_json(trace, {"trajectory_count": 0})
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert not paths["gate"].exists()
    assert "trace evidence SHA-256 changed" in rejected.stderr


def test_release_gate_rejects_manifest_sidecar_tampering(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    created = _create_gate(paths)
    assert created.returncode == 0, created.stderr
    sidecar = Path(f'{paths["manifest"]}.meta.json')
    _write_json(sidecar, {"manifest_sha256": "0" * 64})
    rejected = _verify_gate(paths, paths["candidate"])
    assert rejected.returncode != 0
    assert "manifest sidecar SHA-256 changed" in rejected.stderr


def test_release_gate_rejects_candidate_contract_drift(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    candidate = yaml.safe_load(paths["candidate"].read_text(encoding="utf-8"))
    candidate["dicache"]["rel_l1_thresh"] = 0.5
    paths["candidate"].write_text(yaml.safe_dump(candidate), encoding="utf-8")
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert not paths["gate"].exists()
    assert "threshold differs" in rejected.stderr


def test_release_gate_rejects_final_sampler_drift_from_smoke(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    candidate = yaml.safe_load(paths["candidate"].read_text(encoding="utf-8"))
    if MODEL_FAMILY == "JiT":
        candidate["sampling"]["cfg_scale"] = 7.0
        expected = "sampling protocols must match"
    else:
        candidate["model"]["diffusion_sampler"]["init_args"]["guidance"] = 7.0
        expected = "config SHA-256 mismatch"
    paths["candidate"].write_text(yaml.safe_dump(candidate), encoding="utf-8")
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert expected in rejected.stderr


def test_release_gate_requires_the_smoke_gated_parity_file(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    alternate = tmp_path / "alternate_parity.json"
    _write_json(
        alternate,
        {
            "schema_version": "pixarc-dicache-parity-v1",
            "passed": True,
            "source": release_source_bindings(
                ROOT, ROOT.parents[2] / "third-party" / MODEL_FAMILY
            ),
        },
    )
    paths["parity"] = alternate
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert "not the smoke-gated parity" in rejected.stderr


def test_release_gate_rejects_minimal_compile_claim(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    _write_json(
        paths["compile"],
        {
            "schema_version": "pixarc-dicache-compile-v1",
            "passed": True,
            "source": release_source_bindings(
                ROOT, ROOT.parents[2] / "third-party" / MODEL_FAMILY
            ),
        },
    )
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert not paths["gate"].exists()
    assert "model_family mismatch" in rejected.stderr


def test_release_gate_rejects_compile_config_binding_drift(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    report = json.loads(paths["compile"].read_text(encoding="utf-8"))
    if MODEL_FAMILY == "JiT":
        report["identity"]["input_config_hash"] = "0" * 64
    else:
        alternate = tmp_path / "alternate_compile_candidate.yaml"
        alternate.write_text(
            paths["candidate"].read_text(encoding="utf-8") + "# matrix-only copy\n",
            encoding="utf-8",
        )
        row = report["matrix"]["matched_eager_dicache"]
        row["input_config"] = str(alternate.resolve())
        row["input_config_sha256"] = _sha256(alternate)
    _write_json(paths["compile"], report)
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert not paths["gate"].exists()
    assert "not bound to final DiCache config" in rejected.stderr


def test_release_gate_rejects_nonmatched_final_compile_mode(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    candidate = yaml.safe_load(paths["candidate"].read_text(encoding="utf-8"))
    candidate["runtime"]["compile_mode"] = "blockwise"
    paths["candidate"].write_text(yaml.safe_dump(candidate), encoding="utf-8")
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert "must both use matched_eager" in rejected.stderr


def test_release_gate_rejects_final_dtype_or_batch_protocol_drift(tmp_path: Path):
    paths = _release_fixture(tmp_path)
    candidate = yaml.safe_load(paths["candidate"].read_text(encoding="utf-8"))
    if MODEL_FAMILY == "JiT":
        candidate["sampling"]["dtype"] = "float32"
        expected = "sampling protocols must match"
    else:
        candidate["runtime"]["effective_cfg_batch_size"] = 4
        expected = "effective_cfg_batch_size must be 2"
    paths["candidate"].write_text(yaml.safe_dump(candidate), encoding="utf-8")
    rejected = _create_gate(paths)
    assert rejected.returncode != 0
    assert expected in rejected.stderr


def test_launcher_requires_and_verifies_gate_before_gpu_checks():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Refusing final 50K launch without --release-gate FILE." in source
    verify_at = source.index('python "$RELEASE_GATE_SCRIPT" verify')
    permission_at = source.index("DICACHE_GPU_TESTS_ALLOWED")
    gpu_query_at = source.index("nvidia-smi")
    assert verify_at < permission_at < gpu_query_at
    assert 'ARCHIVED_RELEASE_GATE="$OUTPUT_ROOT/release_gate.json"' in source
    assert 'cmp -s "$RELEASE_GATE" "$ARCHIVED_RELEASE_GATE"' in source
    assert '--release-gate-sha256 "$ARCHIVED_RELEASE_GATE_SHA256"' in source
    worker = GENERATE.read_text(encoding="utf-8")
    assert '"worker-verify"' in worker
    assert "release-critical source changed after worker gate verification" in worker
