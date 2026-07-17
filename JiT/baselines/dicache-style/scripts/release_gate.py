#!/usr/bin/env python3
"""Create or verify a fail-closed evidence gate for final 50K generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

BASELINE_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = BASELINE_ROOT.parents[2]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.source_identity import (  # noqa: E402
    release_source_bindings,
    require_source_identity_current,
)
from record_selection import load_json_object, validate_selection_report
from record_smoke_gate import execution_contract, validate_smoke_gate


SCHEMA_VERSION = "pixarc-dicache-release-gate-v1"
PROFILE = "flux_image_released"
BATCH_SIZE_BY_MODEL = {"JiT": 32, "PixelGen": 4}
FULL_MODES = ("upstream_full", "instrumented_full")
JIT_COMPILE_MODES = {
    "upstream": ("upstream", "upstream_full", ("full",)),
    "matched_eager": (
        "matched_eager", "instrumented_full", ("full", "candidate")
    ),
    "blockwise": ("blockwise", "instrumented_full", ("full", "candidate")),
}
PIXEL_COMPILE_ROWS = {
    "upstream_whole_model": ("upstream", "upstream_full"),
    "matched_eager_full": ("matched_eager", "instrumented_full"),
    "matched_eager_dicache": ("matched_eager", "dicache"),
    "blockwise_full": ("blockwise", "instrumented_full"),
    "blockwise_dicache": ("blockwise", "dicache"),
}
COMPILE_CORRECTNESS_KEYS = {
    "upstream_vs_matched_eager_full",
    "matched_eager_vs_blockwise_full",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bindings(model_family: str) -> dict[str, Any]:
    return release_source_bindings(
        BASELINE_ROOT, PIXARC_ROOT / "third-party" / model_family
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config is not a YAML mapping: {path}")
    return value


def _finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value in {name}")


def _validate_evidence(
    role: str,
    path: Path,
    model_family: str,
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    report = load_json_object(path)
    _finite_tree(report, role)
    if report.get("passed") is not True:
        raise ValueError(f"{role} report passed must be exactly true: {path}")
    schema = report.get("schema_version")
    if not isinstance(schema, str) or role not in schema.lower():
        raise ValueError(f"{role} report schema_version must identify {role}: {path}")
    if report.get("source") != expected_source:
        raise ValueError(f"{role} report source identity differs from release gate")
    if role == "smoke":
        validate_smoke_gate(report, expected_model_family=model_family)
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "schema_version": schema,
        "passed": True,
    }


def _config_core(
    config: Mapping[str, Any], model_family: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError("release configs must use pixarc-dicache-config-v1")
    dicache = config.get("dicache")
    runtime = config.get("runtime")
    if not isinstance(dicache, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("release configs require dicache and runtime mappings")
    if dicache.get("profile") != PROFILE:
        raise ValueError(f"release config profile must be {PROFILE}")
    if dicache.get("probe_depth") != 1 or isinstance(dicache.get("probe_depth"), bool):
        raise ValueError("release config probe_depth must be integer 1")
    expected_batch_size = BATCH_SIZE_BY_MODEL[model_family]
    if (
        runtime.get("batch_size") != expected_batch_size
        or isinstance(runtime.get("batch_size"), bool)
    ):
        raise ValueError(
            f"release config runtime.batch_size must be integer {expected_batch_size}"
        )
    return dicache, runtime


def _validate_provenance(
    config: Mapping[str, Any],
    *,
    expected_status: str,
    expected_model_family: str,
    selection: Mapping[str, Any] | None = None,
    selection_sha256: str | None = None,
) -> None:
    provenance = config.get("selection_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("release config lacks materialized selection_provenance")
    if provenance.get("schema_version") != "pixarc-dicache-selection-v1":
        raise ValueError("selection_provenance schema mismatch")
    if provenance.get("status") != expected_status or provenance.get("passed") is not True:
        raise ValueError(f"config selection provenance must be passed {expected_status}")
    if provenance.get("model_family") != expected_model_family:
        raise ValueError("config selection provenance model_family mismatch")
    if provenance.get("profile") != PROFILE:
        raise ValueError("config selection provenance profile mismatch")
    if provenance.get("batch_size") != BATCH_SIZE_BY_MODEL[expected_model_family]:
        raise ValueError("config selection provenance batch_size mismatch")
    if provenance.get("final_50k_used_for_selection") is not False:
        raise ValueError("config provenance must exclude final 50K from selection")
    if expected_status == "selected":
        if provenance.get("probe_depth") != 1:
            raise ValueError("selected config provenance probe_depth mismatch")
        if provenance.get("threshold_selected_before_final_50k") is not True:
            raise ValueError("selected threshold was not frozen before final 50K")
        if provenance.get("gamma_policy_preregistered") is not True:
            raise ValueError("selected gamma policy was not preregistered")
        if selection is None or selection_sha256 is None:
            raise ValueError("selected provenance validation needs its source report")
        for key in (
            "status",
            "passed",
            "model_family",
            "profile",
            "probe_depth",
            "batch_size",
            "rel_l1_thresh",
            "gamma_nonfinite_policy",
            "final_50k_used_for_selection",
            "decision",
        ):
            if provenance.get(key) != selection.get(key):
                raise ValueError(f"config provenance differs from selection report at {key}")
        if provenance.get("selection_report_sha256") != selection_sha256:
            raise ValueError("config provenance selection-report SHA-256 mismatch")


def _validate_release_configs(
    full_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    *,
    model_family: str,
    selection: Mapping[str, Any],
    selection_sha256: str,
) -> None:
    full_dicache, full_runtime = _config_core(full_config, model_family)
    candidate_dicache, candidate_runtime = _config_core(candidate_config, model_family)
    if full_dicache.get("mode") not in FULL_MODES:
        raise ValueError(f"full config mode must be one of {FULL_MODES}")
    if candidate_dicache.get("mode") != "dicache":
        raise ValueError("candidate release config must use mode=dicache")
    if (
        full_runtime.get("compile_mode") != "matched_eager"
        or candidate_runtime.get("compile_mode") != "matched_eager"
    ):
        raise ValueError("release Full and DiCache configs must both use matched_eager")
    if model_family == "JiT":
        full_model = full_config.get("model")
        candidate_model = candidate_config.get("model")
        if (
            not isinstance(full_model, Mapping)
            or not isinstance(candidate_model, Mapping)
            or full_model.get("ema") != "model_ema1"
            or candidate_model.get("ema") != "model_ema1"
        ):
            raise ValueError("JiT release configs require model.ema=model_ema1")
        full_sampling = full_config.get("sampling")
        candidate_sampling = candidate_config.get("sampling")
        if not isinstance(full_sampling, Mapping) or not isinstance(
            candidate_sampling, Mapping
        ):
            raise ValueError("JiT release configs require sampling mappings")
        if full_sampling != candidate_sampling:
            raise ValueError("JiT Full and DiCache sampling protocols must match")
        if full_sampling.get("dtype") != "bfloat16":
            raise ValueError("JiT release sampling dtype must be bfloat16")
    elif model_family == "PixelGen":
        for role, config, runtime in (
            ("Full", full_config, full_runtime),
            ("DiCache", candidate_config, candidate_runtime),
        ):
            trainer = config.get("trainer")
            data = config.get("data")
            if runtime.get("effective_cfg_batch_size") != 8:
                raise ValueError(
                    f"PixelGen {role} effective_cfg_batch_size must be 8"
                )
            if runtime.get("precision") != "bf16-mixed":
                raise ValueError(f"PixelGen {role} runtime precision must be bf16-mixed")
            if not isinstance(trainer, Mapping) or trainer.get("precision") != "bf16-mixed":
                raise ValueError(f"PixelGen {role} trainer precision must be bf16-mixed")
            if not isinstance(data, Mapping) or data.get("pred_batch_size") != 4:
                raise ValueError(f"PixelGen {role} pred_batch_size must be 4")
    else:
        raise ValueError(f"unsupported release model_family: {model_family}")
    _validate_provenance(
        full_config,
        expected_status="provisional",
        expected_model_family=model_family,
    )
    _validate_provenance(
        candidate_config,
        expected_status="selected",
        expected_model_family=model_family,
        selection=selection,
        selection_sha256=selection_sha256,
    )
    if candidate_dicache.get("rel_l1_thresh") != selection["rel_l1_thresh"]:
        raise ValueError("candidate threshold differs from selected threshold")
    if (
        candidate_dicache.get("gamma_nonfinite_policy")
        != selection["gamma_nonfinite_policy"]
    ):
        raise ValueError("candidate gamma policy differs from selected policy")


def _require_passed_gate(
    value: Any, name: str, *, require_checks: bool = False
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        raise ValueError(f"compile {name} must be a passed mapping")
    if require_checks:
        checks = value.get("checks")
        if (
            not isinstance(checks, Mapping)
            or not checks
            or any(check is not True for check in checks.values())
        ):
            raise ValueError(f"compile {name} checks must all be exactly true")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_compile_report(
    path: Path,
    *,
    model_family: str,
    full_path: Path,
    candidate_path: Path,
    manifest_path: Path,
    expected_source: Mapping[str, Any] | None = None,
) -> None:
    """Revalidate the family-specific three-mode/five-role release matrix."""

    report = load_json_object(path)
    if report.get("model_family") != model_family:
        raise ValueError("compile report model_family mismatch")
    source = dict(expected_source or _source_bindings(model_family))
    if report.get("source") != source:
        raise ValueError("compile report source bytes differ from the current port/upstream")
    if report.get("source_mismatches") != {}:
        raise ValueError("compile report contains source identity mismatches")
    full_sha = sha256_file(full_path)
    candidate_sha = sha256_file(candidate_path)
    full_config = _load_yaml(full_path)
    candidate_config = _load_yaml(candidate_path)
    checkpoint_value = (
        candidate_config.get("model", {}).get("checkpoint")
        if model_family == "JiT"
        else candidate_config.get("checkpoint")
    )
    if not isinstance(checkpoint_value, str):
        raise ValueError("compile-bound candidate config lacks checkpoint")
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = candidate_path.parent / checkpoint_path
    checkpoint_sha = sha256_file(checkpoint_path.resolve(strict=True))
    manifest_sha = sha256_file(manifest_path)
    correctness = report.get("correctness")
    identity_mismatches = report.get("identity_mismatches")
    if identity_mismatches != {}:
        raise ValueError("compile report contains identity mismatches")
    if not isinstance(correctness, Mapping):
        raise ValueError("compile report lacks correctness gates")

    if model_family == "JiT":
        if report.get("schema_version") != "pixarc-jit-compile-matrix-v1":
            raise ValueError("JiT compile matrix schema mismatch")
        required_correctness = COMPILE_CORRECTNESS_KEYS | {
            "matched_eager_vs_blockwise_candidate"
        }
        protocol = report.get("protocol")
        matrix = report.get("matrix")
        modes = report.get("modes")
        mode_gates = report.get("mode_gates")
        identity = report.get("identity")
        if not all(
            isinstance(value, Mapping)
            for value in (protocol, matrix, modes, mode_gates, identity)
        ):
            raise ValueError("JiT compile matrix lacks required mappings")
        if protocol.get("mode_order") != list(JIT_COMPILE_MODES):
            raise ValueError("JiT compile mode order mismatch")
        expected_role_order = {
            mode: list(contract[2]) for mode, contract in JIT_COMPILE_MODES.items()
        }
        if protocol.get("role_order") != expected_role_order:
            raise ValueError("JiT compile role order mismatch")
        if any(
            set(value) != set(JIT_COMPILE_MODES)
            for value in (matrix, modes, mode_gates)
        ):
            raise ValueError("JiT compile matrix must contain exactly three modes")
        candidate_hash = _canonical_hash(candidate_config)
        if identity.get("input_config_hash") != candidate_hash:
            raise ValueError("JiT compile matrix is not bound to final DiCache config")
        if (
            identity.get("checkpoint_sha256") != checkpoint_sha
            or identity.get("manifest_sha256") != manifest_sha
        ):
            raise ValueError("JiT compile matrix checkpoint/manifest binding mismatch")
        for mode, (compile_mode, full_mode, expected_roles) in JIT_COMPILE_MODES.items():
            row = modes[mode]
            gate = _require_passed_gate(
                mode_gates[mode], f"JiT {mode} mode gate", require_checks=True
            )
            if not isinstance(row, Mapping) or row.get("passed") is not True:
                raise ValueError(f"JiT compile mode {mode} did not pass")
            if row.get("source") != source:
                raise ValueError(f"JiT compile worker {mode} source identity mismatch")
            row_protocol = row.get("protocol")
            roles = row.get("roles")
            matrix_roles = matrix[mode]
            if not all(
                isinstance(value, Mapping)
                for value in (row_protocol, roles, matrix_roles)
            ):
                raise ValueError(f"JiT compile mode {mode} is incomplete")
            if (
                row.get("mode") != mode
                or row.get("compile_mode") != compile_mode
                or row.get("full_mode") != full_mode
                or row_protocol.get("compile_mode") != compile_mode
                or row_protocol.get("input_config_hash") != candidate_hash
                or row_protocol.get("batch_size") != 32
            ):
                raise ValueError(f"JiT compile mode {mode} contract mismatch")
            if (
                set(roles) != set(expected_roles)
                or set(matrix_roles) != set(expected_roles)
            ):
                raise ValueError(f"JiT compile mode {mode} role set mismatch")
            role_gates = gate.get("role_gates")
            if not isinstance(role_gates, Mapping) or set(role_gates) != set(
                expected_roles
            ):
                raise ValueError(f"JiT compile mode {mode} role gates are incomplete")
            for role in expected_roles:
                measurement = roles[role]
                if not isinstance(measurement, Mapping) or measurement.get("passed") is not True:
                    raise ValueError(f"JiT compile role {mode}/{role} did not pass")
                _require_passed_gate(
                    role_gates[role],
                    f"JiT {mode}/{role} role gate",
                    require_checks=True,
                )
    elif model_family == "PixelGen":
        if report.get("schema_version") != "pixarc-dicache-compile-matrix-v1":
            raise ValueError("PixelGen compile matrix schema mismatch")
        required_correctness = COMPILE_CORRECTNESS_KEYS | {
            "matched_eager_vs_blockwise_dicache"
        }
        protocol = report.get("protocol")
        matrix = report.get("matrix")
        rows = report.get("rows")
        identity = report.get("identity")
        if not all(
            isinstance(value, Mapping) for value in (protocol, matrix, rows, identity)
        ):
            raise ValueError("PixelGen compile matrix lacks required mappings")
        if protocol.get("row_order") != list(PIXEL_COMPILE_ROWS):
            raise ValueError("PixelGen compile row order mismatch")
        if set(matrix) != set(PIXEL_COMPILE_ROWS):
            raise ValueError("PixelGen compile matrix must contain exactly five rows")
        if set(rows) != set(PIXEL_COMPILE_ROWS):
            raise ValueError("PixelGen compile report must retain exactly five source rows")
        if identity.get("batch_size") != 4 or identity.get(
            "effective_cfg_batch_size"
        ) != 8:
            raise ValueError("PixelGen compile identity must use real batch 4 / CFG batch 8")
        if (
            identity.get("checkpoint_sha256") != checkpoint_sha
            or identity.get("manifest_sha256") != manifest_sha
        ):
            raise ValueError("PixelGen compile matrix checkpoint/manifest binding mismatch")
        expected_dicache_hashes = {
            "matched_eager_full": _canonical_hash(full_config["dicache"]),
            "matched_eager_dicache": _canonical_hash(candidate_config["dicache"]),
            "blockwise_full": _canonical_hash(full_config["dicache"]),
            "blockwise_dicache": _canonical_hash(candidate_config["dicache"]),
        }
        for role, (compile_mode, config_mode) in PIXEL_COMPILE_ROWS.items():
            row = matrix[role]
            source_row = rows[role]
            if not isinstance(row, Mapping):
                raise ValueError(f"PixelGen compile row {role} is not a mapping")
            if not isinstance(source_row, Mapping) or source_row.get("source") != source:
                raise ValueError(f"PixelGen compile row {role} source identity mismatch")
            _require_passed_gate(
                row.get("row_gate"),
                f"PixelGen {role} row gate",
                require_checks=True,
            )
            config_value = row.get("input_config")
            config_sha = row.get("input_config_sha256")
            if not isinstance(config_value, str) or not isinstance(config_sha, str):
                raise ValueError(f"PixelGen compile row {role} lacks config binding")
            config_path = Path(config_value).resolve(strict=True)
            if sha256_file(config_path) != config_sha:
                raise ValueError(f"PixelGen compile row {role} config SHA-256 mismatch")
            row_config = _load_yaml(config_path)
            if (
                row.get("compile_mode") != compile_mode
                or row.get("config_mode") != config_mode
                or row.get("input_config_hash") != _canonical_hash(row_config)
                or (
                    role in expected_dicache_hashes
                    and row.get("dicache_config_hash")
                    != expected_dicache_hashes[role]
                )
            ):
                raise ValueError(f"PixelGen compile row {role} contract mismatch")
        if matrix["matched_eager_full"].get("input_config_sha256") != full_sha:
            raise ValueError("PixelGen compile matrix is not bound to final Full config")
        if matrix["matched_eager_dicache"].get("input_config_sha256") != candidate_sha:
            raise ValueError("PixelGen compile matrix is not bound to final DiCache config")
    else:
        raise ValueError(f"unsupported compile report model_family: {model_family}")

    if set(correctness) != required_correctness:
        raise ValueError("compile correctness gate set mismatch")
    for name in required_correctness:
        _require_passed_gate(correctness[name], f"{name} correctness gate")


def _file_binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
    }


def _validate_smoke_final_contract(
    smoke_path: Path, full_path: Path, candidate_path: Path, model_family: str,
    parity_sha256: str,
) -> None:
    smoke = load_json_object(smoke_path)
    artifacts = smoke.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("smoke gate lacks artifact bindings")
    resume_parity = artifacts.get("resume_parity")
    if (
        not isinstance(resume_parity, Mapping)
        or resume_parity.get("sha256") != parity_sha256
    ):
        raise ValueError("release parity report is not the smoke-gated parity report")
    smoke_contract = smoke.get("execution_contract")
    if not isinstance(smoke_contract, Mapping):
        raise ValueError("smoke gate lacks an execution contract")
    checkpoint = smoke_contract.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("sha256"), str):
        raise ValueError("smoke gate lacks checkpoint SHA-256")
    checkpoint_path_value = checkpoint.get("path")
    if not isinstance(checkpoint_path_value, str):
        raise ValueError("smoke gate lacks checkpoint path")
    checkpoint_path = Path(checkpoint_path_value).resolve(strict=True)
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("checkpoint SHA-256 changed after smoke validation")
    final_contract = execution_contract(
        {"full": full_path, "candidate": candidate_path},
        model_family=model_family,
        final_pair=True,
    )
    if smoke_contract != final_contract:
        raise ValueError(
            "final configs do not match smoke checkpoint/model/sampler contract"
        )


def _manifest_sidecar(path: Path) -> Path:
    sidecar = Path(f"{path}.meta.json").resolve(strict=True)
    metadata = load_json_object(sidecar)
    if metadata.get("manifest_sha256") != sha256_file(path):
        raise ValueError("manifest sidecar manifest_sha256 mismatch")
    return sidecar


def create_gate(arguments: argparse.Namespace) -> dict[str, Any]:
    release_source = _source_bindings(arguments.model_family)
    full_path = arguments.full_config.resolve(strict=True)
    candidate_path = arguments.candidate_config.resolve(strict=True)
    manifest_path = arguments.manifest.resolve(strict=True)
    manifest_sidecar = _manifest_sidecar(manifest_path)
    selection_path = arguments.selection_report.resolve(strict=True)
    if full_path == candidate_path:
        raise ValueError("full and candidate configs must be distinct files")
    selection_sha = sha256_file(selection_path)
    selection = validate_selection_report(
        load_json_object(selection_path),
        expected_model_family=arguments.model_family,
        expected_status="selected",
    )
    full_config = _load_yaml(full_path)
    candidate_config = _load_yaml(candidate_path)
    _validate_release_configs(
        full_config,
        candidate_config,
        model_family=arguments.model_family,
        selection=selection,
        selection_sha256=selection_sha,
    )
    evidence_paths = {
        "parity": arguments.parity_report.resolve(strict=True),
        "smoke": arguments.smoke_report.resolve(strict=True),
        "compile": arguments.compile_report.resolve(strict=True),
    }
    if len(set(evidence_paths.values())) != len(evidence_paths):
        raise ValueError("parity, smoke, and compile reports must be distinct files")
    evidence = {
        role: _validate_evidence(
            role, path, arguments.model_family, release_source
        )
        for role, path in evidence_paths.items()
    }
    _validate_compile_report(
        evidence_paths["compile"],
        model_family=arguments.model_family,
        full_path=full_path,
        candidate_path=candidate_path,
        manifest_path=manifest_path,
        expected_source=release_source,
    )
    _validate_smoke_final_contract(
        evidence_paths["smoke"], full_path, candidate_path, arguments.model_family,
        str(evidence["parity"]["sha256"]),
    )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "model_family": arguments.model_family,
        "profile": PROFILE,
        "final_50k_used_for_selection": False,
        "configs": {
            "full": _file_binding(full_path),
            "candidate": _file_binding(candidate_path),
        },
        "manifest": {
            **_file_binding(manifest_path),
            "sidecar": _file_binding(manifest_sidecar),
        },
        "selection": {
            **_file_binding(selection_path),
            "schema_version": selection["schema_version"],
            "status": "selected",
            "passed": True,
        },
        "source": release_source,
        "evidence": evidence,
        "candidate_contract": {
            "rel_l1_thresh": selection["rel_l1_thresh"],
            "gamma_nonfinite_policy": selection["gamma_nonfinite_policy"],
            "probe_depth": 1,
            "batch_size": BATCH_SIZE_BY_MODEL[arguments.model_family],
        },
    }
    require_source_identity_current(
        release_source,
        BASELINE_ROOT,
        PIXARC_ROOT / "third-party" / arguments.model_family,
        context=f"{arguments.model_family} release-gate creation",
    )
    _atomic_create_json(arguments.output, gate)
    return gate


def _verify_binding(binding: Mapping[str, Any], name: str) -> Path:
    path_value = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"invalid {name} binding")
    path = Path(path_value).resolve(strict=True)
    if sha256_file(path) != digest:
        raise ValueError(f"{name} SHA-256 changed after release-gate creation")
    return path


def verify_gate(arguments: argparse.Namespace) -> dict[str, Any]:
    gate_path = arguments.gate.resolve(strict=True)
    gate = load_json_object(gate_path)
    _finite_tree(gate, "release_gate")
    if gate.get("schema_version") != SCHEMA_VERSION or gate.get("passed") is not True:
        raise ValueError("release gate schema/passed contract failed")
    if gate.get("model_family") != arguments.model_family:
        raise ValueError("release gate model_family mismatch")
    if gate.get("profile") != PROFILE:
        raise ValueError("release gate profile mismatch")
    if gate.get("final_50k_used_for_selection") is not False:
        raise ValueError("release gate selection leaked final 50K")
    configs = gate.get("configs")
    manifest_binding = gate.get("manifest")
    selection_binding = gate.get("selection")
    source_binding = gate.get("source")
    evidence = gate.get("evidence")
    if not all(
        isinstance(value, Mapping)
        for value in (
            configs,
            manifest_binding,
            selection_binding,
            source_binding,
            evidence,
        )
    ):
        raise ValueError("release gate is missing binding mappings")
    expected_source = _source_bindings(arguments.model_family)
    if dict(source_binding) != expected_source:
        raise ValueError("release gate source bytes differ from the current port/upstream")
    if set(configs) != {"full", "candidate"}:
        raise ValueError("release gate must bind exactly full and candidate configs")
    full_path = _verify_binding(configs["full"], "full config")
    candidate_path = _verify_binding(configs["candidate"], "candidate config")
    bound_manifest = _verify_binding(manifest_binding, "manifest")
    sidecar_binding = manifest_binding.get("sidecar")
    if not isinstance(sidecar_binding, Mapping):
        raise ValueError("release gate lacks manifest-sidecar binding")
    bound_sidecar = _verify_binding(sidecar_binding, "manifest sidecar")
    if _manifest_sidecar(bound_manifest) != bound_sidecar:
        raise ValueError("bound manifest sidecar path mismatch")
    selection_path = _verify_binding(selection_binding, "selection report")
    if selection_binding.get("status") != "selected" or selection_binding.get("passed") is not True:
        raise ValueError("release gate selection binding is not passed selected")
    selection_sha = sha256_file(selection_path)
    selection = validate_selection_report(
        load_json_object(selection_path),
        expected_model_family=arguments.model_family,
        expected_status="selected",
    )
    if set(evidence) != {"parity", "smoke", "compile"}:
        raise ValueError("release gate must bind parity, smoke, and compile evidence")
    for role in ("parity", "smoke", "compile"):
        path = _verify_binding(evidence[role], f"{role} report")
        validated = _validate_evidence(
            role, path, arguments.model_family, expected_source
        )
        if (
            evidence[role].get("schema_version") != validated["schema_version"]
            or evidence[role].get("passed") is not True
        ):
            raise ValueError(f"release gate {role} metadata mismatch")
    smoke_path = Path(str(evidence["smoke"]["path"])).resolve(strict=True)
    _validate_smoke_final_contract(
        smoke_path, full_path, candidate_path, arguments.model_family,
        str(evidence["parity"]["sha256"]),
    )
    _validate_compile_report(
        Path(str(evidence["compile"]["path"])).resolve(strict=True),
        model_family=arguments.model_family,
        full_path=full_path,
        candidate_path=candidate_path,
        manifest_path=bound_manifest,
        expected_source=expected_source,
    )
    full_config = _load_yaml(full_path)
    candidate_config = _load_yaml(candidate_path)
    _validate_release_configs(
        full_config,
        candidate_config,
        model_family=arguments.model_family,
        selection=selection,
        selection_sha256=selection_sha,
    )
    contract = gate.get("candidate_contract")
    if not isinstance(contract, Mapping) or dict(contract) != {
        "rel_l1_thresh": selection["rel_l1_thresh"],
        "gamma_nonfinite_policy": selection["gamma_nonfinite_policy"],
        "probe_depth": 1,
        "batch_size": BATCH_SIZE_BY_MODEL[arguments.model_family],
    }:
        raise ValueError("release gate candidate contract mismatch")
    supplied_manifest = arguments.manifest.resolve(strict=True)
    if sha256_file(supplied_manifest) != sha256_file(bound_manifest):
        raise ValueError("supplied manifest is not the release-gated manifest")
    supplied_sidecar = _manifest_sidecar(supplied_manifest)
    if sha256_file(supplied_sidecar) != sha256_file(bound_sidecar):
        raise ValueError("supplied manifest sidecar is not release-gated")
    supplied_config_sha = sha256_file(arguments.config.resolve(strict=True))
    allowed = {configs["full"]["sha256"], configs["candidate"]["sha256"]}
    if supplied_config_sha not in allowed:
        raise ValueError("supplied config is neither release-gated full nor candidate")
    smoke_contract = load_json_object(smoke_path).get("execution_contract")
    if not isinstance(smoke_contract, Mapping):
        raise ValueError("smoke gate lacks an execution contract")
    checkpoint = smoke_contract.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("smoke gate lacks checkpoint identity")
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "gate": str(gate_path),
        "config_sha256": supplied_config_sha,
        "manifest_sha256": sha256_file(supplied_manifest),
        "manifest_sidecar_sha256": sha256_file(supplied_sidecar),
        "port_source_sha256": expected_source["port"]["sha256"],
        "upstream_source_sha256": expected_source["upstream"]["sha256"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_size": checkpoint["size"],
        "checkpoint_sha256": checkpoint["sha256"],
    }


def verify_worker_gate(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify the exact archived gate and its semantics inside one shard worker."""

    expected_digest = arguments.expected_gate_sha256
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError("expected worker release-gate SHA-256 must be lowercase hex")
    output_root = arguments.output_root.resolve(strict=True)
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)
    expected_paths = {
        "gate": output_root / "release_gate.json",
        "config": output_root / "config_resolved.yaml",
        "manifest": output_root / "input_manifest.jsonl",
    }
    supplied_paths = {
        "gate": arguments.gate,
        "config": arguments.config,
        "manifest": arguments.manifest,
    }
    resolved: dict[str, Path] = {}
    for role, expected_path in expected_paths.items():
        if expected_path.is_symlink():
            raise ValueError(f"archived worker {role} must not be a symlink")
        expected_resolved = expected_path.resolve(strict=True)
        supplied_resolved = supplied_paths[role].resolve(strict=True)
        if supplied_resolved != expected_resolved:
            raise ValueError(
                f"worker {role} must be the exact output-root archive: {expected_path}"
            )
        resolved[role] = expected_resolved
    sidecar = output_root / "input_manifest.jsonl.meta.json"
    if sidecar.is_symlink():
        raise ValueError("archived worker manifest sidecar must not be a symlink")
    resolved["sidecar"] = sidecar.resolve(strict=True)
    before = {role: sha256_file(path) for role, path in resolved.items()}
    if before["gate"] != expected_digest:
        raise ValueError("archived worker release gate SHA-256 mismatch")
    source_before = _source_bindings(arguments.model_family)
    report = verify_gate(
        argparse.Namespace(
            model_family=arguments.model_family,
            gate=resolved["gate"],
            config=resolved["config"],
            manifest=resolved["manifest"],
        )
    )
    source_after = _source_bindings(arguments.model_family)
    if source_after != source_before:
        raise RuntimeError("release-critical source changed during worker verification")
    after = {role: sha256_file(path) for role, path in resolved.items()}
    if after != before:
        raise RuntimeError("archived worker inputs changed during release-gate verification")
    if after["gate"] != expected_digest:
        raise RuntimeError("archived worker release gate changed during verification")
    return {**report, "release_gate_sha256": expected_digest}


def _atomic_create_json(destination: Path, value: Mapping[str, Any]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite release gate: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="validate artifacts and create a gate")
    create.add_argument("--model-family", required=True, choices=("JiT", "PixelGen"))
    create.add_argument("--full-config", required=True, type=Path)
    create.add_argument("--candidate-config", required=True, type=Path)
    create.add_argument("--manifest", required=True, type=Path)
    create.add_argument("--selection-report", required=True, type=Path)
    create.add_argument("--parity-report", required=True, type=Path)
    create.add_argument("--smoke-report", required=True, type=Path)
    create.add_argument("--compile-report", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="revalidate a gate and all bound files")
    verify.add_argument("--model-family", required=True, choices=("JiT", "PixelGen"))
    verify.add_argument("--gate", required=True, type=Path)
    verify.add_argument("--config", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    worker = subparsers.add_parser(
        "worker-verify", help="verify an exact archived gate inside one shard worker"
    )
    worker.add_argument("--model-family", required=True, choices=("JiT", "PixelGen"))
    worker.add_argument("--gate", required=True, type=Path)
    worker.add_argument("--expected-gate-sha256", required=True)
    worker.add_argument("--config", required=True, type=Path)
    worker.add_argument("--manifest", required=True, type=Path)
    worker.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "create":
        report = create_gate(arguments)
    elif arguments.command == "verify":
        report = verify_gate(arguments)
    else:
        report = verify_worker_gate(arguments)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
