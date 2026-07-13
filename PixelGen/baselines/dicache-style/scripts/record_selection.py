#!/usr/bin/env python3
"""Create and validate an immutable DiCache threshold-selection report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from record_selection_decision import (
    SCHEMA_VERSION as DECISION_SCHEMA_VERSION,
    validate_selection_decision,
)


SCHEMA_VERSION = "pixarc-dicache-selection-v1"
PROFILE = "flux_image_released"
MODEL_FAMILIES = ("JiT", "PixelGen")
STATUSES = ("provisional", "selected")
GAMMA_POLICIES = (
    "official_propagate",
    "latest_residual_fallback",
    "force_full",
)


def load_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {source}")
    return value


def _optional_threshold(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("selection rel_l1_thresh must be a number or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("selection rel_l1_thresh must be finite and non-negative")
    return result


def validate_selection_report(
    report: Mapping[str, Any],
    *,
    expected_model_family: str | None = None,
    expected_status: str | None = None,
    expected_probe_depth: int | None = None,
    expected_threshold: float | None | object = ...,
    expected_gamma_policy: str | None | object = ...,
) -> dict[str, Any]:
    """Validate the complete selection contract and return normalized fields."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"selection report must use schema_version={SCHEMA_VERSION}")
    if report.get("passed") is not True:
        raise ValueError("selection report passed must be exactly true")
    status = report.get("status")
    if status not in STATUSES:
        raise ValueError(f"selection status must be one of {STATUSES}")
    model_family = report.get("model_family")
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"selection model_family must be one of {MODEL_FAMILIES}")
    if report.get("profile") != PROFILE:
        raise ValueError(f"selection profile must be {PROFILE}")
    probe_depth = report.get("probe_depth")
    if isinstance(probe_depth, bool) or probe_depth not in (1, 2, 3):
        raise ValueError("selection probe_depth must be integer 1, 2, or 3")
    if report.get("batch_size") != 1 or isinstance(report.get("batch_size"), bool):
        raise ValueError("selection batch_size must be integer 1")
    if report.get("final_50k_used_for_selection") is not False:
        raise ValueError("final_50k_used_for_selection must be exactly false")

    threshold = _optional_threshold(report.get("rel_l1_thresh"))
    gamma_policy = report.get("gamma_nonfinite_policy")
    if gamma_policy is not None and gamma_policy not in GAMMA_POLICIES:
        raise ValueError(f"unsupported selection gamma_nonfinite_policy: {gamma_policy!r}")
    decision = report.get("decision")
    normalized_decision = None
    if status == "selected":
        if threshold is None or gamma_policy is None:
            raise ValueError("selected reports require a threshold and gamma policy")
        if probe_depth != 1:
            raise ValueError("selected reports require probe_depth=1")
        if not isinstance(decision, Mapping):
            raise ValueError("selected reports require a bound 8K decision report")
        decision_path_value = decision.get("path")
        decision_sha = decision.get("sha256")
        if not isinstance(decision_path_value, str) or not isinstance(decision_sha, str):
            raise ValueError("invalid selection-decision binding")
        decision_path = Path(decision_path_value).resolve(strict=True)
        actual_sha = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        if actual_sha != decision_sha:
            raise ValueError("selection-decision SHA-256 changed after selection")
        normalized_decision_report = validate_selection_decision(
            load_json_object(decision_path),
            expected_model_family=model_family,
            expected_threshold=threshold,
            expected_gamma_policy=gamma_policy,
        )
        if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
            raise ValueError("selection-decision binding schema mismatch")
        normalized_decision = {
            "path": str(decision_path),
            "sha256": actual_sha,
            "schema_version": normalized_decision_report["schema_version"],
            "passed": True,
        }
    elif decision is not None:
        raise ValueError("provisional reports may not claim a selection decision")
    if expected_model_family is not None and model_family != expected_model_family:
        raise ValueError(
            f"selection model_family mismatch: {model_family!r} != {expected_model_family!r}"
        )
    if expected_status is not None and status != expected_status:
        raise ValueError(f"selection status mismatch: {status!r} != {expected_status!r}")
    if expected_probe_depth is not None and probe_depth != expected_probe_depth:
        raise ValueError("selection probe_depth does not match the materialized config")
    if expected_threshold is not ...:
        normalized_expected = _optional_threshold(expected_threshold)
        if threshold != normalized_expected:
            raise ValueError(
                "selection rel_l1_thresh does not exactly match the materialized config"
            )
    if expected_gamma_policy is not ... and gamma_policy != expected_gamma_policy:
        raise ValueError(
            "selection gamma_nonfinite_policy does not exactly match the materialized config"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "status": status,
        "model_family": model_family,
        "profile": PROFILE,
        "probe_depth": probe_depth,
        "batch_size": 1,
        "rel_l1_thresh": threshold,
        "gamma_nonfinite_policy": gamma_policy,
        "final_50k_used_for_selection": False,
        "decision": normalized_decision,
    }


def _atomic_create_json(destination: Path, value: Mapping[str, Any]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selection report: {destination}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=MODEL_FAMILIES)
    parser.add_argument("--status", required=True, choices=STATUSES)
    parser.add_argument("--probe-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--rel-l1-thresh", type=float)
    parser.add_argument("--gamma-nonfinite-policy", choices=GAMMA_POLICIES)
    parser.add_argument("--decision-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    decision = None
    if args.decision_report is not None:
        decision_path = args.decision_report.resolve(strict=True)
        decision = {
            "path": str(decision_path),
            "sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            "schema_version": DECISION_SCHEMA_VERSION,
            "passed": True,
        }
    report = validate_selection_report(
        {
            "schema_version": SCHEMA_VERSION,
            "passed": True,
            "status": args.status,
            "model_family": args.model_family,
            "profile": PROFILE,
            "probe_depth": args.probe_depth,
            "batch_size": 1,
            "rel_l1_thresh": args.rel_l1_thresh,
            "gamma_nonfinite_policy": args.gamma_nonfinite_policy,
            "final_50k_used_for_selection": False,
            "decision": decision,
        }
    )
    _atomic_create_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
