#!/usr/bin/env python3
"""Validate PRT images, metadata, immutable inputs, traces, and forward counts."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT" / "baselines" / "taylorseer-style"
for item in (METHOD_ROOT, BASELINE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pixel_remainder_taylor.config import load_config  # noqa: E402
from pixel_remainder_taylor.protocol import (  # noqa: E402
    executable_tree_sha256,
    resolve_manifest_sidecar,
)
from taylorseer_style.image_io import load_rank_metadata, validate_outputs  # noqa: E402
from taylorseer_style.manifest import (  # noqa: E402
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
)
from taylorseer_style.metadata import (  # noqa: E402
    PAIRING_FIELDS,
    canonical_hash,
    load_json,
    source_tree_sha256,
)


def _assert_finite_numbers(value: object, path: str = "trace") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite numeric value at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_numbers(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_numbers(item, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--resolution", default=256, type=int)
    args = parser.parse_args()
    root = Path(args.run_root).resolve(strict=True)
    manifest_path = Path(args.manifest).resolve(strict=True)
    records = load_manifest(manifest_path)
    validate_manifest(records, expected_count=args.expected_count)
    run = load_json(root / "run_manifest.json")
    config = load_config(root / "config_resolved.yaml")
    if run.get("input_config_hash") != canonical_hash(config):
        raise ValueError("run manifest is not bound to config_resolved.yaml")
    missing_pairing = [field for field in PAIRING_FIELDS if field not in run]
    if missing_pairing:
        raise ValueError(f"run manifest is missing pairing fields: {missing_pairing}")
    if run.get("port_source_sha256") != source_tree_sha256(BASELINE_ROOT):
        raise ValueError("run manifest TaylorSeer port source hash is stale")
    method_source = executable_tree_sha256(METHOD_ROOT)
    if run.get("model") == "PixelGen-JiT":
        pixelgen_root = PIXARC_ROOT / "PixelGen" / "methods" / "pixel-remainder-taylor"
        method_source = canonical_hash({
            "shared": method_source,
            "pixelgen_adapter": executable_tree_sha256(pixelgen_root),
        })
    if run.get("method_source_sha256") != method_source:
        raise ValueError("run manifest method source hash is stale")
    if run.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("run manifest is not bound to supplied manifest bytes")
    if run.get("manifest_records_sha256") != manifest_records_sha256(records):
        raise ValueError("run manifest is not bound to canonical manifest records")
    if sha256_file(root / "input_manifest.jsonl") != sha256_file(manifest_path):
        raise ValueError("archived manifest differs from supplied manifest")
    if sha256_file(root / "input_manifest.jsonl.meta.json") != sha256_file(
        resolve_manifest_sidecar(manifest_path)
    ):
        raise ValueError("archived manifest sidecar differs from supplied sidecar")
    world_size = int(run["world_size"])
    metadata = load_rank_metadata(root / "metadata", records, world_size=world_size)
    validation = validate_outputs(
        root / "samples", records, metadata=metadata,
        expected_count=args.expected_count, resolution=args.resolution,
    )
    expected_identity = {
        "config_hash": run["input_config_hash"], "checkpoint_path": run["checkpoint_path"],
        "checkpoint_size": run["checkpoint_size"], "manifest_sha256": run["manifest_sha256"],
        "method": run["method"], "interval": int(run["identity_interval"]),
        "max_order": int(run["identity_max_order"]),
        "coordinate_mode": run["coordinate_mode"],
        "resolution": args.resolution,
    }
    errors = [
        f"{key}: output={validation.get(key)!r}, expected={value!r}"
        for key, value in expected_identity.items() if validation.get(key) != value
    ]
    if errors:
        raise ValueError("output identity mismatch:\n- " + "\n- ".join(errors))
    group_count = len({row.batch_group_id for row in records})
    trace_rows = []
    for rank in range(world_size):
        path = root / "traces" / f"rank_{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            trace_rows.extend(json.loads(line) for line in handle if line.strip())
    if len(trace_rows) != group_count:
        raise ValueError(f"trace count {len(trace_rows)} != batch-group count {group_count}")
    expected_forwards = int(run["expected_network_forwards_per_trajectory"])
    traced_sample_ids: list[int] = []
    dynamic_plan_seen = False
    dynamic_taylor_seen = False
    for trace in trace_rows:
        _assert_finite_numbers(trace)
        if trace.get("call_count_valid") is not True:
            raise ValueError("trajectory has invalid call count")
        if int(trace.get("total_nfe", 0)) != 99:
            raise ValueError("trajectory does not have 99 NFE")
        if int(trace.get("network_forward_count", -1)) != expected_forwards:
            raise ValueError("trajectory has an extra or missing model forward")
        full_nfe = int(trace.get("full_nfe", -1))
        taylor_nfe = int(trace.get("taylor_nfe", -1))
        order1_nfe = int(trace.get("order1_taylor_nfe", -1))
        order2_nfe = int(trace.get("order2_taylor_nfe", -1))
        if full_nfe + taylor_nfe != 99:
            raise ValueError("trajectory Full/Taylor counts do not total 99")
        if order1_nfe + order2_nfe != taylor_nfe:
            raise ValueError("trajectory Taylor order counts are inconsistent")
        trace_mode = trace.get("trace_mode", run.get("trace_mode", "full"))
        nested = trace.get("nfe_trace")
        if trace_mode == "full":
            if not isinstance(nested, list) or len(nested) != 99:
                raise ValueError("trajectory is missing its full per-NFE trace")
        elif trace_mode == "summary":
            if nested is not None:
                raise ValueError("summary trace contains a nested per-NFE trace")
            for field in (
                "span_histogram", "stage_statistics", "risk_summary",
                "controller_time_ms", "cache_bytes", "cache_tensor_count",
            ):
                if field not in trace:
                    raise ValueError(f"summary trace is missing {field}")
        else:
            raise ValueError(f"unknown trace_mode {trace_mode!r}")
        dynamic_taylor_seen |= int(trace.get("taylor_nfe", 0)) > 0
        traced_sample_ids.extend(int(value) for value in trace.get("sample_ids", []))
        for nfe in nested or []:
            required = {
                "nfe_index", "macro_step_index", "solver_stage", "continuous_t",
                "action", "full_reason", "active_forecast_order",
                "remaining_taylor_before", "remaining_taylor_after", "planned_span",
                "available_feature_order_min", "pixel_available_order", "tau",
                "max_taylor_span",
            }
            missing = required - set(nfe)
            if missing:
                raise ValueError(f"per-NFE trace is missing fields: {sorted(missing)}")
            dynamic_plan_seen |= int(nfe.get("selected_span", 0)) > 0
        dynamic_plan_seen |= int(trace.get("max_planned_span", 0)) > 0
    expected_sample_ids = sorted(row.sample_id for row in records)
    if sorted(traced_sample_ids) != expected_sample_ids:
        raise ValueError("trajectory traces do not cover each real sample ID exactly once")
    if run["method"] == "pixel_remainder_taylor" and (
        not dynamic_plan_seen or not dynamic_taylor_seen
    ):
        raise ValueError(
            "adaptive run never executed a nonzero dynamic Taylor segment"
        )
    print(json.dumps({
        **validation, "identity_validation": "passed",
        "trajectory_count": len(trace_rows), "forward_contract": "passed",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
