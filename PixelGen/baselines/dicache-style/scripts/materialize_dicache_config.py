#!/usr/bin/env python3
"""Materialize one explicit DiCache threshold into a new immutable YAML config."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.metadata import canonical_hash, validate_dicache_config  # noqa: E402
from dicache_style.manifest import sha256_file  # noqa: E402
from record_selection import load_json_object, validate_selection_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--compile-mode",
        choices=("matched_eager", "blockwise", "upstream"),
    )
    parser.add_argument("--probe-depth", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--gamma-nonfinite-policy",
        choices=("official_propagate", "latest_residual_fallback", "force_full"),
    )
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threshold is not None and (
        not math.isfinite(args.threshold) or args.threshold < 0
    ):
        parser.error("--threshold must be finite and non-negative")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace materialized config: {args.output}")
    with args.base.resolve(strict=True).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError("base must be a pixarc-dicache-config-v1 mapping")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config["checkpoint"] = str(checkpoint)
    dicache = dict(config["dicache"])
    if args.probe_depth is not None:
        if dicache.get("mode") not in {
            "probe_shadow_full",
            "probe_only_ablation",
        }:
            parser.error(
                "--probe-depth is restricted to probe_shadow_full or "
                "probe_only_ablation configs"
            )
        dicache["probe_depth"] = int(args.probe_depth)
    if args.threshold is not None:
        dicache["rel_l1_thresh"] = float(args.threshold)
    if args.gamma_nonfinite_policy is not None:
        dicache["gamma_nonfinite_policy"] = args.gamma_nonfinite_policy
    validate_dicache_config(dicache, require_resolved=True)
    config["dicache"] = dicache
    runtime = dict(config.get("runtime", {}))
    compile_mode = str(
        args.compile_mode
        if args.compile_mode is not None
        else runtime.get("compile_mode", "matched_eager")
    )
    if compile_mode not in {"matched_eager", "blockwise", "upstream"}:
        raise ValueError(f"unsupported runtime.compile_mode: {compile_mode!r}")
    if compile_mode == "upstream" and dicache["mode"] != "upstream_full":
        parser.error("--compile-mode upstream is valid only for upstream_full")
    runtime["compile_mode"] = compile_mode
    config["runtime"] = runtime
    model = config["model"]
    model["compile_mode"] = compile_mode
    denoiser = config["model"]["denoiser"]
    denoiser["init_args"]["compile_mode"] = compile_mode
    if args.probe_depth is not None:
        denoiser["init_args"]["dicache_probe_depth"] = int(args.probe_depth)
    if args.threshold is not None:
        denoiser["init_args"]["dicache_rel_l1_thresh"] = float(args.threshold)
    if args.gamma_nonfinite_policy is not None:
        denoiser["init_args"]["dicache_gamma_nonfinite_policy"] = args.gamma_nonfinite_policy
    if dicache.get("profile") != "flux_image_released":
        raise ValueError("selection provenance requires profile=flux_image_released")
    if runtime.get("batch_size") != 1:
        raise ValueError("selection provenance requires runtime.batch_size=1")
    selection_path = args.selection_report.resolve(strict=True)
    selection = validate_selection_report(
        load_json_object(selection_path),
        expected_model_family="PixelGen",
        expected_probe_depth=dicache.get("probe_depth"),
        expected_threshold=dicache.get("rel_l1_thresh"),
        expected_gamma_policy=dicache.get("gamma_nonfinite_policy"),
    )
    if selection["status"] == "selected":
        if dicache.get("mode") != "dicache":
            raise ValueError("selected reports may materialize only mode=dicache")
        if dicache.get("probe_depth") != 1:
            raise ValueError("selected reports require probe_depth=1")
    config["selection_provenance"] = {
        "schema_version": selection["schema_version"],
        "selection_report_sha256": sha256_file(selection_path),
        "selection_report_name": selection_path.name,
        "status": selection["status"],
        "passed": selection["passed"],
        "model_family": selection["model_family"],
        "profile": selection["profile"],
        "probe_depth": selection["probe_depth"],
        "batch_size": selection["batch_size"],
        "rel_l1_thresh": selection["rel_l1_thresh"],
        "gamma_nonfinite_policy": selection["gamma_nonfinite_policy"],
        "final_50k_used_for_selection": selection[
            "final_50k_used_for_selection"
        ],
        "decision": selection["decision"],
        "threshold_selected_before_final_50k": selection["status"] == "selected",
        "gamma_policy_preregistered": selection["status"] == "selected",
        "checkpoint_resolved_absolute": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "checkpoint": str(checkpoint),
        "rel_l1_thresh": dicache["rel_l1_thresh"],
        "gamma_nonfinite_policy": dicache["gamma_nonfinite_policy"],
        "probe_depth": dicache["probe_depth"],
        "compile_mode": compile_mode,
        "config_hash": canonical_hash(config),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
