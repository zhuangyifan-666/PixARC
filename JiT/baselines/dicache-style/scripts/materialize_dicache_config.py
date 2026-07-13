#!/usr/bin/env python3
"""Materialize a fail-closed candidate config from a frozen selection report."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml

from dicache_style.manifest import sha256_file
from dicache_style.metadata import validate_dicache_config
from record_selection import load_json_object, validate_selection_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="existing checkpoint written as an absolute path into the resolved config",
    )
    parser.add_argument("--rel-l1-thresh", required=True, type=float)
    parser.add_argument(
        "--gamma-nonfinite-policy",
        required=True,
        choices=("official_propagate", "latest_residual_fallback", "force_full"),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("matched_eager", "blockwise", "upstream"),
        help="optional explicit runtime.compile_mode override",
    )
    parser.add_argument(
        "--probe-depth",
        type=int,
        choices=(1, 2, 3),
        help="optional probe-depth diagnostic; only shadow/probe-only modes",
    )
    parser.add_argument("--selection-report", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve(strict=True)
    report = Path(args.selection_report).resolve(strict=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite config: {destination}")
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("model"), dict)
        or not isinstance(config.get("dicache"), dict)
    ):
        raise ValueError("input is not a DiCache generation config")
    config["model"]["checkpoint"] = str(checkpoint)
    config["dicache"]["rel_l1_thresh"] = args.rel_l1_thresh
    config["dicache"]["gamma_nonfinite_policy"] = args.gamma_nonfinite_policy
    if args.probe_depth is not None:
        if config["dicache"]["mode"] not in {
            "probe_shadow_full",
            "probe_only_ablation",
        }:
            raise ValueError(
                "--probe-depth is allowed only for probe_shadow_full or probe_only_ablation"
            )
        config["dicache"]["probe_depth"] = args.probe_depth
    if args.compile_mode is not None:
        if not isinstance(config.get("runtime"), dict):
            raise ValueError("input config has no runtime mapping")
        if args.compile_mode == "upstream" and config["dicache"]["mode"] != "upstream_full":
            raise ValueError("compile_mode=upstream is valid only for upstream_full")
        config["runtime"]["compile_mode"] = args.compile_mode
    validate_dicache_config(config["dicache"], require_resolved=True)
    if config["dicache"].get("profile") != "flux_image_released":
        raise ValueError("selection provenance requires profile=flux_image_released")
    if config.get("runtime", {}).get("batch_size") != 1:
        raise ValueError("selection provenance requires runtime.batch_size=1")
    selection = validate_selection_report(
        load_json_object(report),
        expected_model_family="JiT",
        expected_probe_depth=config["dicache"].get("probe_depth"),
        expected_threshold=config["dicache"].get("rel_l1_thresh"),
        expected_gamma_policy=config["dicache"].get("gamma_nonfinite_policy"),
    )
    if selection["status"] == "selected":
        if config["dicache"].get("mode") != "dicache":
            raise ValueError("selected reports may materialize only mode=dicache")
        if config["dicache"].get("probe_depth") != 1:
            raise ValueError("selected reports require probe_depth=1")
    config["selection_provenance"] = {
        "schema_version": selection["schema_version"],
        "selection_report_sha256": sha256_file(report),
        "selection_report_name": report.name,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(destination)


if __name__ == "__main__":
    main()
