#!/usr/bin/env python3
"""Evaluate one measured PRT 1K run with the repository's metric primitives."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT" / "baselines" / "taylorseer-style"
for item in (METHOD_ROOT, BASELINE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from aggregate_traces import _read_jsonl, aggregate  # noqa: E402
from pixel_remainder_taylor.protocol import resolve_manifest_sidecar  # noqa: E402
from taylorseer_style.distribution_metrics import (  # noqa: E402
    build_adm_sample_npz,
    run_adm_evaluator,
)
from taylorseer_style.image_io import image_path  # noqa: E402
from taylorseer_style.manifest import (  # noqa: E402
    load_manifest,
    sha256_file,
    validate_manifest,
)
from taylorseer_style.metadata import (  # noqa: E402
    PAIRING_FIELDS,
    atomic_write_json,
    load_json,
)
from taylorseer_style.paired_metrics import (  # noqa: E402
    _lpips_values,
    _summary,
    load_rgb_float,
    pair_metrics,
    psnr_from_mse,
    validate_pair_manifests,
    write_rows_csv,
)


QUALITY_KEYS = ("fid", "sfid", "inception_score", "precision", "recall")
PAIRING_PROTOCOL_FIELDS = tuple(
    field for field in PAIRING_FIELDS if field != "git_commit"
)


def _baseline_row(path: Path, model: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["model"] == model and row["run"] == "full"]
    if len(matches) != 1:
        raise ValueError(f"expected one TaylorSeer Full row for {model}, found {len(matches)}")
    return matches[0]


def _write_one(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)


def _candidate_trace_files(patterns: list[str], trace_root: Path) -> list[str]:
    files: list[Path] = []
    for pattern in patterns:
        matches = [Path(value).resolve(strict=True) for value in sorted(glob.glob(pattern))]
        if not matches:
            raise FileNotFoundError(f"trace pattern matched nothing: {pattern}")
        files.extend(matches)
    for path in files:
        if trace_root not in path.parents:
            raise ValueError(f"trace is outside candidate trace root: {path}")
    if len(files) != len(set(files)):
        raise ValueError("duplicate trace files were supplied")
    return [str(path) for path in files]


def validate_pairing_protocol(
    candidate: dict[str, object], reference: dict[str, object]
) -> None:
    pairing_errors = []
    for field in PAIRING_PROTOCOL_FIELDS:
        if field not in candidate or field not in reference:
            pairing_errors.append(f"missing required pairing field {field!r}")
        elif candidate[field] != reference[field]:
            pairing_errors.append(
                f"{field}: candidate={candidate[field]!r}, reference={reference[field]!r}"
            )
    if pairing_errors:
        raise ValueError(
            "candidate/reference protocol mismatch:\n- "
            + "\n- ".join(pairing_errors)
        )


def _paired(
    reference_dir: Path,
    candidate_dir: Path,
    reference_manifest_path: Path,
    candidate_manifest_path: Path,
    *,
    resolution: int,
    include_lpips: bool,
    lpips_device: str,
    lpips_batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    reference_manifest = load_manifest(reference_manifest_path)
    candidate_manifest = load_manifest(candidate_manifest_path)
    sample_ids = validate_pair_manifests(reference_manifest, candidate_manifest)
    index = {row.sample_id: row for row in reference_manifest}
    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    pairs: list[tuple[Path, Path]] = []
    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        reference_path = image_path(reference_dir, sample_id)
        candidate_path = image_path(candidate_dir, sample_id)
        if not reference_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(f"missing paired PNG for sample {sample_id}")
        reference = load_rgb_float(reference_path, resolution)
        candidate = load_rgb_float(candidate_path, resolution)
        mse, psnr, ssim = pair_metrics(reference, candidate)
        mse_values.append(mse); psnr_values.append(psnr); ssim_values.append(ssim)
        pairs.append((reference_path, candidate_path))
        manifest_row = index[sample_id]
        rows.append({
            "sample_id": sample_id, "class_id": manifest_row.class_id,
            "seed": manifest_row.seed, "psnr": psnr, "ssim": ssim, "lpips": None,
            "reference_path": str(reference_path), "candidate_path": str(candidate_path),
        })
    lpips_values: list[float] = []
    lpips_version = None
    if include_lpips:
        lpips_values, lpips_version = _lpips_values(
            pairs, resolution=resolution, device=lpips_device, batch_size=lpips_batch_size
        )
        for row, value in zip(rows, lpips_values, strict=True):
            row["lpips"] = value
    aggregate_mse = float(np.mean(np.asarray(mse_values, dtype=np.float64)))
    result: dict[str, object] = {
        "sample_count": len(rows), "aggregate_mse": aggregate_mse,
        "psnr_from_aggregate_mse": psnr_from_mse(aggregate_mse),
        "per_image_psnr": _summary(psnr_values), "ssim": _summary(ssim_values),
        "exact_pair_count": sum(value == 0.0 for value in mse_values),
        "nan_counts": {
            "psnr": sum(math.isnan(value) for value in psnr_values),
            "ssim": sum(math.isnan(value) for value in ssim_values),
            "lpips": sum(math.isnan(value) for value in lpips_values),
        },
        "inf_counts": {
            "psnr": sum(math.isinf(value) for value in psnr_values),
            "ssim": sum(math.isinf(value) for value in ssim_values),
            "lpips": sum(math.isinf(value) for value in lpips_values),
        },
    }
    if include_lpips:
        result["lpips"] = {**_summary(lpips_values), "package_version": lpips_version,
                           "backbone": "alex", "device": lpips_device}
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("JiT", "PixelGen"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--reference-npz", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--timing")
    parser.add_argument("--baseline-summary", default="results/taylorseer_1k_summary.csv")
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--lpips-batch-size", default=16, type=int)
    parser.add_argument("--resolution", default=256, type=int)
    args = parser.parse_args()
    candidate_root = Path(args.candidate_root).resolve(strict=True)
    reference_root = Path(args.reference_root).resolve(strict=True)
    manifest_path = Path(args.manifest).resolve(strict=True)
    reference_manifest_path = Path(args.reference_manifest).resolve(strict=True)
    records = load_manifest(manifest_path)
    validate_manifest(records, expected_count=1000, expected_per_class=1, expected_num_classes=1000)
    run = load_json(candidate_root / "run_manifest.json")
    reference_run = load_json(reference_root / "run_manifest.json")
    if run.get("method") != "pixel_remainder_taylor":
        raise ValueError("candidate run is not pixel_remainder_taylor")
    if reference_run.get("method") not in {"upstream_full", "instrumented_full"}:
        raise ValueError("reference root is not an existing matched Full run")
    validate_pairing_protocol(run, reference_run)
    expected_model_identity = {"JiT": "JiT-B/16", "PixelGen": "PixelGen-JiT"}
    if run.get("model") != expected_model_identity[args.model]:
        raise ValueError("--model does not match candidate run identity")
    if run.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("candidate run is not bound to --manifest")
    if reference_run.get("manifest_sha256") != sha256_file(reference_manifest_path):
        raise ValueError("reference run is not bound to --reference-manifest")
    if run.get("manifest_sidecar_sha256") != sha256_file(
        resolve_manifest_sidecar(manifest_path)
    ):
        raise ValueError("candidate manifest sidecar identity mismatch")
    if reference_run.get("manifest_sidecar_sha256") != sha256_file(
        resolve_manifest_sidecar(reference_manifest_path)
    ):
        raise ValueError("reference manifest sidecar identity mismatch")
    if run.get("manifest_sha256") is None or int(run.get("expected_nfe_per_trajectory", 0)) != 99:
        raise ValueError("candidate run identity or NFE contract is incomplete")
    trace_files = _candidate_trace_files(
        args.trace, (candidate_root / "traces").resolve(strict=True)
    )
    trace = aggregate(_read_jsonl(trace_files), model=args.model, run=args.run)
    if trace["sample_count"] != 1000:
        raise ValueError(f"trace covers {trace['sample_count']} samples, expected 1000")
    if trace.get("tau") != run.get("tau") or trace.get("max_taylor_span") != run.get("max_taylor_span"):
        raise ValueError("trace settings do not match candidate run manifest")

    paired, pair_rows = _paired(
        reference_root / "samples", candidate_root / "samples",
        reference_manifest_path, manifest_path, resolution=args.resolution,
        include_lpips=not args.skip_lpips, lpips_device=args.lpips_device,
        lpips_batch_size=args.lpips_batch_size,
    )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    sample_npz = output / f"{args.model.lower()}_{args.run}.samples.npz"
    build_adm_sample_npz(
        sample_dir=candidate_root / "samples", manifest=records,
        output_npz=sample_npz, resolution=args.resolution,
    )
    distribution, raw_output = run_adm_evaluator(
        evaluator=args.evaluator, reference_npz=args.reference_npz, sample_npz=sample_npz
    )
    (output / f"{args.model.lower()}_{args.run}_adm_stdout.txt").write_text(raw_output, encoding="utf-8")
    baseline = _baseline_row(Path(args.baseline_summary), args.model)
    reference_timing = load_json(reference_root / "four_gpu_wall_clock.json")
    if reference_timing.get("completed") is not True:
        raise ValueError("reference Full timing is not complete")
    if not math.isclose(
        float(reference_timing["elapsed_seconds"]),
        float(baseline["elapsed_seconds"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("reference Full timing does not match frozen summary CSV")
    timing_path = (
        Path(args.timing).resolve(strict=True)
        if args.timing
        else candidate_root / "launcher_timing.json"
    )
    timing = load_json(timing_path)
    if (
        timing.get("completed") is not True
        or timing.get("timing_provenance_complete") is not True
        or int(timing.get("cumulative_sample_count", -1)) != 1000
        or timing.get("manifest_sha256") != run.get("manifest_sha256")
    ):
        raise ValueError("candidate cumulative launcher timing is incomplete or unbound")
    elapsed = float(timing["cumulative_elapsed_seconds"])
    if args.elapsed_seconds is not None and not math.isclose(
        float(args.elapsed_seconds), elapsed, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("--elapsed-seconds differs from bound cumulative timing")
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("elapsed-seconds must be finite and positive")
    row: dict[str, object] = {
        "model": args.model, "run": args.run, "method": "pixel_remainder_taylor",
        "tau": trace["tau"], "max_taylor_span": trace["max_taylor_span"],
        "sample_count": 1000, "elapsed_seconds": elapsed,
        "images_per_second": 1000.0 / elapsed,
        "speedup_vs_full": float(baseline["elapsed_seconds"]) / elapsed,
        "launcher_invocation_count": timing["invocation_count"],
        "reference_npz_sha256": sha256_file(args.reference_npz),
        "evaluator_sha256": sha256_file(args.evaluator),
    }
    for key in QUALITY_KEYS:
        row[key] = distribution[key]
        baseline_key = "inception_score" if key == "inception_score" else key
        row[f"delta_{'inception_score' if key == 'inception_score' else key}"] = (
            float(distribution[key]) - float(baseline[baseline_key])
        )
    row.update({
        "aggregate_mse": paired["aggregate_mse"],
        "psnr_from_aggregate_mse": paired["psnr_from_aggregate_mse"],
        "mean_per_image_psnr": paired["per_image_psnr"]["mean"],
        "median_per_image_psnr": paired["per_image_psnr"]["median"],
        "mean_ssim": paired["ssim"]["mean"], "median_ssim": paired["ssim"]["median"],
        "mean_lpips": paired.get("lpips", {}).get("mean", ""),
        "median_lpips": paired.get("lpips", {}).get("median", ""),
        "exact_pair_count": paired["exact_pair_count"],
        "nan_psnr": paired["nan_counts"]["psnr"], "nan_ssim": paired["nan_counts"]["ssim"],
        "nan_lpips": paired["nan_counts"]["lpips"], "inf_psnr": paired["inf_counts"]["psnr"],
        "inf_ssim": paired["inf_counts"]["ssim"], "inf_lpips": paired["inf_counts"]["lpips"],
    })
    prefix = f"{args.model.lower()}_{args.run}"
    _write_one(output / f"{prefix}_summary.csv", row)
    _write_one(output / f"{prefix}_trace.csv", trace)
    atomic_write_json(output / f"{prefix}_paired.json", paired)
    atomic_write_json(output / f"{prefix}_distribution.json", distribution)
    write_rows_csv(output / f"{prefix}_paired_samples.csv", pair_rows)
    print(json.dumps({"summary": row, "trace": trace}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
