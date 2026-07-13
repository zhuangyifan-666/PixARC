"""Strictly paired RGB-PNG PSNR, SSIM, and optional AlexNet LPIPS."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .image_io import (
    image_path,
    load_rank_metadata,
    validate_output_run_identity,
    validate_outputs,
)
from .manifest import (
    ManifestRecord,
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
)
from .metadata import (
    PAIRING_FIELDS,
    DICACHE_CONFIG_FIELDS,
    atomic_write_json,
    load_json,
    validate_full_dicache_roles,
    validate_paired_runs,
    validate_run_artifacts,
)


COMMON_IMPLEMENTATION_VERSION = "pixarc-dicache-style-v1"


def _manifest_index(
    records: Sequence[ManifestRecord], name: str
) -> dict[int, ManifestRecord]:
    result: dict[int, ManifestRecord] = {}
    for record in records:
        if record.sample_id in result:
            raise ValueError(f"duplicate sample ID in {name} manifest: {record.sample_id}")
        result[record.sample_id] = record
    return result


def validate_pair_manifests(
    reference: Sequence[ManifestRecord], candidate: Sequence[ManifestRecord]
) -> list[int]:
    ref = _manifest_index(reference, "reference")
    cand = _manifest_index(candidate, "candidate")
    if set(ref) != set(cand):
        raise ValueError(
            f"sample IDs differ: missing={len(set(ref)-set(cand))}, "
            f"extra={len(set(cand)-set(ref))}"
        )
    for sample_id in sorted(ref):
        if ref[sample_id].class_id != cand[sample_id].class_id:
            raise ValueError(f"class mismatch for sample {sample_id}")
        if ref[sample_id].seed != cand[sample_id].seed:
            raise ValueError(f"seed mismatch for sample {sample_id}")
    return sorted(ref)


def load_rgb_float(path: str | os.PathLike[str], resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (resolution, resolution):
            raise ValueError(f"expected RGB {resolution}x{resolution}: {path}")
        array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (resolution, resolution, 3):
        raise ValueError(f"expected uint8 HWC image: {path}")
    return array.astype(np.float32) / 255.0


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def pair_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"paired image shapes differ: {reference.shape} != {candidate.shape}")
    if reference.dtype != np.float32 or candidate.dtype != np.float32:
        raise TypeError("paired images must be float32 [0,1]")
    difference = reference - candidate
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    psnr = psnr_from_mse(mse)
    ssim = float(
        structural_similarity(
            reference,
            candidate,
            channel_axis=-1,
            data_range=1.0,
            win_size=7,
        )
    )
    return mse, psnr, ssim


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    if math.isinf(ordered[high]):
        return ordered[high]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _summary(values: Sequence[float]) -> dict[str, float]:
    finite_or_inf = [float(value) for value in values]
    return {
        "mean": float(np.mean(np.asarray(finite_or_inf, dtype=np.float64))),
        "median": _percentile(finite_or_inf, 50),
        "p90": _percentile(finite_or_inf, 90),
        "p95": _percentile(finite_or_inf, 95),
        "p99": _percentile(finite_or_inf, 99),
    }


def _lpips_values(
    pairs: Sequence[tuple[Path, Path]],
    *,
    resolution: int,
    device: str,
    batch_size: int,
) -> tuple[list[float], str]:
    if batch_size <= 0:
        raise ValueError("LPIPS batch_size must be positive")
    try:
        import lpips
        import torch
    except ImportError as error:
        raise RuntimeError(
            "LPIPS requested but the 'lpips' package is unavailable; see requirements-extra.txt"
        ) from error
    # torchvision otherwise downloads AlexNet weights implicitly.  Require the
    # canonical local cache entry so this evaluator is strictly offline.
    alexnet_weights = (
        Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
    )
    if not alexnet_weights.is_file():
        raise FileNotFoundError(
            "LPIPS AlexNet weights are absent from the local torch cache: "
            f"{alexnet_weights}; no download was attempted"
        )
    model = lpips.LPIPS(net="alex", spatial=False).to(device).eval()
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            reference = torch.from_numpy(
                np.stack(
                    [load_rgb_float(pair[0], resolution) for pair in chunk]
                ).transpose(0, 3, 1, 2)
            ).to(device=device, dtype=torch.float32)
            candidate = torch.from_numpy(
                np.stack(
                    [load_rgb_float(pair[1], resolution) for pair in chunk]
                ).transpose(0, 3, 1, 2)
            ).to(device=device, dtype=torch.float32)
            result = model(reference * 2.0 - 1.0, candidate * 2.0 - 1.0)
            values.extend(float(value) for value in result.reshape(-1).detach().cpu())
    try:
        version = importlib.metadata.version("lpips")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return values, version


def evaluate_paired(
    *,
    reference_dir: str | os.PathLike[str],
    candidate_dir: str | os.PathLike[str],
    reference_manifest: Sequence[ManifestRecord],
    candidate_manifest: Sequence[ManifestRecord],
    reference_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    reference_manifest_path: str | os.PathLike[str],
    candidate_manifest_path: str | os.PathLike[str],
    reference_run_metadata_path: str | os.PathLike[str],
    candidate_run_metadata_path: str | os.PathLike[str],
    resolution: int = 256,
    include_lpips: bool = True,
    lpips_device: str = "cpu",
    lpips_batch_size: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if list(reference_manifest) != load_manifest(reference_manifest_path):
        raise ValueError("reference manifest records differ from the on-disk manifest")
    if list(candidate_manifest) != load_manifest(candidate_manifest_path):
        raise ValueError("candidate manifest records differ from the on-disk manifest")
    if dict(reference_run) != load_json(reference_run_metadata_path):
        raise ValueError("reference run mapping differs from on-disk run metadata")
    if dict(candidate_run) != load_json(candidate_run_metadata_path):
        raise ValueError("candidate run mapping differs from on-disk run metadata")
    reference_manifest_sha256 = sha256_file(reference_manifest_path)
    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    if reference_run.get("manifest_sha256") != reference_manifest_sha256:
        raise ValueError("reference run metadata is not bound to reference manifest")
    if candidate_run.get("manifest_sha256") != candidate_manifest_sha256:
        raise ValueError("candidate run metadata is not bound to candidate manifest")
    if reference_run.get("manifest_records_sha256") != manifest_records_sha256(
        reference_manifest
    ):
        raise ValueError("reference run metadata does not bind manifest record content")
    if candidate_run.get("manifest_records_sha256") != manifest_records_sha256(
        candidate_manifest
    ):
        raise ValueError("candidate run metadata does not bind manifest record content")
    validate_paired_runs(reference_run, candidate_run)
    validate_full_dicache_roles(reference_run, candidate_run)
    sample_ids = validate_pair_manifests(reference_manifest, candidate_manifest)
    reference_root = validate_run_artifacts(
        run_metadata_path=reference_run_metadata_path,
        sample_dir=reference_dir,
        supplied_manifest_path=reference_manifest_path,
        run_metadata=reference_run,
        manifest_records_sha256=manifest_records_sha256(reference_manifest),
    )
    candidate_root = validate_run_artifacts(
        run_metadata_path=candidate_run_metadata_path,
        sample_dir=candidate_dir,
        supplied_manifest_path=candidate_manifest_path,
        run_metadata=candidate_run,
        manifest_records_sha256=manifest_records_sha256(candidate_manifest),
    )
    reference_world_size = reference_run.get("world_size")
    candidate_world_size = candidate_run.get("world_size")
    if (
        isinstance(reference_world_size, bool)
        or not isinstance(reference_world_size, int)
        or isinstance(candidate_world_size, bool)
        or not isinstance(candidate_world_size, int)
    ):
        raise ValueError("paired run world_size values must be integers")
    reference_output_metadata = load_rank_metadata(
        reference_root / "metadata",
        reference_manifest,
        world_size=reference_world_size,
    )
    candidate_output_metadata = load_rank_metadata(
        candidate_root / "metadata",
        candidate_manifest,
        world_size=candidate_world_size,
    )
    reference_validation = validate_outputs(
        reference_dir,
        reference_manifest,
        metadata=reference_output_metadata,
        expected_count=len(reference_manifest),
        resolution=resolution,
        verify_images=False,
    )
    candidate_validation = validate_outputs(
        candidate_dir,
        candidate_manifest,
        metadata=candidate_output_metadata,
        expected_count=len(candidate_manifest),
        resolution=resolution,
        verify_images=False,
    )
    validate_output_run_identity(reference_validation, reference_run)
    validate_output_run_identity(candidate_validation, candidate_run)
    reference_index = _manifest_index(reference_manifest, "reference")
    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_paths: list[tuple[Path, Path]] = []
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        reference_path = image_path(reference_dir, sample_id)
        candidate_path = image_path(candidate_dir, sample_id)
        if not reference_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(f"missing paired PNG for sample {sample_id}")
        reference = load_rgb_float(reference_path, resolution)
        candidate = load_rgb_float(candidate_path, resolution)
        mse, psnr, ssim = pair_metrics(reference, candidate)
        mse_values.append(mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        if include_lpips:
            lpips_paths.append((reference_path, candidate_path))
        record = reference_index[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "class_id": record.class_id,
                "seed": record.seed,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": None,
                "reference_path": str(reference_path),
                "candidate_path": str(candidate_path),
            }
        )
    lpips_version = None
    lpips_values: list[float] = []
    if include_lpips:
        lpips_values, lpips_version = _lpips_values(
            lpips_paths,
            resolution=resolution,
            device=lpips_device,
            batch_size=lpips_batch_size,
        )
        for row, value in zip(rows, lpips_values, strict=True):
            row["lpips"] = value
    aggregate_mse = float(np.mean(np.asarray(mse_values, dtype=np.float64)))
    exact_count = sum(value == 0.0 for value in mse_values)
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "aggregate_mse": aggregate_mse,
        "psnr_from_aggregate_mse": psnr_from_mse(aggregate_mse),
        "per_image_psnr": _summary(psnr_values),
        "ssim": _summary(ssim_values),
        "exact_pair_count": exact_count,
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
        "ssim_protocol": {
            "package": "scikit-image",
            "version": importlib.metadata.version("scikit-image"),
            "channel_axis": -1,
            "data_range": 1.0,
            "win_size": 7,
        },
        "image_protocol": "saved RGB uint8 PNG converted to float32 [0,1]",
        "reference_manifest_sha256": reference_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "reference_run_identity": {
            field: reference_run[field] for field in PAIRING_FIELDS
        },
        "candidate_run_identity": {
            field: candidate_run[field] for field in PAIRING_FIELDS
        },
        "reference_method": reference_run["method"],
        "candidate_method": candidate_run["method"],
        "candidate_dicache_config_hash": candidate_run["dicache_config_hash"],
        "candidate_dicache_config": {
            field: candidate_run[field] for field in DICACHE_CONFIG_FIELDS
        },
        "reference_run_config_hash": reference_run["config_hash"],
        "candidate_run_config_hash": candidate_run["config_hash"],
        # The rank metadata and latency benchmark bind the canonical input
        # YAML, whereas config_hash above binds the expanded run contract.
        "candidate_input_config_hash": candidate_run["input_config_hash"],
    }
    per_class_ssim: dict[int, list[float]] = defaultdict(list)
    per_class_lpips: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        per_class_ssim[int(row["class_id"])].append(float(row["ssim"]))
        if row["lpips"] is not None:
            per_class_lpips[int(row["class_id"])].append(float(row["lpips"]))
    result["per_class_mean_ssim"] = {
        str(key): float(np.mean(value)) for key, value in sorted(per_class_ssim.items())
    }
    if include_lpips:
        result["lpips"] = {
            **_summary(lpips_values),
            "value_count": len(lpips_values),
            "max": max(lpips_values),
            "package_version": lpips_version,
            "backbone": "alex",
            "spatial": False,
            "reduction": "package default scalar",
            "device": lpips_device,
            "batch_size": lpips_batch_size,
        }
        class_means = {
            key: float(np.mean(value)) for key, value in per_class_lpips.items()
        }
        result["per_class_mean_lpips"] = {
            str(key): value for key, value in sorted(class_means.items())
        }
        result["worst_20_classes_by_lpips"] = [
            {"class_id": key, "mean_lpips": value}
            for key, value in sorted(
                class_means.items(), key=lambda item: item[1], reverse=True
            )[:20]
        ]
        result["worst_100_samples_by_lpips"] = sorted(
            (
                {
                    "sample_id": int(row["sample_id"]),
                    "class_id": int(row["class_id"]),
                    "lpips": float(row["lpips"]),
                }
                for row in rows
            ),
            key=lambda item: item["lpips"],
            reverse=True,
        )[:100]
    return result, rows


def write_rows_csv(path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _default_run_metadata(sample_dir: str) -> Path:
    path = Path(sample_dir)
    candidates = [path / "run_manifest.json", path.parent / "run_manifest.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"strict pairing requires run_manifest.json next to or above {sample_dir}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--reference-run-metadata")
    parser.add_argument("--candidate-run-metadata")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--lpips-batch-size", type=int, default=16)
    parser.add_argument("--expected-count", type=int, default=50000)
    parser.add_argument("--expected-per-class", type=int, default=50)
    parser.add_argument("--expected-num-classes", type=int, default=1000)
    arguments = parser.parse_args()
    reference_metadata = arguments.reference_run_metadata or _default_run_metadata(
        arguments.reference_dir
    )
    candidate_metadata = arguments.candidate_run_metadata or _default_run_metadata(
        arguments.candidate_dir
    )
    reference_manifest = load_manifest(arguments.reference_manifest)
    candidate_manifest = load_manifest(arguments.candidate_manifest)
    validate_manifest(
        reference_manifest,
        expected_count=arguments.expected_count,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
    )
    validate_manifest(
        candidate_manifest,
        expected_count=arguments.expected_count,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
    )
    reference_run = load_json(reference_metadata)
    candidate_run = load_json(candidate_metadata)
    result, rows = evaluate_paired(
        reference_dir=arguments.reference_dir,
        candidate_dir=arguments.candidate_dir,
        reference_manifest=reference_manifest,
        candidate_manifest=candidate_manifest,
        reference_run=reference_run,
        candidate_run=candidate_run,
        reference_manifest_path=arguments.reference_manifest,
        candidate_manifest_path=arguments.candidate_manifest,
        reference_run_metadata_path=reference_metadata,
        candidate_run_metadata_path=candidate_metadata,
        resolution=arguments.resolution,
        include_lpips=not arguments.skip_lpips,
        lpips_device=arguments.lpips_device,
        lpips_batch_size=arguments.lpips_batch_size,
    )
    atomic_write_json(arguments.output_json, result)
    write_rows_csv(arguments.output_csv, rows)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    _main()
