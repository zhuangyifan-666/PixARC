"""Validated bridge to one explicit local ADM evaluation suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

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
    atomic_write_json,
    load_json,
    validate_full_seacache_roles,
    validate_run_artifacts,
)


COMMON_IMPLEMENTATION_VERSION = "pixarc-seacache-style-v1"
METRIC_PATTERNS = {
    "fid": re.compile(r"\bFID\b\s*[:=]\s*([-+0-9.eE]+)", re.I),
    "sfid": re.compile(r"\bsFID\b\s*[:=]\s*([-+0-9.eE]+)", re.I),
    "inception_score": re.compile(
        r"\b(?:Inception Score|IS)\b\s*[:=]\s*([-+0-9.eE]+)", re.I
    ),
    "precision": re.compile(r"\bprecision\b\s*[:=]\s*([-+0-9.eE]+)", re.I),
    "recall": re.compile(r"\brecall\b\s*[:=]\s*([-+0-9.eE]+)", re.I),
}


def build_adm_sample_npz(
    *,
    sample_dir: str | os.PathLike[str],
    manifest: Sequence[ManifestRecord],
    output_npz: str | os.PathLike[str],
    resolution: int,
) -> None:
    destination = Path(output_npz)
    if destination.suffix != ".npz":
        raise ValueError("sample NPZ path must end in the exact suffix '.npz'")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite sample NPZ: {destination}")
    bytes_required = len(manifest) * resolution * resolution * 3
    free = shutil.disk_usage(destination.parent).free
    # Temporary NPY plus final uncompressed NPZ can coexist.
    if free < int(bytes_required * 2.2):
        raise OSError(
            f"insufficient free space: need about {int(bytes_required*2.2)} bytes, have {free}"
        )
    ids = sorted(record.sample_id for record in manifest)
    with tempfile.TemporaryDirectory(prefix="adm-npz-", dir=destination.parent) as temp:
        mmap_path = Path(temp) / "samples.npy"
        array = np.lib.format.open_memmap(
            mmap_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(ids), resolution, resolution, 3),
        )
        for index, sample_id in enumerate(ids):
            with Image.open(image_path(sample_dir, sample_id)) as image:
                image.load()
                if image.mode != "RGB" or image.size != (resolution, resolution):
                    raise ValueError(f"invalid sample {sample_id}: {image.mode} {image.size}")
                value = np.asarray(image)
                if value.dtype != np.uint8:
                    raise ValueError(f"sample {sample_id} is not uint8")
                array[index] = value
        array.flush()
        np.savez(destination, arr_0=array)


def run_adm_evaluator(
    *,
    evaluator: str | os.PathLike[str],
    reference_npz: str | os.PathLike[str],
    sample_npz: str | os.PathLike[str],
) -> tuple[dict[str, float], str]:
    evaluator_path = Path(evaluator).resolve(strict=True)
    reference_path = Path(reference_npz).resolve(strict=True)
    sample_path = Path(sample_npz).resolve(strict=True)
    process = subprocess.run(
        [sys.executable, str(evaluator_path), str(reference_path), str(sample_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    metrics: dict[str, float] = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(process.stdout)
        if match:
            metrics[name] = float(match.group(1))
    missing = set(METRIC_PATTERNS) - set(metrics)
    if missing:
        raise RuntimeError(
            f"ADM evaluator output did not expose {sorted(missing)}; raw output retained by caller"
        )
    return metrics, process.stdout


def distribution_deltas(
    full: Mapping[str, Any], seacache: Mapping[str, Any]
) -> dict[str, float]:
    """Compute candidate-minus-Full deltas without changing metric semantics."""

    validate_full_seacache_roles(full, seacache)
    names = ("fid", "sfid", "inception_score", "precision", "recall")
    missing = [name for name in names if name not in full or name not in seacache]
    if missing:
        raise ValueError(f"distribution result is missing metrics: {missing}")
    context_fields = (
        "sample_count",
        "resolution",
        "reference_npz",
        "reference_npz_size",
        "reference_npz_sha256",
        "manifest_sha256",
        "manifest_records_sha256",
        "evaluator_commit_or_version",
        "evaluator_path",
        "evaluator_sha256",
        "image_conversion_protocol",
        "run_identity",
    )
    context_errors = []
    for field in context_fields:
        if field not in full or field not in seacache:
            context_errors.append(f"missing comparison field {field!r}")
        elif full[field] != seacache[field]:
            context_errors.append(
                f"{field}: Full={full[field]!r}, SeaCache={seacache[field]!r}"
            )
    if context_errors:
        raise ValueError(
            "distribution runs are not comparable:\n- " + "\n- ".join(context_errors)
        )
    return {f"delta_{name}": float(seacache[name]) - float(full[name]) for name in names}


def evaluator_identity(path: Path) -> str:
    try:
        process = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return f"git:{process.stdout.strip()}"
    except (OSError, subprocess.CalledProcessError):
        return f"sha256:{sha256_file(path)}"


def _default_run_metadata(sample_dir: str) -> Path:
    path = Path(sample_dir)
    for candidate in (path / "run_manifest.json", path.parent / "run_manifest.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"distribution evaluation requires run_manifest.json next to or above {sample_dir}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-npz", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--run-metadata")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sample-npz")
    parser.add_argument("--expected-count", type=int, default=50000)
    parser.add_argument("--expected-per-class", type=int, default=50)
    parser.add_argument("--expected-num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=256)
    arguments = parser.parse_args()
    reference = Path(arguments.reference_npz)
    evaluator = Path(arguments.evaluator)
    if not reference.is_file():
        raise FileNotFoundError(
            f"ImageNet ADM reference NPZ is required locally: {reference}; no download is attempted"
        )
    if not evaluator.is_file():
        raise FileNotFoundError(
            f"one explicit local ADM evaluator.py is required: {evaluator}; no fallback is used"
        )
    records = load_manifest(arguments.manifest)
    run_metadata_path = arguments.run_metadata or _default_run_metadata(
        arguments.sample_dir
    )
    run_metadata = load_json(run_metadata_path)
    manifest_digest = sha256_file(arguments.manifest)
    if run_metadata.get("manifest_sha256") != manifest_digest:
        raise ValueError("run metadata is not bound to the supplied manifest")
    canonical_manifest_digest = manifest_records_sha256(records)
    if run_metadata.get("manifest_records_sha256") != canonical_manifest_digest:
        raise ValueError("run metadata is not bound to canonical manifest records")
    missing_identity = [field for field in PAIRING_FIELDS if field not in run_metadata]
    if missing_identity:
        raise ValueError(
            f"run metadata is missing distribution identity fields: {missing_identity}"
        )
    run_identity = {field: run_metadata[field] for field in PAIRING_FIELDS}
    validate_manifest(
        records,
        expected_count=arguments.expected_count,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
    )
    run_root = validate_run_artifacts(
        run_metadata_path=run_metadata_path,
        sample_dir=arguments.sample_dir,
        supplied_manifest_path=arguments.manifest,
        run_metadata=run_metadata,
        manifest_records_sha256=canonical_manifest_digest,
    )
    world_size = run_metadata.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise ValueError("run metadata world_size must be an integer")
    output_metadata = load_rank_metadata(
        run_root / "metadata", records, world_size=world_size
    )
    validation = validate_outputs(
        arguments.sample_dir,
        records,
        metadata=output_metadata,
        expected_count=arguments.expected_count,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
        resolution=arguments.resolution,
    )
    validate_output_run_identity(validation, run_metadata)
    sample_npz = Path(
        arguments.sample_npz
        or (Path(arguments.output_json).with_suffix(".samples.npz"))
    )
    build_adm_sample_npz(
        sample_dir=arguments.sample_dir,
        manifest=records,
        output_npz=sample_npz,
        resolution=arguments.resolution,
    )
    atomic_write_json(
        str(sample_npz) + ".meta.json",
        {
            "manifest_sha256": manifest_digest,
            "manifest_records_sha256": canonical_manifest_digest,
            "order": "ascending numeric sample_id",
            "sample_count": len(records),
            "first_sample_id": min(record.sample_id for record in records),
            "last_sample_id": max(record.sample_id for record in records),
        },
    )
    reference_resolved = reference.resolve(strict=True)
    reference_size = reference_resolved.stat().st_size
    reference_digest = sha256_file(reference_resolved)
    evaluator_resolved = evaluator.resolve(strict=True)
    evaluator_version = evaluator_identity(evaluator_resolved)
    evaluator_digest = sha256_file(evaluator_resolved)
    metrics, raw_output = run_adm_evaluator(
        evaluator=evaluator_resolved,
        reference_npz=reference_resolved,
        sample_npz=sample_npz,
    )
    if (
        reference_resolved.stat().st_size != reference_size
        or sha256_file(reference_resolved) != reference_digest
    ):
        raise RuntimeError("reference NPZ changed while the evaluator was running")
    if sha256_file(evaluator_resolved) != evaluator_digest:
        raise RuntimeError("evaluator script changed while it was running")
    result: dict[str, Any] = {
        **metrics,
        "sample_count": validation["sample_count"],
        "resolution": arguments.resolution,
        "reference_npz": str(reference_resolved),
        "reference_npz_size": reference_size,
        "reference_npz_sha256": reference_digest,
        "manifest_sha256": manifest_digest,
        "manifest_records_sha256": canonical_manifest_digest,
        "run_identity": run_identity,
        "run_config_hash": run_metadata["config_hash"],
        "input_config_hash": run_metadata["input_config_hash"],
        "method": run_metadata["method"],
        "threshold": run_metadata["threshold"],
        "run_metadata": str(Path(run_metadata_path).resolve()),
        "evaluator_commit_or_version": evaluator_version,
        "evaluator_path": str(evaluator_resolved),
        "evaluator_sha256": evaluator_digest,
        "image_conversion_protocol": (
            "numeric sample_id order; RGB uint8 "
            f"{arguments.resolution}x{arguments.resolution}; uncompressed arr_0 NPZ"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_evaluator_output": raw_output,
    }
    atomic_write_json(arguments.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
