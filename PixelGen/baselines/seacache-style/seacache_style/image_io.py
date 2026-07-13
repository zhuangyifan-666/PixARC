"""Atomic PNG writing, numeric discovery, resume, and output validation."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .manifest import ManifestRecord


COMMON_IMPLEMENTATION_VERSION = "pixarc-seacache-style-v1"


def image_path(sample_dir: str | os.PathLike[str], sample_id: int) -> Path:
    return Path(sample_dir) / f"{int(sample_id):06d}.png"


def atomic_write_png(
    value: np.ndarray | Image.Image,
    destination: str | os.PathLike[str],
    *,
    resolution: int = 256,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite image: {path}")
    image = value if isinstance(value, Image.Image) else Image.fromarray(value)
    if image.mode != "RGB" or image.size != (resolution, resolution):
        raise ValueError(
            f"expected RGB {resolution}x{resolution}, got {image.mode} {image.size}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        image.save(temporary_name, format="PNG")
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def numeric_image_ids(sample_dir: str | os.PathLike[str]) -> list[int]:
    result = []
    for path in Path(sample_dir).glob("*.png"):
        if not path.stem.isdecimal():
            raise ValueError(f"non-numeric PNG name: {path.name}")
        result.append(int(path.stem))
    if len(result) != len(set(result)):
        raise ValueError("duplicate numeric sample ID (possibly alternate zero padding)")
    return sorted(result)


def load_metadata_jsonl(path: str | os.PathLike[str]) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            sample_id = int(value["sample_id"])
            if sample_id in records:
                raise ValueError(f"duplicate metadata sample ID {sample_id}")
            records[sample_id] = value
    return records


def load_rank_metadata(
    metadata_dir: str | os.PathLike[str],
    manifest: Sequence[ManifestRecord],
    *,
    world_size: int,
) -> dict[int, dict[str, object]]:
    """Load exactly one metadata JSONL per rank and bind rows to manifest shards."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    directory = Path(metadata_dir)
    # Accept no alternate files that could hide duplicated or stale rows.
    expected_paths = {directory / f"rank_{rank}.jsonl" for rank in range(world_size)}
    actual_paths = set(directory.glob("rank_*.jsonl")) if directory.is_dir() else set()
    missing = sorted(path.name for path in expected_paths - actual_paths)
    extra = sorted(path.name for path in actual_paths - expected_paths)
    if missing or extra:
        raise ValueError(f"rank metadata files differ: missing={missing}, extra={extra}")

    by_id = {record.sample_id: record for record in manifest}
    combined: dict[int, dict[str, object]] = {}
    for rank in range(world_size):
        path = directory / f"rank_{rank}.jsonl"
        rows = load_metadata_jsonl(path)
        overlap = set(combined) & set(rows)
        if overlap:
            raise ValueError(
                f"duplicate metadata sample IDs across ranks: {sorted(overlap)[:5]}"
            )
        for sample_id, row in rows.items():
            record = by_id.get(sample_id)
            if record is None:
                raise ValueError(f"metadata contains unknown sample ID {sample_id}")
            if record.shard_id != rank:
                raise ValueError(
                    f"metadata sample {sample_id} is in rank {rank}, expected {record.shard_id}"
                )
            if row.get("status") != "ok":
                raise ValueError(f"metadata sample {sample_id} is not marked ok")
        combined.update(rows)
    return combined


def verify_png(path: str | os.PathLike[str], *, resolution: int = 256) -> None:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (resolution, resolution):
            raise ValueError(f"invalid image {path}: {image.mode} {image.size}")
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.shape != (resolution, resolution, 3):
            raise ValueError(f"invalid array representation for {path}")


def validate_outputs(
    sample_dir: str | os.PathLike[str],
    manifest: Sequence[ManifestRecord],
    *,
    metadata: Mapping[int, Mapping[str, object]] | None = None,
    expected_count: int | None = None,
    expected_per_class: int | None = None,
    expected_num_classes: int | None = None,
    resolution: int = 256,
    verify_images: bool = True,
) -> dict[str, object]:
    ids = numeric_image_ids(sample_dir)
    expected = {record.sample_id for record in manifest}
    actual = set(ids)
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"expected {expected_count} PNGs, found {len(ids)}")
    if actual != expected:
        raise ValueError(
            f"sample IDs differ: missing={len(expected-actual)}, extra={len(actual-expected)}"
        )
    by_id = {record.sample_id: record for record in manifest}
    class_counts = Counter()
    for sample_id in ids:
        if verify_images:
            verify_png(image_path(sample_dir, sample_id), resolution=resolution)
        class_counts[by_id[sample_id].class_id] += 1
        if metadata is not None:
            if sample_id not in metadata:
                raise ValueError(f"missing metadata for sample {sample_id}")
            row = metadata[sample_id]
            expected_record = by_id[sample_id]
            integer_checks = {
                "class_id": expected_record.class_id,
                "seed": expected_record.seed,
                "position_in_batch": expected_record.position_in_batch,
            }
            for key, value in integer_checks.items():
                if key not in row:
                    raise ValueError(f"metadata is missing {key} for sample {sample_id}")
                if int(row[key]) != value:
                    raise ValueError(f"metadata {key} mismatch for sample {sample_id}")
            if row.get("batch_group_id") != expected_record.batch_group_id:
                raise ValueError(
                    f"metadata batch_group_id mismatch for sample {sample_id}"
                )
            if row.get("status") != "ok":
                raise ValueError(f"metadata status mismatch for sample {sample_id}")
    if expected_num_classes is not None and len(class_counts) != expected_num_classes:
        raise ValueError(
            f"expected {expected_num_classes} classes, found {len(class_counts)}"
        )
    if expected_per_class is not None and any(
        count != expected_per_class for count in class_counts.values()
    ):
        raise ValueError("class counts do not match expected_per_class")
    if metadata is not None and set(metadata) != expected:
        raise ValueError("metadata and image sample IDs are not one-to-one")
    result = {
        "sample_count": len(ids),
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "resolution": resolution,
        "mode": "RGB",
        "dtype": "uint8",
    }
    if metadata is not None:
        for field in (
            "config_hash",
            "checkpoint_path",
            "checkpoint_size",
            "manifest_sha256",
            "threshold",
        ):
            values = {row.get(field) for row in metadata.values()}
            if len(values) != 1:
                raise ValueError(f"metadata field {field!r} is not consistent")
            result[field] = next(iter(values))
    return result


def validate_output_run_identity(
    validation: Mapping[str, object], run_metadata: Mapping[str, object]
) -> None:
    """Bind validated per-image metadata to one run-manifest identity."""

    required = (
        "input_config_hash",
        "checkpoint_path",
        "checkpoint_size",
        "manifest_sha256",
        "threshold",
        "resolution",
    )
    missing = [field for field in required if field not in run_metadata]
    if missing:
        raise ValueError(f"run metadata is missing output identity fields: {missing}")
    expected = {
        "config_hash": run_metadata["input_config_hash"],
        "checkpoint_path": run_metadata["checkpoint_path"],
        "checkpoint_size": run_metadata["checkpoint_size"],
        "manifest_sha256": run_metadata["manifest_sha256"],
        "threshold": run_metadata["threshold"],
        "resolution": run_metadata["resolution"],
    }
    errors = [
        f"{field}: output={validation.get(field)!r}, run={value!r}"
        for field, value in expected.items()
        if validation.get(field) != value
    ]
    if errors:
        raise ValueError(
            "validated output metadata is not bound to the run manifest:\n- "
            + "\n- ".join(errors)
        )


def resumable_batch_groups(
    manifest: Sequence[ManifestRecord],
    shard_id: int,
    sample_dir: str | os.PathLike[str],
    metadata: Mapping[int, Mapping[str, object]],
    *,
    manifest_sha256: str,
    config_hash: str,
    checkpoint_path: str,
    checkpoint_size: int,
    threshold: float | None,
    resolution: int = 256,
) -> tuple[list[list[ManifestRecord]], list[str]]:
    """Skip only complete fixed groups; partial groups fail to preserve gating."""

    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in manifest:
        if record.shard_id == shard_id:
            groups[record.batch_group_id].append(record)
    pending: list[list[ManifestRecord]] = []
    skipped: list[str] = []
    for group_id, values in sorted(
        groups.items(), key=lambda item: min(v.position_in_shard for v in item[1])
    ):
        values.sort(key=lambda value: value.position_in_batch)
        existing = [image_path(sample_dir, value.sample_id).exists() for value in values]
        if any(existing) and not all(existing):
            raise RuntimeError(f"partial batch group {group_id}; refusing regrouped resume")
        if not any(existing):
            pending.append(values)
            continue
        for value in values:
            verify_png(image_path(sample_dir, value.sample_id), resolution=resolution)
            row = metadata.get(value.sample_id)
            if row is None:
                raise RuntimeError(f"missing resume metadata for sample {value.sample_id}")
            checks = {
                "class_id": value.class_id,
                "seed": value.seed,
                "batch_group_id": value.batch_group_id,
                "position_in_batch": value.position_in_batch,
                "manifest_sha256": manifest_sha256,
                "config_hash": config_hash,
                "checkpoint_path": checkpoint_path,
                "checkpoint_size": checkpoint_size,
                "threshold": threshold,
            }
            for key, expected in checks.items():
                if row.get(key) != expected:
                    raise RuntimeError(
                        f"resume metadata mismatch for sample {value.sample_id}: {key}"
                    )
        skipped.append(group_id)
    return pending, skipped


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
