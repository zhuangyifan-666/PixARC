"""Atomic PNG writing, numeric discovery, resume, and output validation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .manifest import ManifestRecord
from .metadata import DICACHE_CONFIG_FIELDS


COMMON_IMPLEMENTATION_VERSION = "pixarc-dicache-style-v1"


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


def _finite_trajectory_number(
    row: Mapping[str, object], key: str, *, integer: bool = False
) -> int | float:
    if key not in row:
        raise ValueError(f"metadata is missing {key}")
    value = row[key]
    if isinstance(value, bool):
        raise ValueError(f"metadata {key} must be numeric, not boolean")
    try:
        number = float(value)  # accepts JSON numbers and parseable legacy scalars
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata {key} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"metadata {key} is not finite")
    if integer:
        if not number.is_integer():
            raise ValueError(f"metadata {key} must be an integer")
        return int(number)
    return number


def _validate_finite_diagnostic_tree(value: object, *, path: str) -> None:
    """Reject NaN/Inf anywhere in an opt-in durable trajectory trace."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_diagnostic_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_diagnostic_tree(item, path=f"{path}[{index}]")
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        if isinstance(value, str) and value.strip().lower() in {
            "nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity",
            "+infinity", "-infinity",
        }:
            raise ValueError(f"metadata diagnostic {path} is non-finite")
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"metadata diagnostic {path} is non-finite")
        return
    raise ValueError(
        f"metadata diagnostic {path} has unsupported type {type(value).__name__}"
    )


def _validate_trajectory_metadata(
    row: Mapping[str, object],
    *,
    sample_id: int,
    expected_group_ids: Sequence[int],
) -> str:
    trajectory_id = row.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"metadata sample {sample_id} has no trajectory_id")
    raw_ids = row.get("trajectory_sample_ids")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError(
            f"metadata sample {sample_id} has invalid trajectory_sample_ids"
        )
    sample_ids: list[int] = []
    for value in raw_ids:
        if isinstance(value, bool):
            raise ValueError("trajectory_sample_ids cannot contain booleans")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("trajectory_sample_ids must be parseable integers") from error
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("trajectory_sample_ids must be exact integers")
        sample_ids.append(normalized)
    if sample_ids != list(expected_group_ids):
        raise ValueError(
            f"trajectory sample IDs mismatch for sample {sample_id}: "
            f"{sample_ids} != {list(expected_group_ids)}"
        )
    if sample_id not in sample_ids:
        raise ValueError(f"trajectory does not contain metadata sample {sample_id}")
    if row.get("trajectory_call_count_valid") is not True:
        raise ValueError(f"trajectory call_count_valid is not true for sample {sample_id}")

    total_nfe = _finite_trajectory_number(
        row, "trajectory_total_nfe", integer=True
    )
    total_calls = _finite_trajectory_number(
        row, "trajectory_total_stream_calls", integer=True
    )
    direct = _finite_trajectory_number(
        row, "trajectory_direct_full_count", integer=True
    )
    resumed = _finite_trajectory_number(
        row, "trajectory_resumed_full_count", integer=True
    )
    reused = _finite_trajectory_number(
        row, "trajectory_reuse_count", integer=True
    )
    observed_forwards = _finite_trajectory_number(
        row, "trajectory_network_forward_count", integer=True
    )
    expected_forwards = _finite_trajectory_number(
        row, "trajectory_expected_network_forward_count", integer=True
    )
    if total_nfe <= 0 or total_calls <= 0:
        raise ValueError("trajectory NFE and stream-call counts must be positive")
    if min(direct, resumed, reused, observed_forwards, expected_forwards) < 0:
        raise ValueError("trajectory counters cannot be negative")
    if direct + resumed + reused != total_calls:
        raise ValueError("trajectory actions do not sum to total_stream_calls")
    if observed_forwards != expected_forwards:
        raise ValueError("observed and expected network-forward counts differ")
    if observed_forwards != total_calls:
        raise ValueError("network-forward and stream-call counts differ")
    if total_calls % total_nfe:
        raise ValueError("total_stream_calls is not an integer multiple of total_nfe")

    protocol_batch_size = _finite_trajectory_number(
        row, "real_batch_size", integer=True
    )
    if protocol_batch_size <= 0:
        raise ValueError("real_batch_size must be positive")
    protocol_effective_batch_size = _finite_trajectory_number(
        row, "effective_cfg_batch_size", integer=True
    )
    if protocol_effective_batch_size != 2 * protocol_batch_size:
        raise ValueError("effective_cfg_batch_size must equal 2 * real_batch_size")
    trajectory_batch_size = (
        _finite_trajectory_number(row, "trajectory_real_batch_size", integer=True)
        if "trajectory_real_batch_size" in row
        else protocol_batch_size
    )
    if trajectory_batch_size != len(expected_group_ids):
        raise ValueError("trajectory_real_batch_size differs from trajectory sample IDs")
    if trajectory_batch_size > protocol_batch_size:
        raise ValueError("trajectory_real_batch_size exceeds protocol real_batch_size")
    trajectory_effective_batch_size = (
        _finite_trajectory_number(
            row, "trajectory_effective_cfg_batch_size", integer=True
        )
        if "trajectory_effective_cfg_batch_size" in row
        else protocol_effective_batch_size
    )
    if trajectory_effective_batch_size != 2 * trajectory_batch_size:
        raise ValueError(
            "trajectory_effective_cfg_batch_size must equal 2 * "
            "trajectory_real_batch_size"
        )
    for key, value in row.items():
        if key.startswith("trajectory_"):
            _validate_finite_diagnostic_tree(value, path=key)
    return trajectory_id


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
    group_ids: dict[str, list[int]] = defaultdict(list)
    for record in sorted(manifest, key=lambda item: item.position_in_batch):
        group_ids[record.batch_group_id].append(record.sample_id)
    trajectory_groups: dict[str, str] = {}
    group_trajectories: dict[str, str] = {}
    group_trajectory_payloads: dict[str, str] = {}
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
            trajectory_id = _validate_trajectory_metadata(
                row,
                sample_id=sample_id,
                expected_group_ids=group_ids[expected_record.batch_group_id],
            )
            prior_group = trajectory_groups.setdefault(
                trajectory_id, expected_record.batch_group_id
            )
            if prior_group != expected_record.batch_group_id:
                raise ValueError("one trajectory_id is reused across batch groups")
            prior_trajectory = group_trajectories.setdefault(
                expected_record.batch_group_id, trajectory_id
            )
            if prior_trajectory != trajectory_id:
                raise ValueError("one batch group contains multiple trajectory IDs")
            trajectory_payload = json.dumps(
                {
                    "trajectory_id": trajectory_id,
                    **{
                        key: value
                        for key, value in row.items()
                        if key.startswith("trajectory_")
                    },
                },
                sort_keys=True,
                allow_nan=False,
            )
            prior_payload = group_trajectory_payloads.setdefault(
                expected_record.batch_group_id, trajectory_payload
            )
            if prior_payload != trajectory_payload:
                raise ValueError("trajectory metadata differs within one batch group")
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
            "dicache_config_hash",
            "release_gate_sha256",
            "checkpoint_path",
            "checkpoint_size",
            "manifest_sha256",
            "method",
            "real_batch_size",
            "effective_cfg_batch_size",
            *DICACHE_CONFIG_FIELDS,
        ):
            if field == "release_gate_sha256" and any(
                field not in row for row in metadata.values()
            ):
                raise ValueError("metadata is missing release_gate_sha256")
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
        "dicache_config_hash",
        "release_gate_sha256",
        "checkpoint_path",
        "checkpoint_size",
        "manifest_sha256",
        "method",
        "real_batch_size",
        "effective_cfg_batch_size",
        *DICACHE_CONFIG_FIELDS,
        "resolution",
    )
    missing = [field for field in required if field not in run_metadata]
    if missing:
        raise ValueError(f"run metadata is missing output identity fields: {missing}")
    expected = {
        "config_hash": run_metadata["input_config_hash"],
        "dicache_config_hash": run_metadata["dicache_config_hash"],
        "release_gate_sha256": run_metadata["release_gate_sha256"],
        "checkpoint_path": run_metadata["checkpoint_path"],
        "checkpoint_size": run_metadata["checkpoint_size"],
        "manifest_sha256": run_metadata["manifest_sha256"],
        "method": run_metadata["method"],
        "real_batch_size": run_metadata["real_batch_size"],
        "effective_cfg_batch_size": run_metadata["effective_cfg_batch_size"],
        **{field: run_metadata[field] for field in DICACHE_CONFIG_FIELDS},
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
    dicache_config_hash: str,
    release_gate_sha256: str,
    dicache_config: Mapping[str, object] | None = None,
    checkpoint_path: str,
    checkpoint_size: int,
    method: str,
    protocol_batch_size: int,
    resolution: int = 256,
) -> tuple[list[list[ManifestRecord]], list[str]]:
    """Skip only complete fixed groups; partial groups fail to preserve gating."""

    if isinstance(protocol_batch_size, bool) or protocol_batch_size <= 0:
        raise ValueError("protocol_batch_size must be a positive integer")

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
            checks: dict[str, object] = {
                "class_id": value.class_id,
                "seed": value.seed,
                "batch_group_id": value.batch_group_id,
                "position_in_batch": value.position_in_batch,
                "manifest_sha256": manifest_sha256,
                "config_hash": config_hash,
                "dicache_config_hash": dicache_config_hash,
                "release_gate_sha256": release_gate_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_size": checkpoint_size,
                "method": method,
                "real_batch_size": protocol_batch_size,
                "effective_cfg_batch_size": 2 * protocol_batch_size,
            }
            trajectory_real = row.get("trajectory_real_batch_size")
            trajectory_effective = row.get("trajectory_effective_cfg_batch_size")
            if len(values) != protocol_batch_size and (
                trajectory_real is None or trajectory_effective is None
            ):
                raise RuntimeError(
                    f"resume metadata mismatch for sample {value.sample_id}: "
                    "partial trajectory batch fields are required"
                )
            if trajectory_real is not None:
                checks["trajectory_real_batch_size"] = len(values)
            if trajectory_effective is not None:
                checks["trajectory_effective_cfg_batch_size"] = 2 * len(values)
            if dicache_config is not None:
                missing = [field for field in DICACHE_CONFIG_FIELDS if field not in dicache_config]
                if missing:
                    raise ValueError(f"resume DiCache config is missing fields: {missing}")
                checks.update(
                    {field: dicache_config[field] for field in DICACHE_CONFIG_FIELDS}
                )
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
