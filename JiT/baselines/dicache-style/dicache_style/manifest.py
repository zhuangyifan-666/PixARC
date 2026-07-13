"""Immutable deterministic ImageNet class-to-image manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


COMMON_IMPLEMENTATION_VERSION = "pixarc-dicache-style-v1"
MANIFEST_VERSION = "pixarc-imagenet-c2i-v1"
GENERATOR_ALGORITHM = "per_sample_torch_generator_manual_seed"
NOISE_CONSTRUCTION_PROTOCOL = (
    "for each manifest record create an independent torch.Generator on the "
    "recorded device, manual_seed(record.seed), draw one torch.randn tensor "
    "with the recorded shape/dtype, then stack in fixed batch position order; "
    "model-specific noise_scale is applied after construction"
)


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: int
    class_id: int
    seed: int
    shard_id: int
    position_in_shard: int
    split_name: str
    manifest_version: str
    batch_group_id: str
    position_in_batch: int

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "ManifestRecord":
        return cls(
            sample_id=int(value["sample_id"]),
            class_id=int(value["class_id"]),
            seed=int(value["seed"]),
            shard_id=int(value["shard_id"]),
            position_in_shard=int(value["position_in_shard"]),
            split_name=str(value["split_name"]),
            manifest_version=str(value["manifest_version"]),
            batch_group_id=str(value["batch_group_id"]),
            position_in_batch=int(value["position_in_batch"]),
        )


def build_manifest(
    *,
    samples_per_class: int,
    base_seed: int,
    split_name: str,
    world_size: int = 4,
    batch_size: int,
    num_classes: int = 1000,
) -> list[ManifestRecord]:
    """Build class-major IDs and deterministic modulo shards."""

    if samples_per_class <= 0 or num_classes <= 0:
        raise ValueError("samples_per_class and num_classes must be positive")
    if world_size <= 0 or batch_size <= 0:
        raise ValueError("world_size and batch_size must be positive")
    count = samples_per_class * num_classes
    if base_seed < 0 or base_seed + count >= 2**63:
        raise ValueError("seed range must fit signed 63-bit integers")
    shard_positions = [0] * world_size
    records: list[ManifestRecord] = []
    for sample_id in range(count):
        class_id = sample_id // samples_per_class
        shard_id = sample_id % world_size
        shard_position = shard_positions[shard_id]
        shard_positions[shard_id] += 1
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                class_id=class_id,
                seed=base_seed + sample_id,
                shard_id=shard_id,
                position_in_shard=shard_position,
                split_name=split_name,
                manifest_version=MANIFEST_VERSION,
                batch_group_id=f"{shard_id}:{shard_position // batch_size}",
                position_in_batch=shard_position % batch_size,
            )
        )
    validate_manifest(
        records,
        expected_count=count,
        expected_per_class=samples_per_class,
        expected_num_classes=num_classes,
        world_size=world_size,
        batch_size=batch_size,
        base_seed=base_seed,
    )
    return records


def write_manifest(
    records: Sequence[ManifestRecord],
    path: str | os.PathLike[str],
    *,
    base_seed: int,
    world_size: int,
    batch_size: int,
    generator_device: str,
    noise_dtype: str = "float32",
    noise_shape: Sequence[int] = (3, 256, 256),
) -> dict[str, object]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {destination}")
    sidecar = destination.with_suffix(destination.suffix + ".meta.json")
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite manifest sidecar: {sidecar}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    digest = sha256_file(destination)
    metadata = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_sha256": digest,
        "manifest_records_sha256": manifest_records_sha256(records),
        "record_count": len(records),
        "generator_algorithm": GENERATOR_ALGORITHM,
        "rng_algorithm": (
            "PyTorch Generator implementation selected by the recorded "
            "PyTorch version and generator device"
        ),
        "generator_device": str(generator_device),
        "noise_dtype": str(noise_dtype),
        "noise_shape": [int(value) for value in noise_shape],
        "noise_construction_protocol": NOISE_CONSTRUCTION_PROTOCOL,
        "base_seed": int(base_seed),
        "class_assignment": "class_id=sample_id//samples_per_class",
        "shard_rule": "shard_id=sample_id%world_size",
        "world_size": int(world_size),
        "batch_size": int(batch_size),
        "batch_grouping": (
            "within each modulo shard, contiguous position_in_shard groups of "
            "batch_size with fixed position_in_batch"
        ),
        "pytorch_version": torch.__version__,
    }
    _atomic_json(sidecar, metadata)
    return metadata


def load_manifest(path: str | os.PathLike[str]) -> list[ManifestRecord]:
    source = Path(path)
    records: list[ManifestRecord] = []
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            records = [ManifestRecord.from_mapping(row) for row in csv.DictReader(handle)]
    else:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(ManifestRecord.from_mapping(json.loads(line)))
                except Exception as error:
                    raise ValueError(f"invalid manifest line {line_number}: {error}") from error
    if not records:
        raise ValueError(f"manifest is empty: {source}")
    return records


def validate_manifest_sidecar(
    manifest_path: str | os.PathLike[str],
    records: Sequence[ManifestRecord],
    *,
    world_size: int,
    batch_size: int,
    generator_device: str,
    noise_dtype: str,
    noise_shape: Sequence[int],
) -> dict[str, object]:
    """Bind immutable records to the exact RNG/noise construction protocol."""

    if not records:
        raise ValueError("manifest records must not be empty")
    source = Path(manifest_path)
    sidecar = source.with_suffix(source.suffix + ".meta.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"manifest sidecar is required: {sidecar}")
    with sidecar.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("manifest sidecar must be a JSON object")
    base_seeds = {record.seed - record.sample_id for record in records}
    if len(base_seeds) != 1:
        raise ValueError("manifest does not have one seed=base_seed+sample_id rule")
    expected = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_sha256": sha256_file(source),
        "manifest_records_sha256": manifest_records_sha256(records),
        "record_count": len(records),
        "generator_algorithm": GENERATOR_ALGORITHM,
        "generator_device": str(generator_device),
        "noise_dtype": str(noise_dtype),
        "noise_shape": [int(value) for value in noise_shape],
        "noise_construction_protocol": NOISE_CONSTRUCTION_PROTOCOL,
        "base_seed": next(iter(base_seeds)),
        "world_size": int(world_size),
        "batch_size": int(batch_size),
        "pytorch_version": torch.__version__,
    }
    mismatches = {
        key: {"sidecar": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    required_text = ("rng_algorithm", "class_assignment", "shard_rule", "batch_grouping")
    missing_text = [key for key in required_text if not metadata.get(key)]
    if mismatches or missing_text:
        raise ValueError(
            "manifest sidecar is incompatible with this run: "
            f"mismatches={mismatches}, missing_text={missing_text}"
        )
    return metadata


def validate_manifest(
    records: Sequence[ManifestRecord],
    *,
    expected_count: int | None = None,
    expected_per_class: int | None = None,
    expected_num_classes: int | None = None,
    world_size: int | None = None,
    batch_size: int | None = None,
    base_seed: int | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("manifest must not be empty")
    ids = [record.sample_id for record in records]
    seeds = [record.seed for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id in manifest")
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed in manifest")
    if base_seed is not None and any(
        record.seed != base_seed + record.sample_id for record in records
    ):
        raise ValueError("manifest violates seed=base_seed+sample_id")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, found {len(records)}")
    if set(ids) != set(range(len(records))):
        raise ValueError("sample IDs must cover the contiguous numeric range exactly")
    versions = {record.manifest_version for record in records}
    splits = {record.split_name for record in records}
    if versions != {MANIFEST_VERSION}:
        raise ValueError(f"unexpected manifest versions: {sorted(versions)}")
    if len(splits) != 1:
        raise ValueError(f"manifest mixes splits: {sorted(splits)}")
    class_counts = Counter(record.class_id for record in records)
    if any(class_id < 0 for class_id in class_counts):
        raise ValueError("class IDs must be non-negative")
    if expected_per_class is not None and any(
        value != expected_per_class for value in class_counts.values()
    ):
        raise ValueError("class counts do not match expected_per_class")
    if expected_num_classes is not None and len(class_counts) != expected_num_classes:
        raise ValueError(
            f"expected {expected_num_classes} classes, found {len(class_counts)}"
        )
    if expected_num_classes is not None and set(class_counts) != set(
        range(expected_num_classes)
    ):
        raise ValueError("class IDs must cover the expected contiguous range")
    shard_records: dict[int, list[ManifestRecord]] = defaultdict(list)
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        shard_records[record.shard_id].append(record)
        groups[record.batch_group_id].append(record)
        if world_size is not None and record.shard_id != record.sample_id % world_size:
            raise ValueError(f"sample {record.sample_id} violates modulo sharding")
    if world_size is not None and set(shard_records) != set(range(world_size)):
        raise ValueError("shards do not cover every rank")
    for shard_id, values in shard_records.items():
        ordered = sorted(values, key=lambda value: value.position_in_shard)
        if [value.position_in_shard for value in ordered] != list(range(len(ordered))):
            raise ValueError(f"shard {shard_id} positions are not contiguous")
    if batch_size is not None:
        for record in records:
            expected_group = f"{record.shard_id}:{record.position_in_shard // batch_size}"
            expected_position = record.position_in_shard % batch_size
            if (
                record.batch_group_id != expected_group
                or record.position_in_batch != expected_position
            ):
                raise ValueError(
                    f"sample {record.sample_id} violates fixed batch grouping"
                )
        for group_id, values in groups.items():
            positions = sorted(value.position_in_batch for value in values)
            if positions != list(range(len(positions))) or len(positions) > batch_size:
                raise ValueError(f"invalid fixed batch group {group_id}")
    return {
        "record_count": len(records),
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "shard_counts": {
            key: len(value) for key, value in sorted(shard_records.items())
        },
        "split_name": next(iter(splits)),
    }


def assert_disjoint_seeds(
    first: Sequence[ManifestRecord], second: Sequence[ManifestRecord]
) -> None:
    overlap = {record.seed for record in first} & {record.seed for record in second}
    if overlap:
        raise ValueError(f"manifests share {len(overlap)} sample seeds")


def records_for_shard(
    records: Sequence[ManifestRecord], shard_id: int
) -> list[ManifestRecord]:
    return sorted(
        (record for record in records if record.shard_id == shard_id),
        key=lambda value: value.position_in_shard,
    )


def grouped_records(
    records: Sequence[ManifestRecord], shard_id: int
) -> list[list[ManifestRecord]]:
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records_for_shard(records, shard_id):
        groups[record.batch_group_id].append(record)
    return [
        sorted(values, key=lambda value: value.position_in_batch)
        for _, values in sorted(
            groups.items(), key=lambda item: item[1][0].position_in_shard
        )
    ]


def initial_noise(
    seeds: Sequence[int],
    shape: Sequence[int],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct each sample from its own generator, independent of grouping."""

    if not seeds:
        raise ValueError("at least one seed is required")
    sample_shape = tuple(int(value) for value in shape)
    if any(value <= 0 for value in sample_shape):
        raise ValueError(f"invalid per-sample noise shape: {sample_shape}")
    target = torch.device(device)
    samples = []
    for seed in seeds:
        generator = torch.Generator(device=target)
        generator.manual_seed(int(seed))
        samples.append(
            torch.randn(sample_shape, generator=generator, device=target, dtype=dtype)
        )
    return torch.stack(samples, dim=0)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_records_sha256(records: Sequence[ManifestRecord]) -> str:
    """Hash canonical manifest content independently of JSONL formatting/order."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda value: value.sample_id):
        encoded = json.dumps(
            asdict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", required=True)
    build.add_argument("--samples-per-class", type=int, required=True)
    build.add_argument("--base-seed", type=int, required=True)
    build.add_argument("--split-name", required=True)
    build.add_argument("--world-size", type=int, default=4)
    build.add_argument("--batch-size", type=int, required=True)
    build.add_argument("--generator-device", required=True)
    build.add_argument("--noise-dtype", default="float32")
    build.add_argument("--noise-shape", type=int, nargs="+", default=[3, 256, 256])
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--expected-count", type=int)
    validate.add_argument("--expected-per-class", type=int)
    validate.add_argument("--expected-num-classes", type=int)
    validate.add_argument("--world-size", type=int)
    validate.add_argument("--batch-size", type=int)
    validate.add_argument("--base-seed", type=int)
    validate.add_argument(
        "--disjoint-with",
        action="append",
        default=[],
        help="additional manifest whose seeds must not overlap (repeatable)",
    )
    arguments = parser.parse_args()
    if arguments.command == "build":
        values = build_manifest(
            samples_per_class=arguments.samples_per_class,
            base_seed=arguments.base_seed,
            split_name=arguments.split_name,
            world_size=arguments.world_size,
            batch_size=arguments.batch_size,
        )
        metadata = write_manifest(
            values,
            arguments.output,
            base_seed=arguments.base_seed,
            world_size=arguments.world_size,
            batch_size=arguments.batch_size,
            generator_device=arguments.generator_device,
            noise_dtype=arguments.noise_dtype,
            noise_shape=arguments.noise_shape,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        values = load_manifest(arguments.manifest)
        report = validate_manifest(
            values,
            expected_count=arguments.expected_count,
            expected_per_class=arguments.expected_per_class,
            expected_num_classes=arguments.expected_num_classes,
            world_size=arguments.world_size,
            batch_size=arguments.batch_size,
            base_seed=arguments.base_seed,
        )
        report["manifest_sha256"] = sha256_file(arguments.manifest)
        for other_path in arguments.disjoint_with:
            assert_disjoint_seeds(values, load_manifest(other_path))
        report["disjoint_with"] = list(arguments.disjoint_with)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()

