import pytest

from taylorseer_style.manifest import (
    assert_disjoint_seeds,
    build_manifest,
    manifest_records_sha256,
    validate_manifest,
    validate_manifest_sidecar,
    write_manifest,
)


def test_final_50k_and_independent_8k_manifests():
    final = build_manifest(samples_per_class=50, base_seed=1_000_000, split_name="final", world_size=4, batch_size=32)
    validation = build_manifest(samples_per_class=8, base_seed=2_000_000, split_name="validation", world_size=4, batch_size=32)
    report = validate_manifest(final, expected_count=50_000, expected_per_class=50, expected_num_classes=1000, world_size=4, batch_size=32)
    assert report["shard_counts"] == {0: 12500, 1: 12500, 2: 12500, 3: 12500}
    assert_disjoint_seeds(final, validation)
    assert manifest_records_sha256(final) == manifest_records_sha256(list(reversed(final)))


def test_sidecar_binds_rng_device_shape_and_version(tmp_path):
    records = build_manifest(
        samples_per_class=2,
        num_classes=2,
        base_seed=10,
        split_name="toy",
        world_size=2,
        batch_size=2,
    )
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        records,
        path,
        base_seed=10,
        world_size=2,
        batch_size=2,
        generator_device="cpu",
        noise_shape=(3, 8, 8),
    )
    validate_manifest_sidecar(
        path,
        records,
        world_size=2,
        batch_size=2,
        generator_device="cpu",
        noise_dtype="float32",
        noise_shape=(3, 8, 8),
    )
    with pytest.raises(ValueError, match="generator_device"):
        validate_manifest_sidecar(
            path,
            records,
            world_size=2,
            batch_size=2,
            generator_device="cuda",
            noise_dtype="float32",
            noise_shape=(3, 8, 8),
        )
