import torch

from dicache_style.manifest import build_manifest, initial_noise, validate_manifest


def test_final_50k_manifest_and_four_shards():
    records = build_manifest(samples_per_class=50, base_seed=1000, split_name="final",
                             world_size=4, batch_size=1)
    report = validate_manifest(records, expected_count=50_000, expected_per_class=50,
                               expected_num_classes=1000, world_size=4, batch_size=1,
                               base_seed=1000)
    assert report["shard_counts"] == {0: 12500, 1: 12500, 2: 12500, 3: 12500}


def test_per_sample_noise_is_resume_and_order_independent():
    records = build_manifest(samples_per_class=1, base_seed=123, split_name="pilot",
                             world_size=1, batch_size=1, num_classes=3)
    a = initial_noise([row.seed for row in records], (2, 3), dtype=torch.float32, device="cpu")
    b = initial_noise([row.seed for row in reversed(records)], (2, 3), dtype=torch.float32, device="cpu")
    assert torch.equal(a[0], b[-1]) and torch.equal(a[-1], b[0])
