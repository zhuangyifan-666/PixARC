import torch

from speca_style.manifest import build_manifest, grouped_records, initial_noise, records_for_shard


def test_shards_groups_and_noise_are_order_independent():
    records = build_manifest(samples_per_class=2, base_seed=500, split_name="toy", world_size=4, batch_size=3, num_classes=8)
    shards = [records_for_shard(records, rank) for rank in range(4)]
    assert set.intersection(*(set(value.sample_id for value in shard) for shard in shards)) == set()
    assert set().union(*(set(value.sample_id for value in shard) for shard in shards)) == set(range(16))
    assert all(len(group) <= 3 for rank in range(4) for group in grouped_records(records, rank))
    first = initial_noise([500, 503], (2, 2), device="cpu", dtype=torch.float32)
    second = initial_noise([503, 500], (2, 2), device="cpu", dtype=torch.float32)
    torch.testing.assert_close(first[0], second[1], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[0], rtol=0, atol=0)

