import unittest

from seacache_style.manifest import build_manifest, grouped_records, records_for_shard


class ShardingTest(unittest.TestCase):
    def test_union_intersection_and_fixed_groups(self):
        records = build_manifest(
            samples_per_class=2,
            base_seed=10,
            split_name="toy",
            world_size=4,
            batch_size=7,
            num_classes=10,
        )
        shards = [records_for_shard(records, rank) for rank in range(4)]
        sets = [{record.sample_id for record in shard} for shard in shards]
        self.assertEqual(set.union(*sets), set(range(20)))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(sets[left] & sets[right])
        for rank in range(4):
            groups = grouped_records(records, rank)
            self.assertTrue(all(len(group) <= 7 for group in groups))
            self.assertEqual(sum(map(len, groups)), len(shards[rank]))


if __name__ == "__main__":
    unittest.main()

