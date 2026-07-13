from dicache_style.manifest import build_manifest


def test_validation_and_final_seeds_can_be_disjoint():
    validation = build_manifest(samples_per_class=8, base_seed=0, split_name="validation",
                                world_size=4, batch_size=1)
    final = build_manifest(samples_per_class=50, base_seed=1_000_000, split_name="final",
                           world_size=4, batch_size=1)
    assert not ({row.seed for row in validation} & {row.seed for row in final})
