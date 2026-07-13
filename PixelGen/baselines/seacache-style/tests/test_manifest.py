import tempfile
import unittest
from pathlib import Path

import torch

from seacache_style.manifest import (
    assert_disjoint_seeds,
    build_manifest,
    initial_noise,
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
    write_manifest,
)


class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.final = build_manifest(
            samples_per_class=50,
            base_seed=100_000,
            split_name="final50k",
            world_size=4,
            batch_size=32,
        )
        cls.validation = build_manifest(
            samples_per_class=8,
            base_seed=1_000_000,
            split_name="validation8k",
            world_size=4,
            batch_size=32,
        )

    def test_50k_balance_and_disjoint_validation(self):
        report = validate_manifest(
            self.final,
            expected_count=50_000,
            expected_per_class=50,
            world_size=4,
            batch_size=32,
        )
        self.assertEqual(report["shard_counts"], {0: 12500, 1: 12500, 2: 12500, 3: 12500})
        assert_disjoint_seeds(self.final, self.validation)

    def test_atomic_roundtrip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            metadata = write_manifest(
                self.validation, path, base_seed=1_000_000, world_size=4, batch_size=32
            )
            self.assertEqual(load_manifest(path), self.validation)
            self.assertEqual(metadata["manifest_sha256"], sha256_file(path))
            self.assertEqual(
                metadata["manifest_records_sha256"],
                manifest_records_sha256(list(reversed(self.validation))),
            )
            self.assertTrue(path.with_suffix(".jsonl.meta.json").is_file())

    def test_noise_is_independent_of_batch_grouping_and_resume(self):
        seeds = [record.seed for record in self.validation[:4]]
        all_noise = initial_noise(seeds, (2, 3), device="cpu", dtype=torch.float32)
        regrouped = torch.cat(
            [
                initial_noise(seeds[:1], (2, 3), device="cpu", dtype=torch.float32),
                initial_noise(seeds[1:], (2, 3), device="cpu", dtype=torch.float32),
            ]
        )
        torch.testing.assert_close(all_noise, regrouped, rtol=0, atol=0)

    def test_runtime_batch_must_match_frozen_grouping(self):
        with self.assertRaisesRegex(ValueError, "fixed batch grouping"):
            validate_manifest(self.final, world_size=4, batch_size=16)


if __name__ == "__main__":
    unittest.main()
