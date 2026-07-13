import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seacache_style.manifest import build_manifest, write_manifest
from seacache_style.pixelgen_io import ManifestNoiseDataset, _unbatch_metadata


class PixelGenIOTest(unittest.TestCase):
    def test_manifest_dataset_is_seeded_and_sharded(self):
        records = build_manifest(
            samples_per_class=2,
            base_seed=500,
            split_name="toy",
            world_size=2,
            batch_size=3,
            num_classes=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_manifest(records, manifest, base_seed=500, world_size=2, batch_size=3)
            dataset = ManifestNoiseDataset(
                manifest_path=str(manifest),
                shard_id=1,
                world_size=2,
                output_root=str(root / "output"),
                batch_size=3,
                config_hash="config",
                checkpoint_path="/checkpoint.ckpt",
                checkpoint_size=123,
                threshold=0.1,
                resolution=4,
            )
            first_noise, first_class, first_metadata = dataset[0]
            repeated_noise, _, _ = dataset[0]
            torch.testing.assert_close(first_noise, repeated_noise, rtol=0, atol=0)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(first_metadata["sample_id"] % 2, 1)
            self.assertEqual(first_metadata["class_id"], first_class)
            full_dataset = ManifestNoiseDataset(
                manifest_path=str(manifest),
                shard_id=0,
                world_size=2,
                output_root=str(root / "full-output"),
                batch_size=3,
                config_hash="full-config",
                checkpoint_path="/checkpoint.ckpt",
                checkpoint_size=123,
                threshold=None,
                resolution=4,
            )
            _noise, _labels, full_metadata = next(iter(DataLoader(full_dataset, batch_size=3)))
            self.assertNotIn("threshold", full_metadata)

    def test_metadata_unbatch_accepts_both_collate_shapes(self):
        mapping = {
            "sample_id": torch.tensor([1, 2]),
            "seed": [11, 12],
        }
        self.assertEqual(
            [row["sample_id"].item() for row in _unbatch_metadata(mapping, 2)],
            [1, 2],
        )
        rows = [{"sample_id": 3}, {"sample_id": 4}]
        self.assertEqual(_unbatch_metadata(rows, 2), rows)


if __name__ == "__main__":
    unittest.main()
