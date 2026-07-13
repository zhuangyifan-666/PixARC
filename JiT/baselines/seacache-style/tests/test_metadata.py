import tempfile
import unittest
import json
from pathlib import Path

import yaml

from seacache_style.image_io import load_rank_metadata
from seacache_style.manifest import (
    build_manifest,
    manifest_records_sha256,
    write_manifest,
)
from seacache_style.metadata import (
    PAIRING_FIELDS,
    atomic_create_json,
    atomic_write_json,
    canonical_hash,
    source_tree_sha256,
    validate_paired_runs,
    validate_run_artifacts,
)
from seacache_style.distribution_metrics import build_adm_sample_npz, distribution_deltas


class MetadataTest(unittest.TestCase):
    def setUp(self):
        self.run = {field: f"value-{field}" for field in PAIRING_FIELDS}

    def test_canonical_hash_order_independent(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_source_tree_hash_binds_untracked_code(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "seacache_style"
            package.mkdir()
            source = package / "example.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            before = source_tree_sha256(directory)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(before, source_tree_sha256(directory))

    def test_sample_npz_requires_exact_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_adm_sample_npz(
                    sample_dir=directory,
                    manifest=[],
                    output_npz=Path(directory) / "samples",
                    resolution=8,
                )

    def test_archived_run_and_rank_metadata_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples"
            metadata_dir = root / "metadata"
            samples.mkdir()
            metadata_dir.mkdir()
            records = build_manifest(
                samples_per_class=1,
                base_seed=10,
                split_name="toy",
                world_size=2,
                batch_size=1,
                num_classes=2,
            )
            manifest_path = root / "input_manifest.jsonl"
            manifest_info = write_manifest(
                records, manifest_path, base_seed=10, world_size=2, batch_size=1
            )
            input_config = {"seacache": {"mode": "full", "threshold": None}}
            (root / "config_resolved.yaml").write_text(
                yaml.safe_dump(input_config), encoding="utf-8"
            )
            input_hash = canonical_hash(input_config)
            run_config = {"input_config_hash": input_hash}
            run = {
                "config": run_config,
                "config_hash": canonical_hash(run_config),
                "input_config_hash": input_hash,
                "manifest_sha256": manifest_info["manifest_sha256"],
                "manifest_records_sha256": manifest_records_sha256(records),
                "method": "full",
                "threshold": None,
            }
            run_path = root / "run_manifest.json"
            atomic_write_json(run_path, run)
            create_once_path = root / "create_once.json"
            self.assertTrue(atomic_create_json(create_once_path, {"winner": 1}))
            self.assertFalse(atomic_create_json(create_once_path, {"winner": 2}))
            self.assertEqual(json.loads(create_once_path.read_text()), {"winner": 1})
            for rank in range(2):
                record = next(item for item in records if item.shard_id == rank)
                row = {"sample_id": record.sample_id, "status": "ok"}
                (metadata_dir / f"rank_{rank}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )
            loaded = load_rank_metadata(metadata_dir, records, world_size=2)
            self.assertEqual(set(loaded), {0, 1})
            self.assertEqual(
                validate_run_artifacts(
                    run_metadata_path=run_path,
                    sample_dir=samples,
                    supplied_manifest_path=manifest_path,
                    run_metadata=run,
                    manifest_records_sha256=manifest_records_sha256(records),
                ),
                root,
            )
            with self.assertRaises(ValueError):
                validate_run_artifacts(
                    run_metadata_path=run_path,
                    sample_dir=samples,
                    supplied_manifest_path=manifest_path,
                    run_metadata={**run, "method": "seacache"},
                    manifest_records_sha256=manifest_records_sha256(records),
                )

    def test_strict_pairing(self):
        validate_paired_runs(self.run, dict(self.run))
        changed = dict(self.run)
        changed["dtype"] = "other"
        with self.assertRaises(ValueError):
            validate_paired_runs(self.run, changed)
        missing = dict(self.run)
        del missing["sampler"]
        with self.assertRaises(ValueError):
            validate_paired_runs(self.run, missing)

    def test_distribution_deltas(self):
        full = {
            "fid": 2.0,
            "sfid": 3.0,
            "inception_score": 10.0,
            "precision": 0.8,
            "recall": 0.7,
            "sample_count": 50000,
            "resolution": 256,
            "reference_npz": "reference.npz",
            "reference_npz_size": 123,
            "reference_npz_sha256": "reference-hash",
            "manifest_sha256": "manifest",
            "manifest_records_sha256": "canonical-manifest",
            "evaluator_commit_or_version": "git:example",
            "evaluator_path": "evaluator.py",
            "evaluator_sha256": "evaluator-hash",
            "image_conversion_protocol": "RGB uint8",
            "run_identity": dict(self.run),
            "method": "full",
            "threshold": None,
        }
        candidate = dict(full)
        candidate["method"] = "seacache"
        candidate["threshold"] = 0.1
        for key in ("fid", "sfid", "inception_score", "precision", "recall"):
            candidate[key] = full[key] + 0.5
        result = distribution_deltas(full, candidate)
        self.assertEqual(result["delta_fid"], 0.5)
        self.assertEqual(result["delta_inception_score"], 0.5)
        candidate["evaluator_commit_or_version"] = "git:different"
        with self.assertRaises(ValueError):
            distribution_deltas(full, candidate)
        candidate = dict(full)
        candidate["method"] = "seacache"
        candidate["threshold"] = 0.1
        candidate["run_identity"] = {**self.run, "noise_scale": "different"}
        with self.assertRaises(ValueError):
            distribution_deltas(full, candidate)
        with self.assertRaises(ValueError):
            distribution_deltas(candidate, full)


if __name__ == "__main__":
    unittest.main()
