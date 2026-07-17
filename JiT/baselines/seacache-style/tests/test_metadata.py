import tempfile
import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

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
    validate_archived_model_configs,
    validate_paired_runs,
    validate_run_artifacts,
)
from seacache_style.distribution_metrics import (
    build_adm_sample_npz,
    distribution_deltas,
    run_adm_evaluator,
)


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

    def test_sample_npz_closes_backing_memmap_before_temp_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples"
            samples.mkdir()
            records = build_manifest(
                samples_per_class=1,
                base_seed=10,
                split_name="toy",
                world_size=1,
                batch_size=1,
                num_classes=1,
            )
            Image.new("RGB", (8, 8), color=(1, 2, 3)).save(
                samples / "000000.png"
            )
            captured = []
            real_open_memmap = np.lib.format.open_memmap

            def capture_memmap(*args, **kwargs):
                value = real_open_memmap(*args, **kwargs)
                captured.append(value)
                return value

            with patch(
                "seacache_style.distribution_metrics.np.lib.format.open_memmap",
                side_effect=capture_memmap,
            ):
                build_adm_sample_npz(
                    sample_dir=samples,
                    manifest=records,
                    output_npz=root / "samples.npz",
                    resolution=8,
                )
            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0]._mmap.closed)

    def test_adm_evaluator_runs_from_its_own_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator_dir = root / "evaluator"
            evaluator_dir.mkdir()
            evaluator = evaluator_dir / "evaluator.py"
            reference = root / "reference.npz"
            samples = root / "samples.npz"
            for path in (evaluator, reference, samples):
                path.write_bytes(b"placeholder")
            output = "\n".join(
                (
                    "FID: 1.0",
                    "sFID: 2.0",
                    "Inception Score: 3.0",
                    "precision: 0.4",
                    "recall: 0.5",
                )
            )
            with patch(
                "seacache_style.distribution_metrics.subprocess.run"
            ) as mocked_run:
                mocked_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=output
                )
                metrics, raw_output = run_adm_evaluator(
                    evaluator=evaluator,
                    reference_npz=reference,
                    sample_npz=samples,
                )
            self.assertEqual(metrics["fid"], 1.0)
            self.assertEqual(raw_output, output)
            self.assertEqual(mocked_run.call_args.kwargs["cwd"], str(evaluator_dir))

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

    def test_archived_model_comparison_ignores_only_checkpoint_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            reference_root.mkdir()
            candidate_root.mkdir()
            model = {"variant": "toy", "args": {"dropout": 0.0}}
            (reference_root / "config_resolved.yaml").write_text(
                yaml.safe_dump(
                    {"model": {**model, "checkpoint": "../checkpoint.pth"}}
                ),
                encoding="utf-8",
            )
            (candidate_root / "config_resolved.yaml").write_text(
                yaml.safe_dump(
                    {"model": {**model, "checkpoint": "/abs/checkpoint.pth"}}
                ),
                encoding="utf-8",
            )
            validate_archived_model_configs(reference_root, candidate_root)

            candidate = dict(self.run)
            candidate["model_config_hash"] = "different-spelling-hash"
            with self.assertRaises(ValueError):
                validate_paired_runs(self.run, candidate)
            validate_paired_runs(
                self.run,
                candidate,
                archived_model_configs_match=True,
            )

            (candidate_root / "config_resolved.yaml").write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "variant": "toy",
                            "args": {"dropout": 0.5},
                            "checkpoint": "/abs/checkpoint.pth",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_archived_model_configs(reference_root, candidate_root)

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
