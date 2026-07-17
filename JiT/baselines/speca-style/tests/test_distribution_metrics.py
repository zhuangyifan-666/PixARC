from pathlib import Path

import pytest

from speca_style.distribution_metrics import distribution_deltas, run_adm_evaluator


def _report(method, **metrics):
    context = {
        "sample_count": 50000,
        "resolution": 256,
        "reference_npz": "/reference/imagenet.npz",
        "reference_npz_size": 123,
        "reference_npz_sha256": "reference-hash",
        "manifest_sha256": "manifest-hash",
        "manifest_records_sha256": "records-hash",
        "evaluator_commit_or_version": "git:evaluator",
        "evaluator_path": "/tools/evaluator.py",
        "evaluator_sha256": "evaluator-hash",
        "image_conversion_protocol": "numeric RGB",
        "run_identity": {"paired": True},
    }
    return {
        **context,
        **metrics,
        "method": method,
        "scheduler_mode": "released_code_faithful",
        "interval": None,
        "max_order": 4,
        "base_threshold": 0.3,
        "decay_rate": 0.05,
        "min_taylor_steps": 3,
        "max_taylor_steps": 8,
        "first_enhance": 3,
        "threshold_floor": 0.01,
        "error_metric": "relative_l1",
        "error_eps": 1.0e-10,
        "verify_layer": -1,
        "verification_token_scope": "all_tokens",
        "gate_mode": "batch_global",
        "coordinate_mode": "official_nfe_index",
        "force_last_full": False,
        "cache_dtype": "inherit",
        "trace_mode": "summary",
    }


def test_distribution_delta_names_and_direction():
    full = _report(
        "instrumented_full",
        fid=2.0,
        sfid=3.0,
        inception_score=100.0,
        precision=0.8,
        recall=0.7,
    )
    candidate = _report(
        "speca",
        fid=2.5,
        sfid=2.75,
        inception_score=99.0,
        precision=0.75,
        recall=0.72,
    )
    # FID-style distribution comparison is intentionally not sample paired.
    candidate["manifest_sha256"] = "different-manifest"
    candidate["manifest_records_sha256"] = "different-records"
    candidate["run_identity"] = {"paired": False}
    assert distribution_deltas(full, candidate) == {
        "delta_fid": 0.5,
        "delta_sfid": -0.25,
        "delta_is": -1.0,
        "delta_precision": -0.050000000000000044,
        "delta_recall": 0.020000000000000018,
    }


def _fake_adm_tree(tmp_path: Path, body: str):
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(body, encoding="utf-8")
    (tmp_path / "classify_image_graph_def.pb").write_bytes(b"local-graph")
    reference = tmp_path / "reference.npz"
    sample = tmp_path / "sample.npz"
    reference.write_bytes(b"reference")
    sample.write_bytes(b"sample")
    return evaluator, reference, sample


def test_adm_evaluator_runs_from_local_graph_directory(tmp_path):
    evaluator, reference, sample = _fake_adm_tree(
        tmp_path,
        "import os\n"
        "assert os.path.isfile('classify_image_graph_def.pb')\n"
        "print('Inception Score: 1.5')\n"
        "print('FID: 2.5')\n"
        "print('sFID: 3.5')\n"
        "print('Precision: 0.75')\n"
        "print('Recall: 0.5')\n",
    )
    metrics, output = run_adm_evaluator(
        evaluator=evaluator, reference_npz=reference, sample_npz=sample
    )
    assert metrics == {
        "inception_score": 1.5,
        "fid": 2.5,
        "sfid": 3.5,
        "precision": 0.75,
        "recall": 0.5,
    }
    assert "Inception Score" in output


def test_adm_evaluator_failure_retains_child_output(tmp_path):
    evaluator, reference, sample = _fake_adm_tree(
        tmp_path, "print('specific evaluator failure')\nraise SystemExit(7)\n"
    )
    with pytest.raises(RuntimeError, match="specific evaluator failure"):
        run_adm_evaluator(
            evaluator=evaluator, reference_npz=reference, sample_npz=sample
        )
