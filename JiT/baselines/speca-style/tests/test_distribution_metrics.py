from speca_style.distribution_metrics import distribution_deltas


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
