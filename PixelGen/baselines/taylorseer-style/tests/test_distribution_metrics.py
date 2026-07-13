from taylorseer_style.distribution_metrics import distribution_deltas


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
        "interval": 1 if method == "instrumented_full" else 4,
        "max_order": 0 if method == "instrumented_full" else 3,
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
        "taylorseer",
        fid=2.5,
        sfid=2.75,
        inception_score=99.0,
        precision=0.75,
        recall=0.72,
    )
    assert distribution_deltas(full, candidate) == {
        "delta_fid": 0.5,
        "delta_sfid": -0.25,
        "delta_is": -1.0,
        "delta_precision": -0.050000000000000044,
        "delta_recall": 0.020000000000000018,
    }
