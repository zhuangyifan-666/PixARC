from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_deferred_smoke_tests.sh"
)


def test_deferred_smoke_uses_resume_report_and_derived_counts():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--upstream-config" in source
    assert "--full-config" in source
    assert "--candidate-config" in source
    assert source.index('python "$SCRIPT_ROOT/run_gpu_model_parity.py"') < source.index(
        'DICACHE_INVOCATION_ID="smoke-upstream-'
    )
    assert "expected_nfe = int(parity.get" in source
    assert "expected_forwards = int(parity.get" in source
    assert "nine_resume_invariants" in source
    assert "candidate_sample_is_finite_before_uint8" in source
    assert "candidate_runtime_reset" in source
    assert "candidate_cache_is_zero" in source
    for field in (
        "sum_direct_full_count",
        "sum_resumed_full_count",
        "sum_reuse_count",
        "sum_probe_count",
        "sum_dcta_count",
        "all_call_counts_valid",
        "trajectory_total_nfe",
        "trajectory_total_stream_calls",
        "trajectory_network_forward_count",
    ):
        assert field in source
    for forbidden in ("= 99", "== 99", "= 198", "== 198"):
        assert forbidden not in source
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
