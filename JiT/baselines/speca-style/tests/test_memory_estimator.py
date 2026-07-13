from __future__ import annotations

from speca_style.memory import estimate_speca_memory


CONFIG = {
    "depth": 12, "hidden_size": 768, "input_size": 256,
    "patch_size": 16, "in_context_len": 32, "in_context_start": 4,
}


def estimate(batch=1, order=2, layout="jit_dual_stream"):
    return estimate_speca_memory(
        CONFIG, batch_size=batch, dtype="bfloat16", max_order=order,
        cfg_layout=layout, verify_layer=-1,
    )


def test_cache_and_verifier_memory_are_monotonic():
    assert estimate(order=3)["taylor_cache_bytes"] > estimate(order=2)["taylor_cache_bytes"]
    assert estimate(batch=2)["taylor_cache_bytes"] == 2 * estimate(batch=1)["taylor_cache_bytes"]
    assert estimate(batch=2)["verifier_temporary_bytes"] == 2 * estimate(batch=1)["verifier_temporary_bytes"]


def test_jit_dual_history_has_same_factor_elements_as_combined_2b():
    assert estimate(layout="jit_dual_stream")["taylor_cache_bytes"] == estimate(layout="pixelgen_combined_2b")["taylor_cache_bytes"]


def test_report_exposes_required_components():
    report = estimate()
    assert report["taylor_cache_tensor_count"] == 2 * 12 * 2 * 3
    assert report["verifier_retained_payload_bytes"] > 0
    assert report["verifier_exact_branch_feature_bytes"] > 0
    assert report["error_reduction_temporary_bytes"] > 0
    assert report["analytic_peak_increment_bytes"] == report["taylor_cache_bytes"] + report["verifier_temporary_bytes"]
