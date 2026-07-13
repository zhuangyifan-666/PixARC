from __future__ import annotations

from speca_style.memory import estimate_speca_memory


CONFIG = {
    "depth": 28, "hidden_size": 1152, "input_size": 256,
    "patch_size": 16, "in_context_len": 32, "in_context_start": 8,
}


def estimate(batch=1, order=2):
    return estimate_speca_memory(
        CONFIG, batch_size=batch, dtype="bfloat16", max_order=order,
        cfg_layout="pixelgen_combined_2b", verify_layer=-1,
    )


def test_order_and_real_batch_scale_memory():
    assert estimate(order=4)["taylor_cache_bytes"] > estimate(order=2)["taylor_cache_bytes"]
    assert estimate(batch=2)["taylor_cache_bytes"] == 2 * estimate(batch=1)["taylor_cache_bytes"]
    assert estimate(batch=1)["effective_batch_per_stream"] == 2


def test_verifier_accounting_is_present():
    report = estimate()
    assert report["taylor_cache_tensor_count"] == 28 * 2 * 3
    assert report["verify_layer"] == 27
    assert report["verify_layer_token_count"] == 288
    assert report["verifier_temporary_bytes"] > 0
    assert report["analytic_peak_increment_bytes"] == report["taylor_cache_bytes"] + report["verifier_temporary_bytes"]
