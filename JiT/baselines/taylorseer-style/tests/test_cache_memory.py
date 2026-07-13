import torch

from taylorseer_style.memory import estimate_taylor_cache_bytes
from taylorseer_style.state import TaylorStreamState


MODEL = {"depth": 12, "hidden_size": 768, "input_size": 256, "patch_size": 16, "in_context_len": 32, "in_context_start": 4}


def test_estimator_monotonic_and_cfg_layouts():
    low = estimate_taylor_cache_bytes(MODEL, batch_size=1, dtype="bf16", max_order=1, cfg_layout="single")
    order = estimate_taylor_cache_bytes(MODEL, batch_size=1, dtype="bf16", max_order=3, cfg_layout="single")
    batch = estimate_taylor_cache_bytes(MODEL, batch_size=2, dtype="bf16", max_order=1, cfg_layout="single")
    dual = estimate_taylor_cache_bytes(MODEL, batch_size=1, dtype="bf16", max_order=1, cfg_layout="jit_dual_stream")
    combined = estimate_taylor_cache_bytes(MODEL, batch_size=1, dtype="bf16", max_order=1, cfg_layout="pixelgen_combined_2b")
    assert order["cache_bytes"] > low["cache_bytes"]
    assert batch["cache_bytes"] == dual["cache_bytes"] == combined["cache_bytes"] == 2 * low["cache_bytes"]


def test_unique_storage_and_reset():
    stream = TaylorStreamState("s")
    value = torch.zeros(2, 3)
    stream.update_exact((0, "attn"), value, coordinate=2, max_order=0, cache_dtype="inherit")
    stream.module_states[(0, "mlp")] = stream.module_states[(0, "attn")]
    assert stream.tensor_count() == 2
    assert stream.cache_bytes() == value.untyped_storage().nbytes()
    stream.reset()
    assert stream.cache_bytes() == stream.tensor_count() == 0

