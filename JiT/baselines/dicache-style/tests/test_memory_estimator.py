from dicache_style.memory import estimate_dicache_memory

MODEL = {"depth": 12, "hidden_size": 768, "input_size": 256, "patch_size": 16,
         "in_context_len": 32, "in_context_start": 4}


def test_jit_dual_stream_memory_and_batch_monotonicity():
    one = estimate_dicache_memory(MODEL, batch_size=1, dtype="bfloat16",
                                  cfg_layout="jit_dual_stream", probe_depth=1)
    two = estimate_dicache_memory(MODEL, batch_size=2, dtype="bfloat16",
                                  cfg_layout="jit_dual_stream", probe_depth=5)
    assert one["streams"] == 2 and one["cache_tensor_count"] == 12
    assert one["total_cache_bytes"] == 4_718_592
    assert two["total_cache_bytes"] == 2 * one["total_cache_bytes"]
    assert two["temporary_probe_bytes"] > one["temporary_probe_bytes"]
