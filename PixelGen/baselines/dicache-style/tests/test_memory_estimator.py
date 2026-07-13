from dicache_style.memory import estimate_dicache_memory


CONFIG = {
    "input_size": 256,
    "patch_size": 16,
    "hidden_size": 1152,
    "depth": 28,
    "in_context_len": 32,
    "in_context_start": 8,
}


def test_pixelgen_combined_2b_memory_and_batch_monotonicity():
    one = estimate_dicache_memory(
        CONFIG,
        batch_size=1,
        dtype="bf16",
        cfg_layout="pixelgen_combined_2b",
        probe_depth=1,
    )
    four = estimate_dicache_memory(
        CONFIG,
        batch_size=4,
        dtype="bf16",
        cfg_layout="pixelgen_combined_2b",
        probe_depth=1,
    )
    assert one["effective_batch_per_stream"] == 2
    assert one["total_cache_bytes"] == 7_077_888
    assert one["cache_tensor_count"] == 6
    assert four["total_cache_bytes"] == 4 * one["total_cache_bytes"]


def test_probe_depth_crossing_context_changes_temporary_not_cache():
    shallow = estimate_dicache_memory(
        CONFIG, batch_size=1, dtype="bf16", cfg_layout="pixelgen_combined_2b", probe_depth=1
    )
    deep = estimate_dicache_memory(
        CONFIG, batch_size=1, dtype="bf16", cfg_layout="pixelgen_combined_2b", probe_depth=9
    )
    assert deep["total_cache_bytes"] == shallow["total_cache_bytes"]
    assert deep["temporary_probe_bytes"] > shallow["temporary_probe_bytes"]
