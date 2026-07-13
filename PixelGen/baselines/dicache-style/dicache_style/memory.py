"""Analytic and observed DiCache cache-memory accounting."""

from __future__ import annotations

from typing import Mapping


COMMON_CORE_VERSION = "dicache-core-v1"
DTYPE_BYTES = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
    "float64": 8,
    "fp64": 8,
}


def _dtype_bytes(dtype: object) -> int:
    name = str(dtype).replace("torch.", "").lower()
    if name not in DTYPE_BYTES:
        raise ValueError(f"unsupported cache dtype: {dtype!r}")
    return DTYPE_BYTES[name]


def estimate_dicache_memory(
    model_config: Mapping[str, int],
    *,
    batch_size: int,
    dtype: object,
    cfg_layout: str,
    probe_depth: int = 1,
    residual_anchor_count: int = 2,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    depth = int(model_config["depth"])
    if not 1 <= probe_depth <= depth:
        raise ValueError("probe_depth must lie within model depth")
    if residual_anchor_count != 2:
        raise ValueError("released DCTA retains exactly two anchors")
    input_size = int(model_config["input_size"])
    patch_size = int(model_config["patch_size"])
    hidden = int(model_config["hidden_size"])
    context_len = int(model_config.get("in_context_len", 0))
    context_start = int(model_config.get("in_context_start", depth))
    image_tokens = (input_size // patch_size) ** 2
    if cfg_layout == "jit_dual_stream":
        streams, effective_batch = 2, batch_size
    elif cfg_layout == "pixelgen_combined_2b":
        streams, effective_batch = 1, 2 * batch_size
    elif cfg_layout == "single":
        streams, effective_batch = 1, batch_size
    else:
        raise ValueError(f"unsupported cfg_layout: {cfg_layout}")
    element_bytes = _dtype_bytes(dtype)
    feature_bytes = effective_batch * image_tokens * hidden * element_bytes
    previous_state_bytes = streams * 2 * feature_bytes
    full_residual_anchor_bytes = streams * residual_anchor_count * feature_bytes
    probe_residual_anchor_bytes = streams * residual_anchor_count * feature_bytes
    total_cache_bytes = (
        previous_state_bytes + full_residual_anchor_bytes + probe_residual_anchor_bytes
    )
    cache_tensor_count = streams * (2 + 2 * residual_anchor_count)
    probe_has_context = probe_depth > context_start
    probe_tokens = image_tokens + (context_len if probe_has_context else 0)
    temporary_probe_bytes = effective_batch * probe_tokens * hidden * element_bytes
    return {
        "previous_state_bytes": previous_state_bytes,
        "full_residual_anchor_bytes": full_residual_anchor_bytes,
        "probe_residual_anchor_bytes": probe_residual_anchor_bytes,
        "temporary_probe_bytes": temporary_probe_bytes,
        "total_cache_bytes": total_cache_bytes,
        "total_cache_gib": total_cache_bytes / 2**30,
        "cache_tensor_count": cache_tensor_count,
        "streams": streams,
        "effective_batch_per_stream": effective_batch,
        "image_tokens": image_tokens,
        "probe_internal_tokens": probe_tokens,
        "probe_depth": probe_depth,
        "residual_anchor_count": residual_anchor_count,
        "dtype_bytes": element_bytes,
        "cfg_layout": cfg_layout,
        "excludes_suffix_and_kernel_workspaces": True,
    }


def runtime_cache_memory(runtime: object) -> dict[str, int]:
    return {
        "cache_allocated_bytes": int(runtime.cache_bytes()),
        "cache_tensor_count": int(runtime.tensor_count()),
    }


__all__ = [
    "COMMON_CORE_VERSION",
    "DTYPE_BYTES",
    "estimate_dicache_memory",
    "runtime_cache_memory",
]
