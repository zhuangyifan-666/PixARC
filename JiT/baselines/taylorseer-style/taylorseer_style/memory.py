"""Analytic and observed Taylor cache memory accounting."""

from __future__ import annotations

from typing import Mapping


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
        raise ValueError(f"unsupported cache dtype {dtype!r}")
    return DTYPE_BYTES[name]


def estimate_taylor_cache_bytes(
    model_config: Mapping[str, int],
    *,
    batch_size: int,
    dtype: object,
    max_order: int,
    cfg_layout: str,
    context_token_layout: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Estimate full-order attn+MLP history using per-layer token layouts."""

    if batch_size <= 0 or max_order < 0:
        raise ValueError("batch_size must be positive and max_order non-negative")
    depth = int(model_config["depth"])
    hidden = int(model_config["hidden_size"])
    input_size = int(model_config["input_size"])
    patch_size = int(model_config["patch_size"])
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
        raise ValueError(f"unsupported cfg_layout {cfg_layout!r}")
    bytes_per_element = _dtype_bytes(dtype)
    factor_count = max_order + 1
    per_layer_tokens = [
        image_tokens + (context_len if index >= context_start else 0)
        for index in range(depth)
    ]
    elements_per_stream = sum(
        2 * factor_count * effective_batch * tokens * hidden
        for tokens in per_layer_tokens
    )
    total_bytes = streams * elements_per_stream * bytes_per_element
    tensor_count = streams * depth * 2 * factor_count
    return {
        "cache_bytes": total_bytes,
        "cache_gib": total_bytes / 2**30,
        "cache_tensor_count": tensor_count,
        "streams": streams,
        "effective_batch_per_stream": effective_batch,
        "modules_per_layer": 2,
        "factor_count": factor_count,
        "per_layer_tokens": per_layer_tokens,
        "hidden_size": hidden,
        "dtype_bytes": bytes_per_element,
        "cfg_layout": cfg_layout,
    }


def runtime_cache_memory(runtime: object) -> dict[str, int]:
    return {
        "cache_allocated_bytes": int(runtime.cache_bytes()),
        "cache_tensor_count": int(runtime.tensor_count()),
    }

