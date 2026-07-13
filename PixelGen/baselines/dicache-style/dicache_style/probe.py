"""Probe/resume value objects and image-token extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProbeResult:
    internal_state: torch.Tensor
    image_feature: torch.Tensor
    next_block_index: int
    context_inserted: bool
    image_token_start: int


def extract_image_tokens(
    internal_state: torch.Tensor,
    *,
    body_token_count: int,
    context_inserted: bool,
    context_token_count: int,
) -> tuple[torch.Tensor, int]:
    if internal_state.ndim != 3:
        raise ValueError("Transformer internal state must be [batch,tokens,channels]")
    start = context_token_count if context_inserted else 0
    if internal_state.shape[1] != body_token_count + start:
        raise ValueError(
            "context/image token layout mismatch; refusing shape-by-truncation"
        )
    image = internal_state[:, start:, :]
    if image.shape[1] != body_token_count:
        raise AssertionError("image-token extraction failed")
    return image, start


__all__ = ["ProbeResult", "extract_image_tokens"]
