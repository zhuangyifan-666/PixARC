"""JiT probe/suffix execution with explicit context-token state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ProbeInternalState:
    tokens: torch.Tensor
    next_block_index: int
    context_inserted: bool


@dataclass(frozen=True)
class ProbeResult:
    internal_state: ProbeInternalState
    image_feature: torch.Tensor


def extract_image_tokens(
    tokens: torch.Tensor, *, context_inserted: bool, context_len: int
) -> torch.Tensor:
    if context_len < 0:
        raise ValueError("context_len cannot be negative")
    if not context_inserted:
        return tokens
    if context_len == 0 or tokens.shape[1] <= context_len:
        raise ValueError("invalid prepended context-token layout")
    # Upstream JiT prepends class context, so image tokens occupy the suffix.
    return tokens[:, context_len:]


def run_block_range(
    *,
    blocks: Sequence[nn.Module],
    tokens: torch.Tensor,
    conditioning: torch.Tensor,
    class_embedding: torch.Tensor,
    start: int,
    end: int,
    in_context_start: int,
    in_context_len: int,
    in_context_posemb: torch.Tensor | None,
    rope_before: object,
    rope_after: object,
    context_inserted: bool = False,
    on_block: Callable[[int], None] | None = None,
) -> ProbeInternalState:
    """Execute a half-open JiT block range without repeating context insertion."""

    depth = len(blocks)
    if not 0 <= start <= end <= depth:
        raise ValueError("invalid block range")
    if not 0 <= in_context_start <= depth:
        raise ValueError("in_context_start is outside model depth")
    value = tokens
    inserted = bool(context_inserted)
    for index in range(start, end):
        if in_context_len > 0 and index == in_context_start:
            if inserted:
                raise RuntimeError("context tokens would be inserted twice")
            if in_context_posemb is None:
                raise RuntimeError("context posemb is required")
            context = class_embedding.unsqueeze(1).repeat(1, in_context_len, 1)
            # Match upstream in-place behavior under autocast.
            context += in_context_posemb
            value = torch.cat([context, value], dim=1)
            inserted = True
        rope = rope_before if index < in_context_start else rope_after
        value = blocks[index](value, conditioning, rope)
        if on_block is not None:
            on_block(index)
    return ProbeInternalState(value, end, inserted)


def run_probe(
    *,
    blocks: Sequence[nn.Module],
    body_input: torch.Tensor,
    conditioning: torch.Tensor,
    class_embedding: torch.Tensor,
    probe_depth: int,
    in_context_start: int,
    in_context_len: int,
    in_context_posemb: torch.Tensor | None,
    rope_before: object,
    rope_after: object,
    on_block: Callable[[int], None] | None = None,
) -> ProbeResult:
    if not 1 <= probe_depth <= len(blocks):
        raise ValueError("probe_depth must be within model depth")
    internal = run_block_range(
        blocks=blocks,
        tokens=body_input,
        conditioning=conditioning,
        class_embedding=class_embedding,
        start=0,
        end=probe_depth,
        in_context_start=in_context_start,
        in_context_len=in_context_len,
        in_context_posemb=in_context_posemb,
        rope_before=rope_before,
        rope_after=rope_after,
        context_inserted=False,
        on_block=on_block,
    )
    image = extract_image_tokens(
        internal.tokens,
        context_inserted=internal.context_inserted,
        context_len=in_context_len,
    )
    if image.shape != body_input.shape:
        raise ValueError("probe image feature does not match body_input shape")
    return ProbeResult(internal, image)


def resume_from_probe(
    result: ProbeResult,
    *,
    blocks: Sequence[nn.Module],
    conditioning: torch.Tensor,
    class_embedding: torch.Tensor,
    in_context_start: int,
    in_context_len: int,
    in_context_posemb: torch.Tensor | None,
    rope_before: object,
    rope_after: object,
    on_block: Callable[[int], None] | None = None,
) -> torch.Tensor:
    internal = result.internal_state
    finished = run_block_range(
        blocks=blocks,
        tokens=internal.tokens,
        conditioning=conditioning,
        class_embedding=class_embedding,
        start=internal.next_block_index,
        end=len(blocks),
        in_context_start=in_context_start,
        in_context_len=in_context_len,
        in_context_posemb=in_context_posemb,
        rope_before=rope_before,
        rope_after=rope_after,
        context_inserted=internal.context_inserted,
        on_block=on_block,
    )
    return extract_image_tokens(
        finished.tokens,
        context_inserted=finished.context_inserted,
        context_len=in_context_len,
    )


__all__ = [
    "ProbeInternalState",
    "ProbeResult",
    "extract_image_tokens",
    "resume_from_probe",
    "run_block_range",
    "run_probe",
]
