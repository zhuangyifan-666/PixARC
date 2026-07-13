"""Local final-block verification primitives for released-code SpeCa."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


COMMON_CORE_VERSION = "speca-core-v1"
TOKEN_SCOPES = frozenset({"all_tokens", "image_tokens_only"})


@dataclass(frozen=True)
class VerificationPayload:
    """A transient pair; it is consumed before the NFE is finalized."""

    pred: torch.Tensor
    exact: torch.Tensor
    stream_id: str
    layer_idx: int
    token_scope: str
    image_token_start: int

    def __post_init__(self) -> None:
        if self.pred.shape != self.exact.shape:
            raise ValueError("verification pred/exact shapes differ")
        if self.pred.dtype != self.exact.dtype or self.pred.device != self.exact.device:
            raise ValueError("verification pred/exact tensor contexts differ")
        if self.pred.ndim < 2:
            raise ValueError("verification block outputs must include batch and token axes")
        if self.token_scope not in TOKEN_SCOPES:
            raise ValueError(f"unsupported verification token scope {self.token_scope!r}")
        if self.image_token_start < 0 or self.image_token_start > self.pred.shape[1]:
            raise ValueError("invalid image_token_start")

    @property
    def selected_pred(self) -> torch.Tensor:
        if self.token_scope == "all_tokens":
            return self.pred
        return self.pred[:, self.image_token_start :]

    @property
    def selected_exact(self) -> torch.Tensor:
        if self.token_scope == "all_tokens":
            return self.exact
        return self.exact[:, self.image_token_start :]

    @property
    def element_count(self) -> int:
        return int(self.selected_pred.numel())

    @property
    def batch_size(self) -> int:
        return int(self.pred.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.selected_pred.shape[1])


def resolve_verify_layer(verify_layer: int, *, depth: int) -> int:
    if depth <= 0:
        raise ValueError("model depth must be positive")
    value = int(verify_layer)
    if value == -1:
        return depth - 1
    if value < 0 or value >= depth:
        raise ValueError(f"verify_layer {value} is outside depth {depth}")
    return value


def local_verify(
    x_verify_in: torch.Tensor,
    *,
    draft_block: Callable[[torch.Tensor], torch.Tensor],
    exact_block: Callable[[torch.Tensor], torch.Tensor],
    stream_id: str,
    layer_idx: int,
    token_scope: str = "all_tokens",
    image_token_start: int = 0,
    clone_exact_input: bool = True,
) -> tuple[torch.Tensor, VerificationPayload]:
    """Execute draft and exact branches from one speculative-prefix input.

    The returned model output is always ``x_pred``.  The exact result is only
    present in the transient payload and is never substituted into the model
    path or Taylor history.
    """

    if token_scope not in TOKEN_SCOPES:
        raise ValueError(f"token_scope must be one of {sorted(TOKEN_SCOPES)}")
    x_pred = draft_block(x_verify_in)
    exact_input = x_verify_in.clone() if clone_exact_input else x_verify_in
    x_exact = exact_block(exact_input)
    payload = VerificationPayload(
        pred=x_pred,
        exact=x_exact,
        stream_id=str(stream_id),
        layer_idx=int(layer_idx),
        token_scope=token_scope,
        image_token_start=int(image_token_start),
    )
    return x_pred, payload


__all__ = [
    "COMMON_CORE_VERSION",
    "TOKEN_SCOPES",
    "VerificationPayload",
    "local_verify",
    "resolve_verify_layer",
]
