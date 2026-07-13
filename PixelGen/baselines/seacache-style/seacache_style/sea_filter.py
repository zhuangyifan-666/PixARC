"""Baseline-faithful SEA filtering primitives.

The implementation is independently written from the mathematical behavior in
``baselines/SeaCache/FLUX/util_seacache.py`` at commit
8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2.  The default path deliberately
uses full FFTs and does not cache frequency grids.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


COMMON_IMPLEMENTATION_VERSION = "pixarc-seacache-style-v1"


def _canonical_dims(ndim: int, dims: Iterable[int] | None) -> Tuple[int, ...]:
    if dims is None:
        dims = tuple(range(ndim)) if ndim <= 2 else tuple(range(-2, -ndim, -1))
    result = tuple(int(d) if int(d) >= 0 else ndim + int(d) for d in dims)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"dims must be non-empty and unique, got {dims!r}")
    if min(result) < 0 or max(result) >= ndim:
        raise ValueError(f"dims {dims!r} are invalid for a {ndim}-D tensor")
    return result


def _rfft_full_mean_weights_1d(
    n_last: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    length = n_last // 2 + 1
    weights = torch.ones(length, device=device, dtype=dtype)
    if n_last % 2 == 0:
        if length > 2:
            weights[1:-1] *= 2.0
    elif length > 1:
        weights[1:] *= 2.0
    return weights


def apply_sea_from_ab(
    x: torch.Tensor,
    a: float,
    b: float,
    power_exp: float = 2.0,
    power_const: float = 1.0,
    dims: Iterable[int] | None = (-2, -3),
    eps: float = 1e-16,
    norm_mode: str = "mean",
    *,
    real: bool = False,
) -> torch.Tensor:
    """Apply the separable SEA Wiener gain while preserving shape and dtype."""

    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError(f"x must have a floating dtype, got {x.dtype}")
    if power_exp <= 0 or power_const <= 0 or eps <= 0:
        raise ValueError("power_exp, power_const, and eps must be positive")

    original_dtype = x.dtype
    x32 = x.contiguous().to(torch.float32)
    canonical_dims = _canonical_dims(x32.ndim, dims)

    if real:
        spectrum = torch.fft.rfftn(x32, dim=canonical_dims)
    else:
        spectrum = torch.fft.fftn(x32, dim=canonical_dims)

    gain = None
    for index, axis in enumerate(canonical_dims):
        length = x32.shape[axis]
        if real and index == len(canonical_dims) - 1:
            frequency = torch.fft.rfftfreq(
                length, device=x32.device, dtype=torch.float32
            )
        else:
            frequency = torch.fft.fftfreq(
                length, device=x32.device, dtype=torch.float32
            )
        signal_power = power_const / (frequency.abs().pow(power_exp) + eps)
        axis_gain = (float(a) * signal_power) / (
            float(a) * float(a) * signal_power + float(b) * float(b) + eps
        )
        view_shape = [1] * x32.ndim
        view_shape[axis] = axis_gain.numel()
        axis_gain = axis_gain.reshape(view_shape)
        gain = axis_gain if gain is None else gain * axis_gain

    normalized = norm_mode.lower()
    if normalized == "peak":
        maximum = torch.amax(gain)
        # This Python condition intentionally mirrors the official sync semantics.
        if torch.isfinite(maximum) and maximum > 0:
            gain = gain / maximum
    elif normalized == "mean":
        if real:
            n_last = int(x32.shape[canonical_dims[-1]])
            weights = _rfft_full_mean_weights_1d(
                n_last, x32.device, torch.float32
            )
            weight_shape = [1] * x32.ndim
            weight_shape[canonical_dims[-1]] = weights.numel()
            weights = weights.reshape(weight_shape)
            other_bins = 1
            for axis in canonical_dims[:-1]:
                other_bins *= int(x32.shape[axis])
            mean_gain = torch.sum(gain * weights) / (torch.sum(weights) * other_bins)
        else:
            mean_gain = torch.mean(gain)
        # This Python condition intentionally mirrors the official sync semantics.
        if torch.isfinite(mean_gain) and mean_gain > 0:
            gain = gain / mean_gain
    elif normalized not in {"none", "off"}:
        raise ValueError(f"unsupported norm_mode: {norm_mode!r}")

    filtered_spectrum = spectrum * gain
    if real:
        sizes = [x32.shape[axis] for axis in canonical_dims]
        filtered = torch.fft.irfftn(
            filtered_spectrum, s=sizes, dim=canonical_dims
        )
    else:
        filtered = torch.fft.ifftn(filtered_spectrum, dim=canonical_dims).real
    return filtered.to(original_dtype)


def coefficients_from_time(
    t: torch.Tensor | float,
    *,
    tolerance: float = 1e-6,
    clamp_eps: float = 1e-6,
) -> tuple[float, float]:
    """Map ``z_t=t*x+(1-t)*noise`` to SEA ``(a,b)``.

    A batch must contain one shared time.  Returning Python floats intentionally
    preserves the scalar synchronization present in the audited baseline.
    """

    if torch.is_tensor(t):
        if t.numel() == 0:
            raise ValueError("t must not be empty")
        detached = t.detach()
        first = detached.reshape(-1)[0]
        if not torch.allclose(
            detached, first.expand_as(detached), rtol=0.0, atol=tolerance
        ):
            raise ValueError("all samples in a batch must use the same continuous time")
        value = float(first.cpu())
    else:
        value = float(t)
    if not (value == value) or value in {float("inf"), float("-inf")}:
        raise ValueError(f"t must be finite, got {value}")
    clamped = max(clamp_eps, min(1.0 - clamp_eps, value))
    return clamped, 1.0 - clamped


def rel_l1(current: torch.Tensor, previous: torch.Tensor, eps: float = 1e-16) -> float:
    """Official batch-global relative L1, returned as a host scalar."""

    if current.shape != previous.shape:
        raise ValueError(
            f"probe shape changed: current={tuple(current.shape)}, "
            f"previous={tuple(previous.shape)}"
        )
    numerator = (current - previous).abs().mean()
    denominator = previous.abs().mean() + eps
    return float((numerator / denominator).detach().cpu())
