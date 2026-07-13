"""Derived Heun network-evaluation counts (never hard-coded as 99)."""

from __future__ import annotations


def expected_nfe_count(
    sampler: str, num_steps: int, *, exact_heun: bool = True
) -> int:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if sampler.lower() not in {"heun", "henu"}:
        raise ValueError("only PixelGen Heun is supported")
    return 2 * num_steps - 1 if exact_heun else num_steps


def expected_network_forward_count(
    *, model_family: str, sampler: str, num_steps: int, exact_heun: bool = True
) -> int:
    if model_family.lower() != "pixelgen":
        raise ValueError("this adapter derives PixelGen combined-CFG forwards")
    return expected_nfe_count(sampler, num_steps, exact_heun=exact_heun)


def heun_stage_sequence(num_steps: int, *, exact_heun: bool = True):
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    result: list[tuple[int, str]] = []
    for index in range(num_steps):
        if index == 0 or exact_heun:
            result.append(
                (index, "final_euler" if index == num_steps - 1 else "predictor")
            )
        if index < num_steps - 1:
            result.append((index, "corrector"))
    return tuple(result)


__all__ = [
    "expected_network_forward_count",
    "expected_nfe_count",
    "heun_stage_sequence",
]
