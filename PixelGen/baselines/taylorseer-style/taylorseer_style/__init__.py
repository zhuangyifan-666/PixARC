"""Unofficial TaylorSeer-style inference support for PixelGen.

Model-specific exports are lazy so importing a common-core utility never pulls
in PixelGen, Lightning, or CUDA-facing model modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


COMMON_IMPLEMENTATION_VERSION = "pixarc-taylorseer-style-v1"

_LAZY_EXPORTS = {
    "TaylorSeerPixelGenJiT": (".pixelgen_model", "TaylorSeerPixelGenJiT"),
    "TaylorSeerHeunSamplerJiT": (
        ".pixelgen_sampler",
        "TaylorSeerHeunSamplerJiT",
    ),
    "TaylorSeerPixelGenLightning": (
        ".pixelgen_lightning",
        "TaylorSeerPixelGenLightning",
    ),
    "InferenceOnlyTrainer": (".pixelgen_lightning", "InferenceOnlyTrainer"),
    "TaylorSeerRuntime": (".runtime", "TaylorSeerRuntime"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = ["COMMON_IMPLEMENTATION_VERSION", *_LAZY_EXPORTS]
