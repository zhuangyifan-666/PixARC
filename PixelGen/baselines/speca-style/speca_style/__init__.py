"""CPU-safe package root for the unofficial PixelGen SpeCa-style port."""

from __future__ import annotations

from importlib import import_module
from typing import Any


COMMON_IMPLEMENTATION_VERSION = "speca-core-v1"
__version__ = "0.2.0"

_LAZY_EXPORTS = {
    "SpeCaPixelGenJiT": (".pixelgen_model", "SpeCaPixelGenJiT"),
    "SpeCaHeunSamplerJiT": (".pixelgen_sampler", "SpeCaHeunSamplerJiT"),
    "SpeCaPixelGenLightning": (".pixelgen_lightning", "SpeCaPixelGenLightning"),
    "InferenceOnlyTrainer": (".pixelgen_lightning", "InferenceOnlyTrainer"),
    "SpeCaRuntime": (".runtime", "SpeCaRuntime"),
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
