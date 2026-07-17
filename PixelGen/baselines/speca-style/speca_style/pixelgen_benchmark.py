"""Deferred PixelGen factory for matched instrumented-Full/SpeCa GPU timing."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Type

import torch
import yaml
from jsonargparse import ArgumentParser
from torch import nn

from src.diffusion.base.sampling import BaseSampler
from src.models.autoencoder.base import BaseAE, fp2uint8

from .latency import BenchmarkSpec
from .manifest import initial_noise
from .metadata import SPECA_CONFIG_FIELDS, canonical_hash, validate_speca_config
from .scheduler import expected_nfe_count


PIXARC_ROOT = Path(__file__).resolve().parents[4]


def _instantiate(specification: Mapping[str, Any], base_class: Type[Any]) -> Any:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(base_class, "component", required=True)
    parsed = parser.parse_object({"component": dict(specification)})
    return parser.instantiate_classes(parsed).component


def _checkpoint(path_value: str, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [config_path.parent / path, PIXARC_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint not found; checked {candidates}")


def _load_ema_denoiser(net: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise KeyError("PixelGen checkpoint must contain state_dict")
    prefix = "ema_denoiser."
    ema = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not ema:
        raise KeyError("checkpoint contains no ema_denoiser parameters")
    net.load_state_dict(ema, strict=True)
    del checkpoint


def _benchmark_denoiser_spec(
    specification: Mapping[str, Any], speca: Mapping[str, Any], compile_mode: str
) -> dict[str, Any]:
    """Return one denoiser spec with the complete released-code SpeCa protocol."""

    validate_speca_config(speca, require_resolved=True)
    if speca.get("mode") != "speca":
        raise ValueError("the candidate benchmark config must use speca.mode=speca")
    if speca.get("interval") is not None:
        raise ValueError("released-code SpeCa benchmark requires interval=null")
    result = dict(specification)
    expected_class = "speca_style.pixelgen_model.SpeCaPixelGenJiT"
    if result.get("class_path") != expected_class:
        raise ValueError(f"benchmark denoiser must be {expected_class}")
    init_args = dict(result.get("init_args", {}))
    init_args.update(
        {
            f"speca_{field}": speca[field]
            for field in SPECA_CONFIG_FIELDS
            if field != "scheduler_mode"
        }
    )
    init_args.update({"speca_mode": "speca", "compile_mode": compile_mode})
    result["init_args"] = init_args
    return result


def _sampler_name(specification: Mapping[str, Any]) -> str:
    name = str(specification.get("class_path", "")).lower()
    init_args = dict(specification.get("init_args", {}))
    if ("heun" in name or "henu" in name) and init_args.get("exact_henu"):
        return "exact_heun"
    if "heun" in name or "henu" in name:
        return "heun"
    if "adam" in name or "lms" in name:
        return "adam_lm"
    if "euler" in name:
        return "euler"
    raise ValueError(f"cannot identify sampler class {specification.get('class_path')!r}")


def build_benchmark_spec(runner: Mapping[str, Any]) -> BenchmarkSpec:
    """Construct one EMA model and two matched closures; explicitly CUDA-only."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PixelGen latency requires exactly one visible CUDA GPU")
    config_path = Path(str(runner["model_config"])).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "pixarc-speca-config-v1":
        raise ValueError("unsupported PixelGen SpeCa config schema")
    speca = dict(config["speca"])
    runtime = dict(config["runtime"])
    noise_scale_raw = runtime.get("noise_scale", 1.0)
    if isinstance(noise_scale_raw, bool) or not isinstance(
        noise_scale_raw, (int, float)
    ):
        raise ValueError("runtime.noise_scale must be a finite non-negative number")
    noise_scale = float(noise_scale_raw)
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("runtime.noise_scale must be a finite non-negative number")
    compile_mode = str(runtime.get("compile_mode", "matched_eager"))
    if compile_mode == "upstream":
        raise ValueError(
            "matched speedup cannot use upstream outer compile; use matched_eager or blockwise"
        )
    model_config = dict(config["model"])
    denoiser_spec = _benchmark_denoiser_spec(model_config["denoiser"], speca, compile_mode)
    expected_sampler = "speca_style.pixelgen_sampler.SpeCaHeunSamplerJiT"
    if model_config["diffusion_sampler"].get("class_path") != expected_sampler:
        raise ValueError(f"benchmark sampler must be {expected_sampler}")
    net = _instantiate(denoiser_spec, nn.Module)
    sampler = _instantiate(model_config["diffusion_sampler"], BaseSampler)
    vae = _instantiate(model_config["vae"], BaseAE)
    checkpoint_value = Path(str(config["checkpoint"])).expanduser()
    origin = Path(str(runner.get("config_origin_dir", config_path.parent))).resolve()
    checkpoint_path = _checkpoint(
        str(checkpoint_value if checkpoint_value.is_absolute() else origin / checkpoint_value),
        config_path,
    )
    _load_ema_denoiser(net, checkpoint_path)
    net = net.cuda().eval()
    sampler = sampler.cuda().eval()
    vae = vae.cuda().eval()
    net.compile()
    unwrapped_for_eager = int(getattr(net, "compile_wrappers_unwrapped", 0))

    batch_size = int(runner.get("batch_size", runtime["batch_size"]))
    if int(runtime["batch_size"]) != 4 or int(runtime["effective_cfg_batch_size"]) != 8:
        raise ValueError("primary PixelGen protocol requires real batch=4 and CFG batch=8")
    if batch_size != 4:
        raise ValueError("PixelGen SpeCa latency must use real batch_size=4")
    sample_ids = tuple(int(value) for value in runner["sample_ids"])
    seeds = tuple(int(value) for value in runner["seeds"])
    class_ids = tuple(int(value) for value in runner["class_ids"])
    if not (len(sample_ids) == len(seeds) == len(class_ids) == batch_size):
        raise ValueError("sample_ids, seeds, class_ids, and batch_size must match")
    input_size = int(model_config["denoiser"]["init_args"].get("input_size", 256))
    noise_cpu = initial_noise(
        seeds,
        (3, input_size, input_size),
        device="cpu",
        dtype=torch.float32,
    )
    noise = (noise_cpu * noise_scale).cuda()
    condition = torch.tensor(class_ids, device="cuda", dtype=torch.long)
    num_classes = int(model_config["denoiser"]["init_args"].get("num_classes", 1000))
    uncondition = torch.full_like(condition, num_classes)

    precision = str(runtime.get("precision", "bf16-mixed"))
    if precision.startswith("bf16"):
        autocast_dtype = torch.bfloat16
    elif precision.startswith("16"):
        autocast_dtype = torch.float16
    elif precision.startswith("32"):
        autocast_dtype = None
    else:
        raise ValueError(f"unsupported benchmark precision: {precision}")

    def _run(mode: str) -> torch.Tensor:
        net.speca_runtime.mode = mode
        sampler.set_speca_batch_context(
            sample_ids=sample_ids,
            trajectory_id=f"pixelgen-latency-{mode}",
        )
        autocast_context = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            samples = sampler(net, noise, condition, uncondition)
            return fp2uint8(vae.decode(samples))

    def full() -> torch.Tensor:
        return _run("instrumented_full")

    def candidate() -> torch.Tensor:
        return _run("speca")

    expected = expected_nfe_count(
        "heun", int(sampler.num_steps), exact_heun=bool(sampler.exact_henu)
    )

    def full_summary() -> dict[str, Any]:
        summary = sampler.last_speca_summary
        if not isinstance(summary, dict) or summary.get("mode") != "instrumented_full":
            raise RuntimeError("instrumented-Full benchmark has no completed summary")
        if int(summary.get("network_forward_count", -1)) != expected:
            raise RuntimeError("instrumented-Full network-forward count mismatch")
        return dict(summary)

    def candidate_summary() -> dict[str, Any]:
        summary = sampler.last_speca_summary
        if not isinstance(summary, dict):
            raise RuntimeError("SpeCa benchmark has no completed trajectory summary")
        if int(summary.get("network_forward_count", -1)) != expected:
            raise RuntimeError("SpeCa network-forward count mismatch")
        return dict(summary)

    full.speca_summary = full_summary  # type: ignore[attr-defined]
    candidate.speca_summary = candidate_summary  # type: ignore[attr-defined]
    sampler_args = dict(model_config["diffusion_sampler"]["init_args"])
    return BenchmarkSpec(
        full=full,
        speca=candidate,
        batch_size=batch_size,
        effective_cfg_batch_size=2 * batch_size,
        compile_mode=compile_mode,
        dtype=precision,
        metadata={
            "model": "PixelGen-JiT",
            "model_config_hash": canonical_hash(
                {
                    **{
                        key: value
                        for key, value in model_config.items()
                        if key not in {"diffusion_sampler", "denoiser", "compile_mode"}
                    },
                    "denoiser": {
                        "class_path": model_config["denoiser"]["class_path"],
                        "init_args": {
                            key: value
                            for key, value in model_config["denoiser"]["init_args"].items()
                            if not key.startswith("speca_") and key != "compile_mode"
                        },
                    },
                }
            ),
            "checkpoint": str(checkpoint_path),
            "ema": "ema_denoiser",
            "sampler": _sampler_name(model_config["diffusion_sampler"]),
            "sampler_config_hash": canonical_hash(
                model_config["diffusion_sampler"]
            ),
            "steps": int(sampler_args["num_steps"]),
            "cfg_scale": float(sampler_args["guidance"]),
            "guidance_interval": [
                float(sampler_args.get("guidance_interval_min", 0.0)),
                float(sampler_args.get("guidance_interval_max", 1.0)),
            ],
            "timeshift": float(sampler_args.get("timeshift", 1.0)),
            "sample_ids": list(sample_ids),
            "speca": {field: speca[field] for field in SPECA_CONFIG_FIELDS},
            "noise_scale": noise_scale,
            "cfg_execution": "single combined [unconditional, conditional] 2B forward",
            "compile_wrappers_unwrapped": unwrapped_for_eager,
            "outer_compile_enabled": False,
        },
    )


__all__ = ["build_benchmark_spec"]
