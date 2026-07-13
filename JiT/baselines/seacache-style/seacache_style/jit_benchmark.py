"""Deferred, model-specific JiT factory for the matched latency harness."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml

from .controller import finalize_timing_summary
from .jit_denoiser import SeaCacheDenoiser, expected_model_calls_per_stream
from .jit_model import configure_jit_compile_mode
from .latency import BenchmarkSpec
from .manifest import initial_noise
from .metadata import canonical_hash


PIXARC_ROOT = Path(__file__).resolve().parents[4]


def _args(config: Mapping[str, Any]) -> SimpleNamespace:
    model = dict(config["model"])
    sampling = dict(config["sampling"])
    extra = dict(model.get("args", {}))
    low, high = sampling.get("guidance_interval", [0.1, 1.0])
    return SimpleNamespace(
        model=model["variant"],
        img_size=int(model.get("image_size", 256)),
        class_num=int(model.get("num_classes", 1000)),
        attn_dropout=float(extra.get("attn_dropout", 0.0)),
        proj_dropout=float(extra.get("proj_dropout", 0.0)),
        label_drop_prob=float(extra.get("label_drop_prob", 0.1)),
        P_mean=float(extra.get("P_mean", -0.8)),
        P_std=float(extra.get("P_std", 0.8)),
        t_eps=float(extra.get("t_eps", 0.05)),
        noise_scale=float(sampling.get("noise_scale", 1.0)),
        ema_decay1=float(extra.get("ema_decay1", 0.9999)),
        ema_decay2=float(extra.get("ema_decay2", 0.9996)),
        sampling_method=str(sampling["method"]),
        num_sampling_steps=int(sampling["steps"]),
        cfg=float(sampling["cfg_scale"]),
        interval_min=float(low),
        interval_max=float(high),
    )


def _checkpoint(path_value: str, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [config_path.parent / path, PIXARC_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint not found; checked {candidates}")


def _load_ema1(model: SeaCacheDenoiser, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema = checkpoint["model_ema1"]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema:
                raise KeyError(f"EMA1 is missing {name}")
            parameter.copy_(ema[name])
    del checkpoint


def build_benchmark_spec(runner: Mapping[str, Any]) -> BenchmarkSpec:
    """Build matched Full/SeaCache closures; this function intentionally uses CUDA."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("JiT latency requires exactly one visible CUDA GPU")
    config_path = Path(str(runner["model_config"])).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seacache = dict(config["seacache"])
    threshold = seacache.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            "benchmark model_config must contain a finite non-negative numeric threshold"
        )
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value < 0:
        raise ValueError(
            "benchmark model_config must contain a finite non-negative numeric threshold"
        )
    sampling_config = dict(config["sampling"])
    noise_scale_raw = sampling_config.get("noise_scale", 1.0)
    if isinstance(noise_scale_raw, bool) or not isinstance(
        noise_scale_raw, (int, float)
    ):
        raise ValueError("sampling.noise_scale must be a finite non-negative number")
    if not math.isfinite(float(noise_scale_raw)) or float(noise_scale_raw) < 0:
        raise ValueError("sampling.noise_scale must be a finite non-negative number")
    runtime = dict(config["runtime"])
    batch_size = int(runner.get("batch_size", runtime["batch_size"]))
    sample_ids = tuple(int(value) for value in runner["sample_ids"])
    seeds = tuple(int(value) for value in runner["seeds"])
    labels_values = tuple(int(value) for value in runner["class_ids"])
    if not (len(sample_ids) == len(seeds) == len(labels_values) == batch_size):
        raise ValueError("sample_ids, seeds, class_ids, and batch_size must match")
    arguments = _args(config)
    checkpoint_path = _checkpoint(config["model"]["checkpoint"], config_path)
    model = SeaCacheDenoiser(
        arguments,
        seacache_mode="seacache",
        seacache_threshold=threshold_value,
        seacache_trace_mode="summary",
    )
    unwrapped_for_eager = configure_jit_compile_mode(
        model.net, str(runtime.get("compile_mode", "matched_eager"))
    )
    _load_ema1(model, checkpoint_path)
    model = model.cuda().eval()
    labels = torch.tensor(labels_values, device="cuda", dtype=torch.long)
    noise = initial_noise(
        seeds,
        (3, arguments.img_size, arguments.img_size),
        device="cuda",
        dtype=torch.float32,
    )
    dtype_name = str(config["sampling"].get("dtype", "bfloat16"))
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    if autocast_dtype is None:
        raise ValueError("benchmark dtype must be bfloat16 or float16")

    def _run(mode: str) -> torch.Tensor:
        object.__setattr__(model, "_seacache_mode", mode)
        object.__setattr__(model.net, "_seacache_mode", mode)
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            result = model.generate(
                labels,
                noise=noise,
                sample_ids=sample_ids,
                trajectory_id=f"latency-{mode}",
            )
            normalized = torch.clamp((result + 1.0) / 2.0, 0.0, 1.0)
            return torch.round(normalized * 255.0).to(torch.uint8)

    def full() -> torch.Tensor:
        return _run("full")

    def candidate() -> torch.Tensor:
        return _run("seacache")

    expected = expected_model_calls_per_stream(
        arguments.num_sampling_steps, arguments.sampling_method
    )

    def full_summary() -> dict[str, Any]:
        return {
            "total_model_calls": 2 * expected,
            "full_calls": 2 * expected,
            "reuse_calls": 0,
            "full_body_ratio": 1.0,
            "gate_time_ms": 0.0,
            "fft_time_ms": 0.0,
            "cache_io_time_ms": 0.0,
        }

    def candidate_summary() -> dict[str, Any]:
        summaries = getattr(model, "_last_seacache_summaries", {})
        for value in summaries.values():
            finalize_timing_summary(value)
        full_calls = sum(int(value["full_calls"]) for value in summaries.values())
        reuse_calls = sum(int(value["reuse_calls"]) for value in summaries.values())
        total = full_calls + reuse_calls
        return {
            "total_model_calls": total,
            "full_calls": full_calls,
            "reuse_calls": reuse_calls,
            "full_body_ratio": full_calls / total if total else 0.0,
            "gate_time_ms": sum(float(value.get("gate_time_ms", 0.0)) for value in summaries.values()),
            "fft_time_ms": sum(float(value.get("fft_time_ms", 0.0)) for value in summaries.values()),
            "cache_io_time_ms": sum(float(value.get("cache_io_time_ms", 0.0)) for value in summaries.values()),
        }

    full.seacache_summary = full_summary  # type: ignore[attr-defined]
    candidate.seacache_summary = candidate_summary  # type: ignore[attr-defined]
    return BenchmarkSpec(
        full=full,
        seacache=candidate,
        batch_size=batch_size,
        effective_cfg_batch_size=batch_size,
        compile_mode=str(runtime.get("compile_mode", "matched_eager")),
        dtype=dtype_name,
        metadata={
            "model": arguments.model,
            "model_config_hash": canonical_hash(config["model"]),
            "checkpoint": str(checkpoint_path),
            "ema": "EMA1",
            "sampler": arguments.sampling_method,
            "sampler_config_hash": canonical_hash(config["sampling"]),
            "steps": arguments.num_sampling_steps,
            "cfg_scale": arguments.cfg,
            "guidance_interval": [arguments.interval_min, arguments.interval_max],
            "sample_ids": list(sample_ids),
            "threshold": threshold_value,
            "noise_scale": arguments.noise_scale,
            "cfg_execution": "separate cond then uncond forwards",
            "compile_wrappers_unwrapped": unwrapped_for_eager,
        },
    )


__all__ = ["build_benchmark_spec"]
