"""Deferred JiT factory for a matched instrumented-Full/SpeCa benchmark."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml

from .jit_denoiser import SpeCaDenoiser
from .latency import BenchmarkSpec
from .manifest import initial_noise
from .metadata import SPECA_CONFIG_FIELDS, canonical_hash, validate_speca_config
from .scheduler import expected_network_forward_count, expected_nfe_count


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
    candidates = (
        [path]
        if path.is_absolute()
        else [config_path.parent / path, PIXARC_ROOT / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint not found; checked {candidates}")


def _load_ema1(model: SpeCaDenoiser, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not all(key in checkpoint for key in ("model", "model_ema1")):
        raise KeyError("JiT checkpoint must contain model and model_ema1")
    incompatibility = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"strict checkpoint mismatch: {incompatibility}")
    ema = checkpoint["model_ema1"]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema:
                raise KeyError(f"EMA1 is missing {name}")
            parameter.copy_(ema[name])
    del checkpoint


def build_benchmark_spec(runner: Mapping[str, Any]) -> BenchmarkSpec:
    """Build matched instrumented-Full and released-code SpeCa closures."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("JiT latency requires exactly one visible CUDA GPU")
    config_path = Path(str(runner["model_config"])).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != "pixarc-speca-config-v1":
        raise ValueError("benchmark requires a pixarc-speca-config-v1 mapping")
    speca = validate_speca_config(dict(config["speca"]))
    if speca["mode"] != "speca":
        raise ValueError("matched benchmark candidate config must use mode=speca")
    sampling = dict(config["sampling"])
    if sampling.get("method") != "heun" or sampling.get("exact_heun") is not True:
        raise ValueError("primary JiT benchmark requires exact Heun")
    noise_scale = sampling.get("noise_scale", 1.0)
    if (
        isinstance(noise_scale, bool)
        or not isinstance(noise_scale, (int, float))
        or not math.isfinite(float(noise_scale))
        or noise_scale < 0
    ):
        raise ValueError("sampling.noise_scale must be finite and non-negative")
    runtime = dict(config["runtime"])
    batch_size = int(runner.get("batch_size", runtime["batch_size"]))
    if int(runtime["batch_size"]) != 32 or batch_size != 32:
        raise ValueError("JiT SpeCa latency requires real batch_size=32")
    sample_ids = tuple(int(value) for value in runner["sample_ids"])
    seeds = tuple(int(value) for value in runner["seeds"])
    labels_values = tuple(int(value) for value in runner["class_ids"])
    if not (len(sample_ids) == len(seeds) == len(labels_values) == batch_size):
        raise ValueError("sample_ids, seeds, class_ids, and batch_size must match")
    arguments = _args(config)
    checkpoint_value = Path(str(config["model"]["checkpoint"])).expanduser()
    origin = Path(str(runner.get("config_origin_dir", config_path.parent))).resolve()
    checkpoint_path = _checkpoint(
        str(checkpoint_value if checkpoint_value.is_absolute() else origin / checkpoint_value),
        config_path,
    )
    compile_mode = str(runtime.get("compile_mode", "matched_eager"))
    if compile_mode not in {"matched_eager", "blockwise"}:
        raise ValueError("matched speedup requires matched_eager or blockwise")
    model = SpeCaDenoiser(
        arguments,
        mode="speca",
        max_order=int(speca["max_order"]),
        base_threshold=float(speca["base_threshold"]),
        decay_rate=float(speca["decay_rate"]),
        min_taylor_steps=int(speca["min_taylor_steps"]),
        max_taylor_steps=int(speca["max_taylor_steps"]),
        first_enhance=int(speca["first_enhance"]),
        threshold_floor=float(speca["threshold_floor"]),
        error_metric=str(speca["error_metric"]),
        error_eps=float(speca["error_eps"]),
        verify_layer=int(speca["verify_layer"]),
        verification_token_scope=str(speca["verification_token_scope"]),
        gate_mode=str(speca["gate_mode"]),
        coordinate_mode=str(speca["coordinate_mode"]),
        force_last_full=bool(speca["force_last_full"]),
        cache_dtype=str(speca["cache_dtype"]),
        trace_mode="summary",
        interval=speca["interval"],
        compile_mode=compile_mode,
    )
    unwrapped_for_eager = int(getattr(model.net, "compile_wrappers_unwrapped", 0))
    if compile_mode == "blockwise":
        model.net.compile()
    _load_ema1(model, checkpoint_path)
    model = model.cuda().eval()
    labels = torch.tensor(labels_values, device="cuda", dtype=torch.long)
    noise = initial_noise(
        seeds,
        (3, arguments.img_size, arguments.img_size),
        device="cuda",
        dtype=torch.float32,
    )
    dtype_name = str(sampling.get("dtype", "bfloat16"))
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    if autocast_dtype is None:
        raise ValueError("benchmark dtype must be bfloat16 or float16")

    def _run(mode: str) -> torch.Tensor:
        model.speca_runtime.mode = mode
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
        return _run("instrumented_full")

    def candidate() -> torch.Tensor:
        return _run("speca")

    expected_nfe = expected_nfe_count(
        arguments.sampling_method, arguments.num_sampling_steps, exact_heun=True
    )
    expected_forwards = expected_network_forward_count(
        model_family="jit",
        sampler=arguments.sampling_method,
        num_steps=arguments.num_sampling_steps,
        exact_heun=True,
    )

    def full_summary() -> dict[str, Any]:
        return {
            "total_nfe": expected_nfe,
            "network_forward_count": expected_forwards,
            "expected_network_forward_count": expected_forwards,
            "full_nfe": expected_nfe,
            "taylor_nfe": 0,
            "verified_taylor_nfe": 0,
            "verification_block_calls": 0,
            "full_ratio": 1.0,
        }

    def candidate_summary() -> dict[str, Any]:
        summary = getattr(model, "_last_speca_summary", None)
        if not isinstance(summary, dict):
            raise RuntimeError("SpeCa benchmark has no completed trajectory summary")
        if int(summary.get("network_forward_count", -1)) != expected_forwards:
            raise RuntimeError("SpeCa benchmark network-forward count mismatch")
        return dict(summary)

    full.speca_summary = full_summary  # type: ignore[attr-defined]
    candidate.speca_summary = candidate_summary  # type: ignore[attr-defined]
    return BenchmarkSpec(
        full=full,
        speca=candidate,
        batch_size=batch_size,
        effective_cfg_batch_size=2 * batch_size,
        compile_mode=compile_mode,
        dtype=dtype_name,
        metadata={
            "model": arguments.model,
            "model_config_hash": canonical_hash(config["model"]),
            "checkpoint": str(checkpoint_path),
            "ema": "EMA1",
            "sampler": arguments.sampling_method,
            "sampler_config_hash": canonical_hash(sampling),
            "steps": arguments.num_sampling_steps,
            "cfg_scale": arguments.cfg,
            "guidance_interval": [arguments.interval_min, arguments.interval_max],
            "sample_ids": list(sample_ids),
            "speca_config_hash": canonical_hash(speca),
            **{field: speca[field] for field in SPECA_CONFIG_FIELDS},
            "noise_scale": arguments.noise_scale,
            "cfg_execution": "separate cond then uncond forwards; one shared scheduler",
            "compile_wrappers_unwrapped": unwrapped_for_eager,
        },
    )


__all__ = ["build_benchmark_spec"]
