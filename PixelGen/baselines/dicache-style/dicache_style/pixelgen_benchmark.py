"""Deferred PixelGen factory for matched timing and real-model parity gates."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Type

import torch
import yaml
from torch import nn

from src.diffusion.base.sampling import BaseSampler
from src.models.autoencoder.base import BaseAE, fp2uint8

from .latency import BenchmarkSpec, SingleBenchmarkSpec
from .manifest import initial_noise, sha256_file
from .metadata import DICACHE_CONFIG_FIELDS, canonical_hash, validate_dicache_config
from .scheduler import expected_nfe_count


PIXARC_ROOT = Path(__file__).resolve().parents[4]


COMPILE_MATRIX_ROWS: dict[str, tuple[str, str, str]] = {
    # role: (dicache.mode, runtime.compile_mode, callable selector)
    "upstream_whole_model": ("upstream_full", "upstream", "full"),
    "matched_eager_full": ("instrumented_full", "matched_eager", "full"),
    "matched_eager_dicache": ("dicache", "matched_eager", "dicache"),
    "blockwise_full": ("instrumented_full", "blockwise", "full"),
    "blockwise_dicache": ("dicache", "blockwise", "dicache"),
}


def _instantiate(specification: Mapping[str, Any], base_class: Type[Any]) -> Any:
    # jsonargparse is a deferred PixelGen runtime dependency; keeping the
    # import here lets CPU-only config/artifact tests import this module.
    from jsonargparse import ArgumentParser

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
    specification: Mapping[str, Any],
    dicache: Mapping[str, Any],
    compile_mode: str,
    *,
    required_mode: str,
) -> dict[str, Any]:
    """Return one denoiser spec with the complete released-code DiCache protocol."""

    validate_dicache_config(dicache, require_resolved=True)
    if dicache.get("mode") != required_mode:
        raise ValueError(
            f"benchmark config must use dicache.mode={required_mode}, got "
            f"{dicache.get('mode')!r}"
        )
    result = dict(specification)
    expected_class = "dicache_style.pixelgen_model.DiCachePixelGenJiT"
    if result.get("class_path") != expected_class:
        raise ValueError(f"benchmark denoiser must be {expected_class}")
    init_args = dict(result.get("init_args", {}))
    if required_mode == "upstream_full":
        # The matrix oracle is the actual upstream class, not merely the
        # instrumented subclass with its runtime disabled.
        result["class_path"] = "src.models.transformer.JiT.JiT"
        result["init_args"] = {
            key: value
            for key, value in init_args.items()
            if not key.startswith("dicache_") and key != "compile_mode"
        }
        return result
    init_args.update(
        {
            f"dicache_{field}": dicache[field]
            for field in DICACHE_CONFIG_FIELDS
            if field not in {"warmup_semantics", "threshold_compare", "probe_token_scope", "dcta_order", "residual_anchor_count"}
        }
    )
    init_args.update({"dicache_mode": required_mode, "compile_mode": compile_mode})
    result["init_args"] = init_args
    return result


def validate_compile_role_config(
    config: Mapping[str, Any], role: str
) -> tuple[str, str, str]:
    """Fail closed if a matrix row is materialized on inconsistent surfaces."""

    if role not in COMPILE_MATRIX_ROWS:
        raise ValueError(
            f"benchmark_role must be one of {sorted(COMPILE_MATRIX_ROWS)}"
        )
    if config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError("unsupported PixelGen DiCache config schema")
    required_mode, required_compile, selector = COMPILE_MATRIX_ROWS[role]
    dicache = config.get("dicache")
    runtime = config.get("runtime")
    model = config.get("model")
    if not isinstance(dicache, Mapping) or not isinstance(runtime, Mapping) or not isinstance(model, Mapping):
        raise TypeError("compile config must contain dicache/runtime/model mappings")
    if runtime.get("batch_size") != 1 or runtime.get("effective_cfg_batch_size") != 2:
        raise ValueError("compile matrix requires real batch=1 and CFG batch=2")
    denoiser = model.get("denoiser")
    if not isinstance(denoiser, Mapping) or not isinstance(denoiser.get("init_args"), Mapping):
        raise TypeError("compile config model.denoiser.init_args must be a mapping")
    sampler = model.get("diffusion_sampler")
    if not isinstance(sampler, Mapping) or not isinstance(sampler.get("init_args"), Mapping):
        raise TypeError("compile config model.diffusion_sampler.init_args must be a mapping")
    if sampler.get("class_path") != "dicache_style.pixelgen_sampler.DiCacheHeunSamplerJiT":
        raise ValueError("compile matrix requires the DiCache PixelGen Heun sampler")
    sampler_args = sampler["init_args"]
    if sampler_args.get("exact_henu") is not True:
        raise ValueError("compile matrix requires exact_henu=true")
    num_steps = sampler_args.get("num_steps")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("compile matrix sampler num_steps must be a positive integer")
    observed = {
        "dicache.mode": dicache.get("mode"),
        "runtime.compile_mode": runtime.get("compile_mode"),
        "model.compile_mode": model.get("compile_mode"),
        "model.denoiser.init_args.dicache_mode": denoiser["init_args"].get(
            "dicache_mode"
        ),
        "model.denoiser.init_args.compile_mode": denoiser["init_args"].get(
            "compile_mode"
        ),
    }
    expected = {
        "dicache.mode": required_mode,
        "runtime.compile_mode": required_compile,
        "model.compile_mode": required_compile,
        "model.denoiser.init_args.dicache_mode": required_mode,
        "model.denoiser.init_args.compile_mode": required_compile,
    }
    mismatches = {
        key: {"observed": observed[key], "expected": value}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise ValueError(f"compile matrix config surfaces disagree: {mismatches}")
    validate_dicache_config(dicache, require_resolved=True)
    return required_mode, required_compile, selector


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


def _build_benchmark_spec(
    runner: Mapping[str, Any], *, required_config_mode: str
) -> BenchmarkSpec:
    """Construct one EMA model and two closures; explicitly CUDA-only."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PixelGen latency requires exactly one visible CUDA GPU")
    config_path = Path(str(runner["model_config"])).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError("unsupported PixelGen DiCache config schema")
    dicache = dict(config["dicache"])
    validate_dicache_config(dicache, require_resolved=True)
    if dicache.get("mode") != required_config_mode:
        raise ValueError(
            f"benchmark expected dicache.mode={required_config_mode}, got "
            f"{dicache.get('mode')!r}"
        )
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
    if compile_mode == "upstream" and required_config_mode != "upstream_full":
        raise ValueError(
            "upstream whole-model compile is valid only for upstream_full"
        )
    if compile_mode not in {"upstream", "matched_eager", "blockwise"}:
        raise ValueError(f"unsupported benchmark compile mode: {compile_mode!r}")
    model_config = dict(config["model"])
    denoiser_spec = _benchmark_denoiser_spec(
        model_config["denoiser"],
        dicache,
        compile_mode,
        required_mode=required_config_mode,
    )
    expected_sampler = "dicache_style.pixelgen_sampler.DiCacheHeunSamplerJiT"
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
    if compile_mode == "blockwise":
        # Upstream JiTBlock.forward is decorator-compiled.  Remove that wrapper
        # before nn.Module.compile wraps each block, avoiding nested Dynamo
        # wrappers in the explicit blockwise matrix row.
        from .pixelgen_model import configure_pixelgen_compile_mode

        object.__setattr__(
            net,
            "compile_wrappers_unwrapped",
            configure_pixelgen_compile_mode(net, "matched_eager"),
        )
    net.compile()
    unwrapped_for_eager = int(getattr(net, "compile_wrappers_unwrapped", 0))

    batch_size = int(runner.get("batch_size", runtime["batch_size"]))
    if int(runtime["batch_size"]) != 1 or int(runtime["effective_cfg_batch_size"]) != 2:
        raise ValueError("primary PixelGen protocol requires real batch=1 and CFG batch=2")
    if batch_size != 1:
        raise ValueError("PixelGen DiCache latency must use real batch_size=1")
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

    def _sample(mode: str, *, trajectory_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        runtime_object = getattr(net, "dicache_runtime", None)
        if runtime_object is not None:
            runtime_object.mode = mode
        sampler.set_dicache_batch_context(
            sample_ids=sample_ids,
            trajectory_id=trajectory_id,
        )
        autocast_context = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            samples = sampler(net, noise, condition, uncondition)
            decoded = vae.decode(samples)
        return samples, decoded

    def _run(mode: str) -> torch.Tensor:
        _samples, decoded = _sample(
            mode, trajectory_id=f"pixelgen-latency-{mode}"
        )
        return fp2uint8(decoded)

    def _parity_run(mode: str) -> dict[str, Any]:
        """Capture one complete exact trajectory without changing model code.

        Forward hooks observe every block, the probe-depth feature, and the
        fresh final head.  The temporary ``complete_full`` observer verifies
        the actual cached residual after the runtime writes each exact anchor.
        It is restored even when sampling fails.
        """

        if mode not in {"upstream_full", "probe_only_ablation"}:
            raise ValueError(f"unsupported parity mode: {mode}")
        depth = len(net.blocks)
        probe_block_index = int(net.dicache_runtime.probe_depth) - 1
        image_tokens = int(net.x_embedder.num_patches)
        context_tokens = int(net.in_context_len)
        context_start = int(net.in_context_start)
        expected_order = tuple(range(depth))
        block_call_counts = [0 for _ in range(depth)]
        block_token_layout_valid = True
        rope_valid = True
        block_order_valid = True
        current_order: list[int] = []
        completed_block_orders = 0
        gradient_enabled_observations: list[bool] = []
        body_outputs: list[torch.Tensor] = []
        probe_features: list[torch.Tensor] = []
        final_head_calls = 0
        final_head_image_tokens_valid = True
        anchor_checks: list[bool] = []
        probe_anchor_checks: list[bool] = []
        distinguishable_anchor_checks = 0
        wrong_probe_baseline_matches = 0
        hooks: list[Any] = []

        def block_pre_hook(_module, arguments, *, index: int) -> None:
            nonlocal block_token_layout_valid, rope_valid
            if len(arguments) < 3 or not torch.is_tensor(arguments[0]):
                raise RuntimeError("PixelGen block hook did not receive (x, condition, RoPE)")
            tokens = int(arguments[0].shape[1])
            expected_tokens = image_tokens + (
                context_tokens if index >= context_start else 0
            )
            block_token_layout_valid = (
                block_token_layout_valid and tokens == expected_tokens
            )
            expected_rope = (
                net.feat_rope if index < context_start else net.feat_rope_incontext
            )
            rope_valid = rope_valid and arguments[2] is expected_rope
            block_call_counts[index] += 1
            current_order.append(index)
            gradient_enabled_observations.append(torch.is_grad_enabled())

        def probe_output_hook(_module, _arguments, output) -> None:
            if not torch.is_tensor(output):
                raise RuntimeError("PixelGen probe block hook expected a tensor")
            feature = output
            if probe_block_index >= context_start and context_tokens:
                feature = feature[:, context_tokens:, :]
            if int(feature.shape[1]) != image_tokens:
                raise RuntimeError("probe hook extracted the wrong image-token span")
            probe_features.append(feature.detach().cpu().clone())

        def final_pre_hook(_module, arguments) -> None:
            nonlocal block_order_valid, completed_block_orders
            nonlocal final_head_image_tokens_valid
            if not arguments or not torch.is_tensor(arguments[0]):
                raise RuntimeError("PixelGen final-layer hook expected a body tensor")
            block_order_valid = block_order_valid and tuple(current_order) == expected_order
            current_order.clear()
            completed_block_orders += 1
            body = arguments[0]
            final_head_image_tokens_valid = (
                final_head_image_tokens_valid and int(body.shape[1]) == image_tokens
            )
            body_outputs.append(body.detach().cpu().clone())
            gradient_enabled_observations.append(torch.is_grad_enabled())

        def final_post_hook(_module, _arguments, _output) -> None:
            nonlocal final_head_calls
            final_head_calls += 1

        for index, block in enumerate(net.blocks):
            hooks.append(
                block.register_forward_pre_hook(
                    lambda module, arguments, index=index: block_pre_hook(
                        module, arguments, index=index
                    )
                )
            )
        hooks.append(net.blocks[probe_block_index].register_forward_hook(probe_output_hook))
        hooks.append(net.final_layer.register_forward_pre_hook(final_pre_hook))
        hooks.append(net.final_layer.register_forward_hook(final_post_hook))

        runtime_object = net.dicache_runtime
        original_complete_full = runtime_object.complete_full
        had_instance_complete_full = "complete_full" in runtime_object.__dict__
        previous_instance_complete_full = runtime_object.__dict__.get("complete_full")

        def observed_complete_full(**kwargs) -> None:
            nonlocal distinguishable_anchor_checks, wrong_probe_baseline_matches
            original_complete_full(**kwargs)
            trajectory = runtime_object.trajectory
            if trajectory is None:
                raise RuntimeError("runtime reset before exact anchor observation")
            state = trajectory.streams[kwargs["plan"].stream_id]
            anchor = state.anchors.latest
            exact_body = kwargs["exact_body_output"]
            body_input = kwargs["body_input"]
            probe_feature = kwargs["probe_feature"]
            expected_full = (exact_body - body_input).to(anchor.full_residual.dtype)
            expected_probe = (probe_feature - body_input).to(anchor.probe_residual.dtype)
            wrong_full = (exact_body - probe_feature).to(anchor.full_residual.dtype)
            anchor_checks.append(torch.equal(anchor.full_residual, expected_full))
            probe_anchor_checks.append(torch.equal(anchor.probe_residual, expected_probe))
            if not torch.equal(expected_full, wrong_full):
                distinguishable_anchor_checks += 1
                wrong_probe_baseline_matches += int(
                    torch.equal(anchor.full_residual, wrong_full)
                )

        object.__setattr__(runtime_object, "complete_full", observed_complete_full)
        grad_enabled_before = torch.is_grad_enabled()
        try:
            samples, decoded = _sample(
                mode, trajectory_id=f"pixelgen-parity-{mode}"
            )
            summary = sampler.last_dicache_summary
            if not isinstance(summary, dict):
                raise RuntimeError(f"{mode} parity run has no completed summary")
        finally:
            for hook in hooks:
                hook.remove()
            if had_instance_complete_full:
                object.__setattr__(
                    runtime_object,
                    "complete_full",
                    previous_instance_complete_full,
                )
            else:
                delattr(runtime_object, "complete_full")
        grad_enabled_after = torch.is_grad_enabled()
        return {
            "body_outputs": body_outputs,
            "probe_features": probe_features,
            "sample": samples.detach().cpu().clone(),
            "decoded_image": decoded.detach().cpu().clone(),
            "summary": dict(summary),
            "runtime_active_after": bool(runtime_object.active),
            "cache_bytes_after": int(runtime_object.cache_bytes()),
            "cache_tensor_count_after": int(runtime_object.tensor_count()),
            "diagnostics": {
                "depth": depth,
                "probe_depth": probe_block_index + 1,
                "image_token_count": image_tokens,
                "context_token_count": context_tokens,
                "context_start": context_start,
                "block_call_counts": block_call_counts,
                "block_token_layout_valid": block_token_layout_valid,
                "correct_rope_for_every_block": rope_valid,
                "per_forward_block_order_valid": block_order_valid,
                "completed_block_orders": completed_block_orders,
                "final_head_calls": final_head_calls,
                "final_head_image_tokens_valid": final_head_image_tokens_valid,
                "gradient_disabled_during_capture": not any(
                    gradient_enabled_observations
                ),
                "gradient_state_restored": grad_enabled_before == grad_enabled_after,
                "anchor_check_count": len(anchor_checks),
                "all_full_anchors_use_current_body_input": all(anchor_checks),
                "all_probe_anchors_use_current_body_input": all(probe_anchor_checks),
                "distinguishable_anchor_check_count": distinguishable_anchor_checks,
                "wrong_probe_baseline_match_count": wrong_probe_baseline_matches,
            },
        }

    full_runtime_mode = (
        "upstream_full"
        if required_config_mode == "upstream_full"
        else "instrumented_full"
    )

    def full() -> torch.Tensor:
        return _run(full_runtime_mode)

    def candidate() -> torch.Tensor:
        return _run("dicache")

    def _raw_finiteness(mode: str) -> dict[str, Any]:
        """Run one real row and inspect floats before uint8 conversion."""

        samples, decoded = _sample(
            mode, trajectory_id=f"pixelgen-compile-{mode}-finiteness"
        )
        summary = sampler.last_dicache_summary
        if not isinstance(summary, dict):
            raise RuntimeError(f"{mode} raw-finiteness run has no completed summary")
        runtime_object = getattr(net, "dicache_runtime", None)
        return {
            "sample_finite": bool(torch.isfinite(samples).all().item()),
            "decoded_image_finite": bool(torch.isfinite(decoded).all().item()),
            "summary": dict(summary),
            "runtime_active_after": (
                bool(runtime_object.active) if runtime_object is not None else False
            ),
            "cache_bytes_after": (
                int(runtime_object.cache_bytes()) if runtime_object is not None else 0
            ),
            "cache_tensor_count_after": (
                int(runtime_object.tensor_count()) if runtime_object is not None else 0
            ),
        }

    expected = expected_nfe_count(
        "heun", int(sampler.num_steps), exact_heun=bool(sampler.exact_henu)
    )

    def full_summary() -> dict[str, Any]:
        summary = sampler.last_dicache_summary
        if not isinstance(summary, dict) or summary.get("mode") != full_runtime_mode:
            raise RuntimeError(
                f"{full_runtime_mode} benchmark has no completed summary"
            )
        if int(summary.get("network_forward_count", -1)) != expected:
            raise RuntimeError("instrumented-Full network-forward count mismatch")
        return dict(summary)

    def candidate_summary() -> dict[str, Any]:
        summary = sampler.last_dicache_summary
        if not isinstance(summary, dict):
            raise RuntimeError("DiCache benchmark has no completed trajectory summary")
        if int(summary.get("network_forward_count", -1)) != expected:
            raise RuntimeError("DiCache network-forward count mismatch")
        return dict(summary)

    full.dicache_summary = full_summary  # type: ignore[attr-defined]
    candidate.dicache_summary = candidate_summary  # type: ignore[attr-defined]
    full.upstream_body_and_image = lambda: _parity_run("upstream_full")  # type: ignore[attr-defined]
    full.probe_only_body_and_image = lambda: _parity_run("probe_only_ablation")  # type: ignore[attr-defined]
    full.raw_finiteness = lambda: _raw_finiteness(full_runtime_mode)  # type: ignore[attr-defined]
    candidate.raw_finiteness = lambda: _raw_finiteness("dicache")  # type: ignore[attr-defined]
    sampler_args = dict(model_config["diffusion_sampler"]["init_args"])
    return BenchmarkSpec(
        full=full,
        dicache=candidate,
        batch_size=batch_size,
        effective_cfg_batch_size=2 * batch_size,
        compile_mode=compile_mode,
        dtype=precision,
        metadata={
            "model": "PixelGen-JiT",
            "input_config": str(config_path),
            "input_config_sha256": sha256_file(config_path),
            "input_config_hash": canonical_hash(config),
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
                            if not key.startswith("dicache_") and key != "compile_mode"
                        },
                    },
                }
            ),
            "checkpoint": str(checkpoint_path),
            "checkpoint_size": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
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
            "seeds": list(seeds),
            "class_ids": list(class_ids),
            "manifest": str(Path(str(runner["manifest"])).resolve(strict=True)),
            "manifest_sha256": sha256_file(runner["manifest"]),
            "dicache": {field: dicache[field] for field in DICACHE_CONFIG_FIELDS},
            "dicache_config_hash": canonical_hash(dicache),
            "noise_scale": noise_scale,
            "expected_network_forward_count": expected,
            "cfg_execution": "single combined [unconditional, conditional] 2B forward",
            "compile_wrappers_unwrapped": unwrapped_for_eager,
            "outer_compile_enabled": compile_mode == "upstream",
            "config_mode": required_config_mode,
            "denoiser_execution_class": denoiser_spec["class_path"],
        },
    )


def build_benchmark_spec(runner: Mapping[str, Any]) -> BenchmarkSpec:
    """Construct the preserved matched Full/DiCache benchmark pair API."""

    spec = _build_benchmark_spec(runner, required_config_mode="dicache")
    if spec.compile_mode == "upstream":
        raise ValueError(
            "matched speedup cannot use upstream outer compile; use "
            "matched_eager or blockwise"
        )
    return spec


def build_compile_benchmark_spec(runner: Mapping[str, Any]) -> SingleBenchmarkSpec:
    """Construct one isolated row of the deferred five-row compile matrix."""

    role = str(runner.get("benchmark_role", ""))
    config_path = Path(str(runner["model_config"])).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise TypeError("compile matrix config must be a YAML mapping")
    required_mode, required_compile, selector = validate_compile_role_config(
        config, role
    )
    pair = _build_benchmark_spec(runner, required_config_mode=required_mode)
    if pair.compile_mode != required_compile:
        raise AssertionError("validated compile mode changed during construction")
    function = pair.full if selector == "full" else pair.dicache
    validation = getattr(function, "raw_finiteness", None)
    if not callable(validation):
        raise TypeError("compile row callable lacks raw-finiteness validation")
    metadata = {
        **dict(pair.metadata),
        "benchmark_role": role,
        "correctness_output": "decoded RGB uint8 tensor before CPU copy/PNG",
    }
    return SingleBenchmarkSpec(
        function=function,
        validation=validation,
        role=role,
        batch_size=pair.batch_size,
        effective_cfg_batch_size=pair.effective_cfg_batch_size,
        compile_mode=pair.compile_mode,
        dtype=pair.dtype,
        metadata=metadata,
    )


__all__ = [
    "COMPILE_MATRIX_ROWS",
    "build_benchmark_spec",
    "build_compile_benchmark_spec",
    "validate_compile_role_config",
]
