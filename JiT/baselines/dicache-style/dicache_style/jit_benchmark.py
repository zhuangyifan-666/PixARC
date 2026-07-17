"""Deferred JiT factory for the matched CUDA-event benchmark harness."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml

from .jit_denoiser import DiCacheDenoiser
from .jit_model import configure_jit_compile_mode
from .latency import BenchmarkSpec
from .manifest import initial_noise, load_manifest, sha256_file
from .metadata import canonical_hash, validate_dicache_config


PIXARC_ROOT = Path(__file__).resolve().parents[4]


def _args(config: Mapping[str, Any]) -> SimpleNamespace:
    model = dict(config["model"])
    sampling = dict(config["sampling"])
    extra = dict(model.get("args", {}))
    low, high = sampling.get("guidance_interval", [0.1, 1.0])
    return SimpleNamespace(
        model=str(model["variant"]),
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


def _load_ema1(model: DiCacheDenoiser, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not all(key in checkpoint for key in ("model", "model_ema1")):
        raise KeyError("JiT checkpoint must contain model and model_ema1")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema = checkpoint["model_ema1"]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema:
                raise KeyError(f"EMA1 is missing parameter {name}")
            parameter.copy_(ema[name])
    del checkpoint


def validate_runner_config(
    runner: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Pure validation used before the deferred factory allocates a model."""

    if config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError("benchmark config has the wrong schema_version")
    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("ema") != "model_ema1":
        raise ValueError("JiT benchmark requires model.ema=model_ema1")
    dicache = validate_dicache_config(dict(config["dicache"]), require_resolved=True)
    if dicache["mode"] != "dicache":
        raise ValueError("benchmark model_config must be a resolved dicache config")
    sampling = dict(config["sampling"])
    if sampling.get("method") != "heun" or sampling.get("exact_heun") is not True:
        raise ValueError("JiT benchmark requires exact Heun")
    noise_scale = sampling.get("noise_scale", 1.0)
    if (
        isinstance(noise_scale, bool)
        or not isinstance(noise_scale, (int, float))
        or not math.isfinite(float(noise_scale))
        or float(noise_scale) < 0
    ):
        raise ValueError("sampling.noise_scale must be finite and non-negative")
    runtime = dict(config["runtime"])
    compile_mode = str(
        runner.get(
            "compile_mode_override",
            runtime.get("compile_mode", "matched_eager"),
        )
    )
    full_mode = str(runner.get("full_mode_override", "instrumented_full"))
    if compile_mode not in {"upstream", "matched_eager", "blockwise"}:
        raise ValueError("unsupported benchmark compile_mode")
    if compile_mode == "upstream" and full_mode != "upstream_full":
        raise ValueError("whole-model upstream compile requires upstream_full")
    if compile_mode != "upstream" and full_mode != "instrumented_full":
        raise ValueError("adaptive compile modes require instrumented_full")
    runtime["compile_mode"] = compile_mode
    runtime["benchmark_full_mode"] = full_mode
    batch_size = int(runner.get("batch_size", runtime["batch_size"]))
    purpose = str(runner.get("purpose", "latency"))
    if int(runtime["batch_size"]) != 32:
        raise ValueError("JiT primary protocol requires runtime.batch_size=32")
    if purpose == "latency":
        if batch_size != 32:
            raise ValueError("JiT DiCache latency requires real batch_size=32")
    elif purpose == "model_parity":
        if batch_size != 1:
            raise ValueError("JiT model parity requires one immutable sample")
    else:
        raise ValueError(f"unsupported JiT benchmark purpose: {purpose}")
    sample_ids = tuple(int(value) for value in runner["sample_ids"])
    seeds = tuple(int(value) for value in runner["seeds"])
    class_ids = tuple(int(value) for value in runner["class_ids"])
    if batch_size <= 0 or not (
        len(sample_ids) == len(seeds) == len(class_ids) == batch_size
    ):
        raise ValueError("sample_ids, seeds, class_ids, and batch_size must match")
    manifest_value = runner.get("manifest")
    group_id = runner.get("batch_group_id")
    if not isinstance(manifest_value, str) or not isinstance(group_id, str):
        raise ValueError("benchmark runner requires manifest and batch_group_id")
    records = load_manifest(Path(manifest_value).resolve(strict=True))
    if group_id != records[0].batch_group_id:
        raise ValueError("benchmark runner must use the first immutable manifest group")
    group = sorted(
        (record for record in records if record.batch_group_id == group_id),
        key=lambda record: record.position_in_batch,
    )
    expected_group = group if purpose == "latency" else group[:1]
    expected = (
        tuple(record.sample_id for record in expected_group),
        tuple(record.seed for record in expected_group),
        tuple(record.class_id for record in expected_group),
    )
    if len(expected_group) != batch_size or (sample_ids, seeds, class_ids) != expected:
        raise ValueError("benchmark runner IDs differ from its immutable manifest group")
    return dicache, runtime, batch_size, sample_ids, seeds, class_ids


def build_benchmark_spec(runner: Mapping[str, Any]) -> BenchmarkSpec:
    """Build matched instrumented-Full/DiCache closures on one visible GPU."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("JiT latency requires exactly one visible CUDA GPU")
    config_path = Path(str(runner["model_config"])).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("benchmark model_config must be a YAML mapping")
    dicache, runtime, batch_size, sample_ids, seeds, class_ids = validate_runner_config(
        runner, config
    )
    arguments = _args(config)
    checkpoint_path = _checkpoint(str(config["model"]["checkpoint"]), config_path)
    manifest_path = Path(str(runner["manifest"])).resolve(strict=True)
    compile_mode = str(runtime.get("compile_mode", "matched_eager"))
    full_mode = str(runtime["benchmark_full_mode"])
    initial_mode = "upstream_full" if compile_mode == "upstream" else "dicache"
    model = DiCacheDenoiser(
        arguments,
        mode=initial_mode,
        profile=str(dicache["profile"]),
        probe_depth=int(dicache["probe_depth"]),
        error_choice=str(dicache["error_choice"]),
        rel_l1_thresh=float(dicache["rel_l1_thresh"]),
        ret_ratio=float(dicache["ret_ratio"]),
        gamma_min=float(dicache["gamma_min"]),
        gamma_max=float(dicache["gamma_max"]),
        force_last_full=bool(dicache["force_last_full"]),
        numeric_mode=str(dicache["numeric_mode"]),
        epsilon=float(dicache["epsilon"]),
        nonfinite_policy=str(dicache["nonfinite_policy"]),
        gamma_nonfinite_policy=str(dicache["gamma_nonfinite_policy"]),
        gate_mode=str(dicache["gate_mode"]),
        cache_dtype=str(dicache["cache_dtype"]),
        trace_mode="summary",
        compile_mode=compile_mode,
        warmup_semantics=str(dicache["warmup_semantics"]),
    )
    _load_ema1(model, checkpoint_path)
    model = model.cuda().eval()
    if compile_mode == "blockwise":
        # The pinned upstream source decorates every block/final-layer forward
        # with torch.compile.  Remove those decorators before applying explicit
        # nn.Module.compile wrappers so this matrix row is not double compiled.
        changed = configure_jit_compile_mode(model.net, "matched_eager")
        object.__setattr__(
            model.net,
            "compile_wrappers_unwrapped",
            int(getattr(model.net, "compile_wrappers_unwrapped", 0)) + changed,
        )
    if compile_mode in {"upstream", "blockwise"}:
        model.net.compile()
    compile_wrappers_unwrapped = int(
        getattr(model.net, "compile_wrappers_unwrapped", 0)
    )
    labels = torch.tensor(class_ids, device="cuda", dtype=torch.long)
    noise = initial_noise(
        seeds,
        (3, arguments.img_size, arguments.img_size),
        device="cpu",
        dtype=torch.float32,
    ).to(device="cuda", non_blocking=False)
    dtype_name = str(config["sampling"].get("dtype", "bfloat16"))
    autocast_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(dtype_name)
    if autocast_dtype is None:
        raise ValueError("benchmark dtype must be bfloat16 or float16")

    def _run_raw(mode: str) -> torch.Tensor:
        # Every generate() closes and releases its trajectory, so changing only
        # this non-parameter execution mode cannot leak cache state across runs.
        model.dicache_runtime.mode = mode
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            result = model.generate(
                labels,
                noise=noise,
                sample_ids=sample_ids,
                trajectory_id=f"latency-{mode}",
            )
            return result

    def _run(mode: str) -> torch.Tensor:
        result = _run_raw(mode)
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError(f"non-finite JiT float output in {mode}")
        normalized = torch.clamp((result + 1.0) / 2.0, 0.0, 1.0)
        return torch.round(normalized * 255.0).to(torch.uint8)

    def full() -> torch.Tensor:
        return _run(full_mode)

    def candidate() -> torch.Tensor:
        return _run("dicache")

    def summary() -> Mapping[str, Any]:
        value = getattr(model, "_last_dicache_summary", None)
        if not isinstance(value, Mapping):
            raise RuntimeError("benchmark run did not expose a DiCache summary")
        return value

    def parity_run(mode: str) -> dict[str, Any]:
        """Capture every exact block/body/head event for one full trajectory."""

        if mode not in {"upstream_full", "probe_only_ablation"}:
            raise ValueError(f"unsupported JiT parity mode: {mode}")
        net = model.net
        runtime_object = model.dicache_runtime
        depth = len(net.blocks)
        probe_block_index = int(runtime_object.probe_depth) - 1
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
        inference_mode_observations: list[bool] = []
        body_outputs: list[torch.Tensor] = []
        probe_features: list[torch.Tensor] = []
        final_head_calls = 0
        final_head_image_tokens_valid = True
        anchor_checks: list[bool] = []
        probe_anchor_checks: list[bool] = []
        distinguishable_anchor_checks = 0
        wrong_probe_baseline_matches = 0
        anchor_stream_counts: dict[str, int] = {}
        hooks: list[Any] = []

        def block_pre_hook(_module, arguments, *, index: int) -> None:
            nonlocal block_token_layout_valid, rope_valid
            if len(arguments) < 3 or not torch.is_tensor(arguments[0]):
                raise RuntimeError("JiT block hook did not receive (x, condition, RoPE)")
            expected_tokens = image_tokens + (
                context_tokens if index >= context_start else 0
            )
            block_token_layout_valid = (
                block_token_layout_valid
                and int(arguments[0].shape[1]) == expected_tokens
            )
            expected_rope = (
                net.feat_rope if index < context_start else net.feat_rope_incontext
            )
            rope_valid = rope_valid and arguments[2] is expected_rope
            block_call_counts[index] += 1
            current_order.append(index)
            gradient_enabled_observations.append(torch.is_grad_enabled())
            inference_mode_observations.append(torch.is_inference_mode_enabled())

        def probe_output_hook(_module, _arguments, output) -> None:
            if not torch.is_tensor(output):
                raise RuntimeError("JiT probe block hook expected a tensor")
            feature = output
            if probe_block_index >= context_start and context_tokens:
                feature = feature[:, context_tokens:, :]
            if int(feature.shape[1]) != image_tokens:
                raise RuntimeError("JiT probe hook extracted the wrong image-token span")
            probe_features.append(feature.detach().cpu().clone())

        def final_pre_hook(_module, arguments) -> None:
            nonlocal block_order_valid, completed_block_orders
            nonlocal final_head_image_tokens_valid
            if not arguments or not torch.is_tensor(arguments[0]):
                raise RuntimeError("JiT final-layer hook expected a body tensor")
            block_order_valid = block_order_valid and tuple(current_order) == expected_order
            current_order.clear()
            completed_block_orders += 1
            body = arguments[0]
            final_head_image_tokens_valid = (
                final_head_image_tokens_valid and int(body.shape[1]) == image_tokens
            )
            body_outputs.append(body.detach().cpu().clone())
            gradient_enabled_observations.append(torch.is_grad_enabled())
            inference_mode_observations.append(torch.is_inference_mode_enabled())

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

        original_complete_full = runtime_object.complete_full
        had_instance_complete_full = "complete_full" in runtime_object.__dict__
        previous_instance_complete_full = runtime_object.__dict__.get("complete_full")

        def observed_complete_full(**kwargs) -> None:
            nonlocal distinguishable_anchor_checks, wrong_probe_baseline_matches
            original_complete_full(**kwargs)
            trajectory = runtime_object.trajectory
            if trajectory is None:
                raise RuntimeError("runtime reset before exact anchor observation")
            stream_id = kwargs["plan"].stream_id
            state = trajectory.streams[stream_id]
            anchor = state.anchors.latest
            exact_body = kwargs["exact_body_output"]
            body_input = kwargs["body_input"]
            probe_feature = kwargs["probe_feature"]
            full_dtype = anchor.full_residual.dtype
            probe_dtype = anchor.probe_residual.dtype
            expected_full = exact_body.to(full_dtype) - body_input.to(full_dtype)
            expected_probe = (
                probe_feature.to(probe_dtype) - body_input.to(probe_dtype)
            )
            wrong_full = exact_body.to(full_dtype) - probe_feature.to(full_dtype)
            anchor_checks.append(torch.equal(anchor.full_residual, expected_full))
            probe_anchor_checks.append(torch.equal(anchor.probe_residual, expected_probe))
            stream_key = str(stream_id)
            anchor_stream_counts[stream_key] = anchor_stream_counts.get(stream_key, 0) + 1
            if not torch.equal(expected_full, wrong_full):
                distinguishable_anchor_checks += 1
                wrong_probe_baseline_matches += int(
                    torch.equal(anchor.full_residual, wrong_full)
                )

        object.__setattr__(runtime_object, "complete_full", observed_complete_full)
        grad_enabled_before = torch.is_grad_enabled()
        inference_mode_before = torch.is_inference_mode_enabled()
        try:
            sample = _run_raw(mode)
            runtime_summary = dict(summary())
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
        inference_mode_after = torch.is_inference_mode_enabled()
        return {
            "body_outputs": body_outputs,
            "probe_features": probe_features,
            "sample": sample.detach().cpu().clone(),
            "summary": runtime_summary,
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
                "inference_mode_enabled_during_capture": all(
                    inference_mode_observations
                ),
                "inference_state_restored": (
                    inference_mode_before == inference_mode_after
                ),
                "anchor_check_count": len(anchor_checks),
                "anchor_stream_counts": anchor_stream_counts,
                "all_full_anchors_use_current_body_input": all(anchor_checks),
                "all_probe_anchors_use_current_body_input": all(probe_anchor_checks),
                "distinguishable_anchor_check_count": distinguishable_anchor_checks,
                "wrong_probe_baseline_match_count": wrong_probe_baseline_matches,
            },
        }

    def candidate_raw_finiteness() -> dict[str, Any]:
        sample = _run_raw("dicache")
        runtime_summary = dict(summary())
        runtime_object = model.dicache_runtime
        return {
            "sample_finite": bool(torch.isfinite(sample).all().item()),
            "summary": runtime_summary,
            "runtime_active_after": bool(runtime_object.active),
            "cache_bytes_after": int(runtime_object.cache_bytes()),
            "cache_tensor_count_after": int(runtime_object.tensor_count()),
        }

    def full_raw() -> torch.Tensor:
        return _run_raw(full_mode)

    def candidate_raw() -> torch.Tensor:
        if compile_mode == "upstream":
            raise RuntimeError("upstream whole-model compile has no adaptive candidate")
        return _run_raw("dicache")

    def runtime_state() -> dict[str, Any]:
        runtime_object = model.dicache_runtime
        return {
            "runtime_active_after": bool(runtime_object.active),
            "cache_bytes_after": int(runtime_object.cache_bytes()),
            "cache_tensor_count_after": int(runtime_object.tensor_count()),
        }

    full.dicache_summary = summary  # type: ignore[attr-defined]
    candidate.dicache_summary = summary  # type: ignore[attr-defined]
    full.upstream_raw = lambda: _run_raw("upstream_full")  # type: ignore[attr-defined]
    full.instrumented_raw = lambda: _run_raw("instrumented_full")  # type: ignore[attr-defined]
    full.upstream_body_and_image = lambda: parity_run("upstream_full")  # type: ignore[attr-defined]
    full.probe_only_body_and_image = lambda: parity_run("probe_only_ablation")  # type: ignore[attr-defined]
    candidate.raw_finiteness = candidate_raw_finiteness  # type: ignore[attr-defined]
    full.raw = full_raw  # type: ignore[attr-defined]
    full_raw.dicache_summary = summary  # type: ignore[attr-defined]
    full_raw.runtime_state = runtime_state  # type: ignore[attr-defined]
    candidate.raw = candidate_raw  # type: ignore[attr-defined]
    candidate_raw.dicache_summary = summary  # type: ignore[attr-defined]
    candidate_raw.runtime_state = runtime_state  # type: ignore[attr-defined]
    return BenchmarkSpec(
        full=full,
        dicache=candidate,
        batch_size=batch_size,
        effective_cfg_batch_size=2 * batch_size,
        compile_mode=compile_mode,
        dtype=dtype_name,
        metadata={
            "model": arguments.model,
            "input_config": str(config_path),
            "input_config_hash": canonical_hash(config),
            "model_config_hash": canonical_hash(config["model"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_size": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "ema": "EMA1",
            "sampler": arguments.sampling_method,
            "exact_heun": bool(config["sampling"].get("exact_heun", False)),
            "sampler_config_hash": canonical_hash(config["sampling"]),
            "dicache_config_hash": canonical_hash(dicache),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "steps": arguments.num_sampling_steps,
            "cfg_scale": arguments.cfg,
            "guidance_interval": [arguments.interval_min, arguments.interval_max],
            "sample_ids": list(sample_ids),
            "seeds": list(seeds),
            "class_ids": list(class_ids),
            "rel_l1_thresh": float(dicache["rel_l1_thresh"]),
            "noise_scale": arguments.noise_scale,
            "initial_noise_protocol": (
                "per-sample CPU torch.Generator float32, then copied to GPU"
            ),
            "cfg_execution": "separate cond then uncond forwards with isolated states",
            "full_mode": full_mode,
            "candidate_mode": "dicache",
            "compile_scope": (
                "whole_model" if compile_mode == "upstream" else (
                    "blocks_and_final_layer" if compile_mode == "blockwise" else "eager"
                )
            ),
            "compile_wrappers_unwrapped": compile_wrappers_unwrapped,
        },
    )


__all__ = ["build_benchmark_spec", "validate_runner_config"]
