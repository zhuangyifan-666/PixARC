#!/usr/bin/env python3
"""Generate one deterministic JiT manifest shard (GPU execution is deferred)."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from speca_style.image_io import (
    atomic_write_png,
    image_path,
    load_metadata_jsonl,
    resumable_batch_groups,
)
from speca_style.jit_denoiser import SpeCaDenoiser
from speca_style.manifest import (
    initial_noise,
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
    validate_manifest_sidecar,
)
from speca_style.metadata import (
    SPECA_CONFIG_FIELDS,
    atomic_create_json,
    atomic_write_json,
    build_run_metadata,
    canonical_hash,
    checkpoint_identity,
    git_revision,
    load_json,
    source_tree_sha256,
    validate_speca_config,
)


PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = Path(__file__).resolve().parents[1]
CACHE4DIFFUSION_COMMIT = "91a1949fcc88acab46547f0b5f295f5de2df2870"
TAYLORSEER_COMMIT = "704ee98c74f7f04da443daa3c0aa2cc7803d86e3"

# These are stored once per trajectory (once per row for the main batch-1
# protocol).  They are deliberately scalar: 50K runs never persist features.
_SUMMARY_INT_FIELDS = (
    "total_nfe",
    "full_nfe",
    "taylor_nfe",
    "verified_taylor_nfe",
    "unverified_taylor_nfe",
    "verification_pass_count",
    "verification_fail_count",
    "full_due_first_enhance",
    "full_due_max_taylor",
    "full_due_previous_error",
    "verification_block_calls",
    "network_forward_count",
    "expected_network_forward_count",
    "cache_bytes",
    "cache_allocated_bytes",
    "cache_tensor_count",
    "peak_memory_allocated",
    "peak_memory_reserved",
)
_SUMMARY_FLOAT_FIELDS = (
    "full_ratio",
    "taylor_ratio",
    "verification_ratio",
    "verification_ratio_among_taylor",
    "verification_pass_rate",
    "verification_fail_rate",
    "mean_speculative_span",
    "max_speculative_span",
    "mean_forecast_horizon",
    "max_forecast_horizon",
    "mean_available_order",
    "max_available_order",
    "mean_verification_error",
    "p50_verification_error",
    "p90_verification_error",
    "p95_verification_error",
    "p99_verification_error",
    "mean_threshold",
    "min_threshold",
    "max_threshold",
    "full_due_error_ratio",
    "full_due_max_span_ratio",
    "full_due_first_enhance_ratio",
    "predictor_time_ms",
    "history_update_time_ms",
    "verification_time_ms",
    "verification_block_time_ms",
    "metric_reduction_time_ms",
    "error_reduction_time_ms",
    "scalar_sync_time_ms",
    "scheduler_time_ms",
    "cache_io_time_ms",
)


def _resolve_checkpoint(
    value: str, config_path: Path, config_origin_dir: Path | None = None
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        candidates = [path]
    elif config_origin_dir is not None:
        candidates = [config_origin_dir / path]
    else:
        candidates = [config_path.parent / path]
    if not path.is_absolute():
        candidates.append(PIXARC_ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint not found; checked: {candidates}")


def _denoiser_args(config: Mapping[str, Any]) -> SimpleNamespace:
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
                raise KeyError(f"EMA1 is missing parameter {name}")
            parameter.copy_(ema[name])
    del checkpoint


def _append_metadata(handle, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _archive_input(source: Path, destination: Path) -> None:
    """Create one immutable byte-for-byte snapshot safely across four ranks."""

    content = source.resolve(strict=True).read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"archived input differs: {destination}")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(f"concurrent archived input differs: {destination}")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema_version") != "pixarc-speca-config-v1":
        raise ValueError("config must use schema_version=pixarc-speca-config-v1")
    for section in ("model", "sampling", "speca", "runtime"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"config.{section} must be a mapping")
    speca = validate_speca_config(dict(config["speca"]), require_resolved=True)
    sampling = dict(config["sampling"])
    if sampling.get("method") != "heun" or sampling.get("exact_heun") is not True:
        raise ValueError("JiT SpeCa generation requires exact Heun")
    steps = sampling.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("sampling.steps must be an integer >= 2 for exact Heun")
    noise_scale = sampling.get("noise_scale", 1.0)
    if (
        isinstance(noise_scale, bool)
        or not isinstance(noise_scale, (int, float))
        or not math.isfinite(float(noise_scale))
        or float(noise_scale) < 0
    ):
        raise ValueError("sampling.noise_scale must be finite and non-negative")
    runtime = dict(config["runtime"])
    batch_size = runtime.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("runtime.batch_size must be a positive integer")
    compile_mode = str(runtime.get("compile_mode", "matched_eager"))
    if compile_mode not in {"upstream", "matched_eager", "blockwise"}:
        raise ValueError("unsupported runtime.compile_mode")
    if compile_mode == "upstream" and speca["mode"] != "upstream_full":
        raise ValueError("compile_mode=upstream is valid only for upstream_full")
    return speca, runtime


def _normalized_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Select the scalar 50K trace contract and retain diagnostic traces."""

    result: dict[str, Any] = {
        field: int(summary.get(field, 0)) for field in _SUMMARY_INT_FIELDS
    }
    result.update(
        {
            field: float(
                summary.get(
                    field,
                    -1.0
                    if field in {"mean_available_order", "max_available_order"}
                    else 0.0,
                )
            )
            for field in _SUMMARY_FLOAT_FIELDS
        }
    )
    result.update(
        {
            "trajectory_id": str(summary["trajectory_id"]),
            "sample_ids": [int(value) for value in summary["sample_ids"]],
            "real_batch_size": int(summary["real_batch_size"]),
            "effective_cfg_batch_size": int(summary["effective_cfg_batch_size"]),
            "mode": str(summary["mode"]),
            "call_count_valid": bool(summary.get("call_count_valid", False)),
        }
    )
    # Full/shadow traces contain only scalar records.  They are opt-in and are
    # never emitted by the 50K summary configuration.
    for field in ("nfe_trace", "shadow_trace"):
        if field in summary:
            result[field] = summary[field]
    return result


def _new_counts(prior_sample_count: int, skipped_group_count: int) -> dict[str, Any]:
    return {
        "generated": prior_sample_count,
        "generated_this_invocation": 0,
        "skipped_groups": skipped_group_count,
        "trajectory_count": 0,
        **{f"sum_{field}": 0 for field in _SUMMARY_INT_FIELDS},
        **{
            f"sum_{field}": 0.0
            for field in _SUMMARY_FLOAT_FIELDS
            if field.endswith("_time_ms")
        },
        "max_cache_bytes": 0,
        "max_cache_tensor_count": 0,
        "max_peak_memory_allocated": 0,
        "max_peak_memory_reserved": 0,
        "all_call_counts_valid": True,
    }


def _accumulate(counts: dict[str, Any], summary: Mapping[str, Any]) -> None:
    counts["trajectory_count"] += 1
    for field in _SUMMARY_INT_FIELDS:
        counts[f"sum_{field}"] += int(summary[field])
    for field in _SUMMARY_FLOAT_FIELDS:
        if field.endswith("_time_ms"):
            counts[f"sum_{field}"] += float(summary[field])
    counts["max_cache_bytes"] = max(counts["max_cache_bytes"], int(summary["cache_bytes"]))
    counts["max_cache_tensor_count"] = max(
        counts["max_cache_tensor_count"], int(summary["cache_tensor_count"])
    )
    counts["max_peak_memory_allocated"] = max(
        counts["max_peak_memory_allocated"], int(summary["peak_memory_allocated"])
    )
    counts["max_peak_memory_reserved"] = max(
        counts["max_peak_memory_reserved"], int(summary["peak_memory_reserved"])
    )
    counts["all_call_counts_valid"] = bool(counts["all_call_counts_valid"]) and bool(
        summary["call_count_valid"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-origin-dir")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acknowledge-gpu-job", action="store_true")
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.shard_id < args.world_size:
        raise ValueError("shard-id must satisfy 0 <= shard-id < world-size")
    if (
        not args.acknowledge_gpu_job
        or os.environ.get("SPECA_GPU_TESTS_ALLOWED") != "1"
    ):
        raise RuntimeError(
            "GPU generation is deferred; set SPECA_GPU_TESTS_ALLOWED=1 and pass "
            "--acknowledge-gpu-job only after the target GPU is idle"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("this generation entry requires one visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one shard process must see exactly one CUDA GPU")

    config_path = Path(args.config).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    speca, runtime = _validate_config(config)
    sampling = dict(config["sampling"])
    model_config = dict(config["model"])
    config_origin_dir = (
        Path(args.config_origin_dir).resolve(strict=True)
        if args.config_origin_dir
        else config_path.parent
    )
    if not config_origin_dir.is_dir():
        raise NotADirectoryError(config_origin_dir)
    checkpoint = _resolve_checkpoint(
        str(model_config["checkpoint"]), config_path, config_origin_dir
    )
    identity = checkpoint_identity(checkpoint)

    records = load_manifest(args.manifest)
    batch_size = int(runtime["batch_size"])
    validate_manifest(records, world_size=args.world_size, batch_size=batch_size)
    manifest_path = Path(args.manifest).resolve(strict=True)
    manifest_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    image_size = int(model_config.get("image_size", 256))
    validate_manifest_sidecar(
        manifest_path,
        records,
        world_size=args.world_size,
        batch_size=batch_size,
        generator_device="cuda",
        noise_dtype="float32",
        noise_shape=(3, image_size, image_size),
    )

    output_root = Path(args.output_root).resolve()
    sample_dir = output_root / "samples"
    metadata_path = output_root / "metadata" / f"rank_{args.shard_id}.jsonl"
    summary_path = output_root / "summaries" / f"rank_{args.shard_id}_summary.json"
    for path in (sample_dir, metadata_path.parent, summary_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    _archive_input(config_path, output_root / "config_resolved.yaml")
    _archive_input(manifest_path, output_root / "input_manifest.jsonl")
    _archive_input(manifest_sidecar, output_root / "input_manifest.jsonl.meta.json")
    if metadata_path.exists() and not args.resume:
        raise FileExistsError(f"rank metadata already exists: {metadata_path}")
    prior_metadata = load_metadata_jsonl(metadata_path) if metadata_path.exists() else {}
    config_hash = canonical_hash(config)
    speca_config_hash = canonical_hash(speca)
    manifest_digest = sha256_file(manifest_path)
    groups, skipped_groups = resumable_batch_groups(
        records,
        args.shard_id,
        sample_dir,
        prior_metadata,
        manifest_sha256=manifest_digest,
        config_hash=config_hash,
        speca_config_hash=speca_config_hash,
        speca_config=speca,
        checkpoint_path=str(checkpoint),
        checkpoint_size=int(identity["checkpoint_size"]),
        method=str(speca["mode"]),
        interval=speca["interval"],
        max_order=int(speca["max_order"]),
        coordinate_mode=str(speca["coordinate_mode"]),
        resolution=image_size,
    )

    denoiser_args = _denoiser_args(config)
    compile_mode = str(runtime.get("compile_mode", "matched_eager"))
    model = SpeCaDenoiser(
        denoiser_args,
        mode=str(speca["mode"]),
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
        trace_mode=str(speca["trace_mode"]),
        interval=speca["interval"],
        compile_mode=compile_mode,
    )
    unwrapped_for_eager = int(getattr(model.net, "compile_wrappers_unwrapped", 0))
    if compile_mode != "matched_eager":
        model.net.compile()
    _load_ema1(model, checkpoint)
    model = model.cuda().eval()
    dtype_name = str(sampling.get("dtype", "bfloat16"))
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype_name
    )
    if autocast_dtype is None:
        raise ValueError("sampling.dtype must be bfloat16 or float16")

    current_git_revision = git_revision(PIXARC_ROOT)
    run_config = {
        "model": denoiser_args.model,
        "model_config_hash": canonical_hash(model_config),
        "ema": "EMA1",
        "input_config_hash": config_hash,
        "speca_config_hash": speca_config_hash,
        "port_source_sha256": source_tree_sha256(BASELINE_ROOT),
        "manifest_sha256": manifest_digest,
        "manifest_sidecar_sha256": sha256_file(manifest_sidecar),
        "manifest_records_sha256": manifest_records_sha256(records),
        "initial_noise_protocol": (
            "per-sample torch.Generator(device='cuda') standard Gaussian; "
            "JiT noise_scale applied by config"
        ),
        "rng_device": "cuda",
        "generator_type": "torch.Generator.manual_seed per sample",
        "initial_noise_dtype": "float32",
        "initial_noise_shape": [3, denoiser_args.img_size, denoiser_args.img_size],
        "noise_scale": denoiser_args.noise_scale,
        "sampler": denoiser_args.sampling_method,
        "sampler_config_hash": canonical_hash(sampling),
        "steps": denoiser_args.num_sampling_steps,
        "cfg_scale": denoiser_args.cfg,
        "guidance_interval": [denoiser_args.interval_min, denoiser_args.interval_max],
        "timeshift": None,
        "dtype": dtype_name,
        "resolution": denoiser_args.img_size,
        "image_postprocessing": "(x+1)/2; clip; round(x*255); RGB uint8 PNG",
        "world_size": args.world_size,
        "batch_size": batch_size,
        "real_batch_size": batch_size,
        "effective_cfg_batch_size": 2 * batch_size,
        "batch_grouping": "manifest batch_group_id/position_in_batch",
        "gate_protocol": "one batch-global scheduler action shared by all real samples and CFG streams",
        "cfg_execution": "separate cond then uncond forwards; one shared scheduler",
        "scheduler_semantics": "released-code-faithful; current verification error schedules the next NFE",
        "compile_mode": compile_mode,
        "compile_wrappers_unwrapped": unwrapped_for_eager,
        "cache4diffusion_commit": CACHE4DIFFUSION_COMMIT,
        "taylorseer_commit": TAYLORSEER_COMMIT,
        **{field: speca[field] for field in SPECA_CONFIG_FIELDS},
    }
    run_manifest_path = output_root / "run_manifest.json"
    if args.resume and not run_manifest_path.exists():
        has_samples = next(sample_dir.glob("*.png"), None) is not None
        has_metadata = next(metadata_path.parent.glob("rank_*.jsonl"), None) is not None
        if has_samples or has_metadata:
            raise FileNotFoundError(
                "resume without run_manifest.json is allowed only before any "
                "durable sample or rank metadata exists"
            )
    if not run_manifest_path.exists():
        run_metadata = build_run_metadata(
            model=denoiser_args.model,
            method=str(speca["mode"]),
            config=run_config,
            checkpoint=checkpoint,
            manifest_sha256=manifest_digest,
            git_commit=current_git_revision,
            cache4diffusion_commit=CACHE4DIFFUSION_COMMIT,
            taylorseer_commit=TAYLORSEER_COMMIT,
        )
        run_metadata.update(
            {
                "input_config_hash": config_hash,
                "speca_config_hash": speca_config_hash,
                **{field: speca[field] for field in SPECA_CONFIG_FIELDS},
            }
        )
        atomic_create_json(run_manifest_path, run_metadata)
    existing_run = load_json(run_manifest_path)
    expected_existing = {
        "config": run_config,
        "manifest_sha256": manifest_digest,
        "manifest_sidecar_sha256": sha256_file(manifest_sidecar),
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": int(identity["checkpoint_size"]),
        "method": str(speca["mode"]),
        "input_config_hash": config_hash,
        "speca_config_hash": speca_config_hash,
        "git_commit": current_git_revision,
        "cache4diffusion_commit": CACHE4DIFFUSION_COMMIT,
        "taylorseer_commit": TAYLORSEER_COMMIT,
        "pytorch_version": torch.__version__,
        **{field: speca[field] for field in SPECA_CONFIG_FIELDS},
    }
    for key, expected_value in expected_existing.items():
        if existing_run.get(key) != expected_value:
            raise RuntimeError(f"existing run_manifest mismatch: {key}")

    counts = _new_counts(len(prior_metadata), len(skipped_groups))
    prior_groups: dict[str, Mapping[str, Any]] = {}
    for row in prior_metadata.values():
        prior_groups.setdefault(str(row["batch_group_id"]), row)
    for row in prior_groups.values():
        restored = {
            field: row[f"trajectory_{field}"]
            for field in (*_SUMMARY_INT_FIELDS, *_SUMMARY_FLOAT_FIELDS)
        }
        restored["call_count_valid"] = row["trajectory_call_count_valid"]
        _accumulate(counts, restored)

    file_mode = "a" if args.resume else "x"
    with metadata_path.open(file_mode, encoding="utf-8") as metadata_handle:
        for group in groups:
            seeds = [record.seed for record in group]
            labels = torch.tensor(
                [record.class_id for record in group], device="cuda", dtype=torch.long
            )
            noise = initial_noise(
                seeds,
                (3, denoiser_args.img_size, denoiser_args.img_size),
                device="cuda",
                dtype=torch.float32,
            )
            trajectory_id = f"shard-{args.shard_id}-group-{group[0].batch_group_id}"
            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                images = model.generate(
                    labels,
                    noise=noise,
                    sample_ids=[record.sample_id for record in group],
                    trajectory_id=trajectory_id,
                )
            arrays = np.round(
                np.clip((images.float().cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
                * 255.0
            ).astype(np.uint8)
            arrays = arrays.transpose(0, 2, 3, 1)
            raw_summary = getattr(model, "_last_speca_summary", None)
            if not isinstance(raw_summary, dict):
                raise RuntimeError("JiT denoiser did not expose a SpeCa summary")
            group_summary = _normalized_summary(raw_summary)
            if not group_summary["call_count_valid"]:
                raise RuntimeError("SpeCa trajectory failed its NFE/call-count invariant")
            trace_payload = {
                f"trajectory_{field}": group_summary[field]
                for field in ("nfe_trace", "shadow_trace")
                if field in group_summary
            }
            scalar_summary = {
                f"trajectory_{field}": group_summary[field]
                for field in (
                    *_SUMMARY_INT_FIELDS,
                    *_SUMMARY_FLOAT_FIELDS,
                    "call_count_valid",
                )
            }
            scalar_summary.update(
                {
                    "trajectory_id": group_summary["trajectory_id"],
                    "trajectory_sample_ids": group_summary["sample_ids"],
                    "real_batch_size": group_summary["real_batch_size"],
                    "effective_cfg_batch_size": group_summary[
                        "effective_cfg_batch_size"
                    ],
                    "trajectory_mode": group_summary["mode"],
                }
            )
            for record, array in zip(group, arrays, strict=True):
                atomic_write_png(
                    array,
                    image_path(sample_dir, record.sample_id),
                    resolution=denoiser_args.img_size,
                )
                _append_metadata(
                    metadata_handle,
                    {
                        "sample_id": record.sample_id,
                        "class_id": record.class_id,
                        "seed": record.seed,
                        "batch_group_id": record.batch_group_id,
                        "position_in_batch": record.position_in_batch,
                        "manifest_sha256": manifest_digest,
                        "config_hash": config_hash,
                        "speca_config_hash": speca_config_hash,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_size": int(identity["checkpoint_size"]),
                        "method": str(speca["mode"]),
                        **{field: speca[field] for field in SPECA_CONFIG_FIELDS},
                        **scalar_summary,
                        **trace_payload,
                        "status": "ok",
                    },
                )
                counts["generated"] += 1
                counts["generated_this_invocation"] += 1
            _accumulate(counts, group_summary)

    summary = {
        **counts,
        "shard_id": args.shard_id,
        "method": str(speca["mode"]),
        "speca_config_hash": speca_config_hash,
        "invocation_id": os.environ.get("SPECA_INVOCATION_ID"),
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
