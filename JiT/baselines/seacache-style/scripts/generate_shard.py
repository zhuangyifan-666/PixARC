#!/usr/bin/env python3
"""Generate one deterministic JiT manifest shard (GPU execution is deferred)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from seacache_style.controller import finalize_timing_summary
from seacache_style.image_io import (
    atomic_write_png,
    image_path,
    load_metadata_jsonl,
    resumable_batch_groups,
)
from seacache_style.jit_denoiser import SeaCacheDenoiser
from seacache_style.jit_model import configure_jit_compile_mode
from seacache_style.manifest import (
    initial_noise,
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
)
from seacache_style.metadata import (
    atomic_create_json,
    atomic_write_json,
    build_run_metadata,
    canonical_hash,
    checkpoint_identity,
    git_revision,
    load_json,
    source_tree_sha256,
)


PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = Path(__file__).resolve().parents[1]
SEACACHE_COMMIT = "8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2"


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


def _denoiser_args(config: dict[str, object]) -> SimpleNamespace:
    model = dict(config["model"])
    sampling = dict(config["sampling"])
    extra = dict(model.get("args", {}))
    interval = sampling.get("guidance_interval", [0.1, 1.0])
    values = {
        "model": model["variant"],
        "img_size": int(model.get("image_size", 256)),
        "class_num": int(model.get("num_classes", 1000)),
        "attn_dropout": float(extra.get("attn_dropout", 0.0)),
        "proj_dropout": float(extra.get("proj_dropout", 0.0)),
        "label_drop_prob": float(extra.get("label_drop_prob", 0.1)),
        "P_mean": float(extra.get("P_mean", -0.8)),
        "P_std": float(extra.get("P_std", 0.8)),
        "t_eps": float(extra.get("t_eps", 0.05)),
        "noise_scale": float(sampling.get("noise_scale", 1.0)),
        "ema_decay1": float(extra.get("ema_decay1", 0.9999)),
        "ema_decay2": float(extra.get("ema_decay2", 0.9996)),
        "sampling_method": str(sampling["method"]),
        "num_sampling_steps": int(sampling["steps"]),
        "cfg": float(sampling["cfg_scale"]),
        "interval_min": float(interval[0]),
        "interval_max": float(interval[1]),
    }
    return SimpleNamespace(**values)


def _load_ema1(model: SeaCacheDenoiser, checkpoint_path: Path) -> None:
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


def _append_metadata(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _validate_seacache_config(value: dict[str, object]) -> None:
    mode = str(value.get("mode", "full"))
    if mode not in {"full", "force_full_with_gate", "seacache"}:
        raise ValueError(f"unsupported seacache.mode: {mode!r}")
    threshold = value.get("threshold")
    if mode != "seacache":
        return
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("seacache mode requires a finite non-negative numeric threshold")
    if not math.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError("seacache mode requires a finite non-negative numeric threshold")


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
    if not args.acknowledge_gpu_job or os.environ.get("SEACACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("GPU generation is deferred; explicit safety acknowledgement is required")
    if not torch.cuda.is_available():
        raise RuntimeError("this generation entry requires one visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one shard process must see exactly one CUDA GPU")

    config_path = Path(args.config).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    runtime = dict(config["runtime"])
    seacache = dict(config["seacache"])
    _validate_seacache_config(seacache)
    sampling = dict(config["sampling"])
    noise_scale_raw = sampling.get("noise_scale", 1.0)
    if isinstance(noise_scale_raw, bool) or not isinstance(
        noise_scale_raw, (int, float)
    ):
        raise ValueError("sampling.noise_scale must be a finite non-negative number")
    if not math.isfinite(float(noise_scale_raw)) or float(noise_scale_raw) < 0:
        raise ValueError("sampling.noise_scale must be a finite non-negative number")
    model_config = dict(config["model"])
    if str(runtime.get("compile_mode", "matched_eager")) not in {
        "matched_eager",
        "blockwise",
        "upstream",
    }:
        raise ValueError("unsupported runtime.compile_mode")
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
    validate_manifest(
        records,
        world_size=args.world_size,
        batch_size=int(runtime["batch_size"]),
    )

    output_root = Path(args.output_root).resolve()
    sample_dir = output_root / "samples"
    metadata_path = output_root / "metadata" / f"rank_{args.shard_id}.jsonl"
    summary_path = output_root / "summaries" / f"rank_{args.shard_id}_summary.json"
    for path in (sample_dir, metadata_path.parent, summary_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists() and not args.resume:
        raise FileExistsError(f"rank metadata already exists: {metadata_path}")
    prior_metadata = load_metadata_jsonl(metadata_path) if metadata_path.exists() else {}
    config_hash = canonical_hash(config)
    manifest_digest = sha256_file(args.manifest)
    groups, skipped_groups = resumable_batch_groups(
        records,
        args.shard_id,
        sample_dir,
        prior_metadata,
        manifest_sha256=manifest_digest,
        config_hash=config_hash,
        checkpoint_path=str(checkpoint),
        checkpoint_size=int(identity["checkpoint_size"]),
        threshold=seacache.get("threshold"),
        resolution=int(model_config.get("image_size", 256)),
    )

    denoiser_args = _denoiser_args(config)
    model = SeaCacheDenoiser(
        denoiser_args,
        seacache_mode=str(seacache["mode"]),
        seacache_threshold=seacache.get("threshold"),
        seacache_trace_mode=str(seacache.get("trace_mode", "summary")),
    )
    unwrapped_for_eager = configure_jit_compile_mode(
        model.net, str(runtime.get("compile_mode", "matched_eager"))
    )
    _load_ema1(model, checkpoint)
    model = model.cuda().eval()
    dtype_name = str(sampling.get("dtype", "bfloat16"))
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    if autocast_dtype is None:
        raise ValueError("sampling.dtype must be bfloat16 or float16")

    current_git_revision = git_revision(PIXARC_ROOT)
    run_config = {
        "model": denoiser_args.model,
        # Checkpoint identity is bound independently by its resolved path and
        # size.  Exclude its relative/absolute spelling from the architecture
        # hash so equivalent frozen configs remain comparable.
        "model_config_hash": canonical_hash(
            {key: value for key, value in model_config.items() if key != "checkpoint"}
        ),
        "ema": "EMA1",
        "input_config_hash": config_hash,
        "port_source_sha256": source_tree_sha256(BASELINE_ROOT),
        "manifest_sha256": manifest_digest,
        "manifest_records_sha256": manifest_records_sha256(records),
        "initial_noise_protocol": "per-sample torch.Generator(device='cuda') standard Gaussian; JiT noise_scale applied by config",
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
        "batch_size": int(runtime["batch_size"]),
        "batch_grouping": "manifest batch_group_id/position_in_batch",
        "compile_mode": runtime.get("compile_mode", "matched_eager"),
        "compile_wrappers_unwrapped": unwrapped_for_eager,
    }
    run_manifest_path = output_root / "run_manifest.json"
    if args.resume and not run_manifest_path.exists():
        has_samples = next((output_root / "samples").glob("*.png"), None) is not None
        has_metadata = (
            next((output_root / "metadata").glob("rank_*.jsonl"), None) is not None
        )
        if has_samples or has_metadata:
            raise FileNotFoundError(
                "resume without run_manifest.json is allowed only before any "
                "durable sample or rank metadata exists"
            )
    if not run_manifest_path.exists():
        run_metadata = build_run_metadata(
            model=denoiser_args.model,
            method=str(seacache["mode"]),
            config=run_config,
            checkpoint=checkpoint,
            manifest_sha256=run_config["manifest_sha256"],
            git_commit=current_git_revision,
            seacache_commit=SEACACHE_COMMIT,
        )
        run_metadata["threshold"] = seacache.get("threshold")
        run_metadata["input_config_hash"] = config_hash
        atomic_create_json(run_manifest_path, run_metadata)
    existing_run = load_json(run_manifest_path)
    expected_existing = {
        "config": run_config,
        "manifest_sha256": manifest_digest,
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": int(identity["checkpoint_size"]),
        "method": str(seacache["mode"]),
        "threshold": seacache.get("threshold"),
        "git_commit": current_git_revision,
        "seacache_commit": SEACACHE_COMMIT,
        "pytorch_version": torch.__version__,
    }
    for key, expected_value in expected_existing.items():
        if existing_run.get(key) != expected_value:
            raise RuntimeError(f"existing run_manifest mismatch: {key}")

    mode = "a" if args.resume else "x"
    counts = {
        "generated": len(prior_metadata),
        "generated_this_invocation": 0,
        "skipped_groups": len(skipped_groups),
        "full_calls": 0,
        "reuse_calls": 0,
        "gate_time_ms": 0.0,
        "fft_time_ms": 0.0,
        "cache_io_time_ms": 0.0,
    }
    prior_groups: dict[str, dict[str, object]] = {}
    for row in prior_metadata.values():
        group_id = str(row["batch_group_id"])
        prior_groups.setdefault(group_id, row)
    for row in prior_groups.values():
        for field in ("full_calls", "reuse_calls"):
            counts[field] += int(row[f"trajectory_{field}"])
        for field in ("gate_time_ms", "fft_time_ms", "cache_io_time_ms"):
            counts[field] += float(row[f"trajectory_{field}"])
    with metadata_path.open(mode, encoding="utf-8") as metadata_handle:
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
                np.clip(((images.float().cpu().numpy() + 1.0) / 2.0), 0.0, 1.0)
                * 255.0
            ).astype(np.uint8)
            arrays = arrays.transpose(0, 2, 3, 1)
            summaries = list(
                getattr(model, "_last_seacache_summaries", {}).values()
            )
            for summary in summaries:
                finalize_timing_summary(summary)
            group_summary = {
                "full_calls": sum(int(value["full_calls"]) for value in summaries),
                "reuse_calls": sum(int(value["reuse_calls"]) for value in summaries),
                "gate_time_ms": sum(
                    float(value.get("gate_time_ms", 0.0)) for value in summaries
                ),
                "fft_time_ms": sum(
                    float(value.get("fft_time_ms", 0.0)) for value in summaries
                ),
                "cache_io_time_ms": sum(
                    float(value.get("cache_io_time_ms", 0.0)) for value in summaries
                ),
            }
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
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_size": int(identity["checkpoint_size"]),
                        "threshold": seacache.get("threshold"),
                        **{
                            f"trajectory_{key}": value
                            for key, value in group_summary.items()
                        },
                        "status": "ok",
                    },
                )
                counts["generated"] += 1
                counts["generated_this_invocation"] += 1
            for key, value in group_summary.items():
                counts[key] += value
    summary = {
        **counts,
        "shard_id": args.shard_id,
        "invocation_id": os.environ.get("SEACACHE_INVOCATION_ID"),
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
