#!/usr/bin/env python3
"""Resolve and execute one manifest-backed PixelGen GPU shard (deferred)."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from speca_style.manifest import (
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
    validate_manifest_sidecar,
)
from speca_style.image_io import load_metadata_jsonl, resumable_batch_groups
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
    validate_speca_config as validate_common_speca_config,
)


PIXARC_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = PIXARC_ROOT / "third-party" / "PixelGen"
LOCAL_ROOT = Path(__file__).resolve().parents[1]
CACHE4DIFFUSION_COMMIT = "91a1949fcc88acab46547f0b5f295f5de2df2870"
TAYLORSEER_COMMIT = "704ee98c74f7f04da443daa3c0aa2cc7803d86e3"


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


def _write_yaml(path: Path, value: object) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if existing != value:
            raise FileExistsError(
                f"resolved rank config exists with different content: {path}"
            )
        return
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())


def _archive_input(source: Path, destination: Path) -> None:
    """Create an immutable byte-for-byte snapshot, safely across four ranks."""

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


def _validate_speca_config(value: dict[str, object]) -> None:
    validate_common_speca_config(value, require_resolved=True)


def _sampler_name(specification: dict[str, object]) -> str:
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
    if not args.acknowledge_gpu_job or os.environ.get("SPECA_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("GPU generation is deferred; explicit safety acknowledgement is required")
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible) != 1:
        raise RuntimeError("one PixelGen shard process must see exactly one CUDA GPU")

    config_path = Path(args.config).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    if config.get("schema_version") != "pixarc-speca-config-v1":
        raise ValueError("unsupported config schema; expected pixarc-speca-config-v1")
    runtime = dict(config["runtime"])
    speca = dict(config["speca"])
    _validate_speca_config(speca)
    noise_scale_raw = runtime.get("noise_scale", 1.0)
    if isinstance(noise_scale_raw, bool) or not isinstance(
        noise_scale_raw, (int, float)
    ):
        raise ValueError("runtime.noise_scale must be a finite non-negative number")
    noise_scale = float(noise_scale_raw)
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("runtime.noise_scale must be a finite non-negative number")
    compile_mode = str(runtime.get("compile_mode", "matched_eager"))
    if compile_mode not in {"matched_eager", "blockwise", "upstream"}:
        raise ValueError("unsupported runtime.compile_mode")
    if compile_mode == "upstream" and speca["mode"] != "upstream_full":
        raise ValueError("compile_mode=upstream is valid only for upstream_full")
    batch_size_raw = runtime.get("batch_size")
    effective_cfg_batch_size_raw = runtime.get("effective_cfg_batch_size")
    if (
        isinstance(batch_size_raw, bool)
        or not isinstance(batch_size_raw, int)
        or isinstance(effective_cfg_batch_size_raw, bool)
        or not isinstance(effective_cfg_batch_size_raw, int)
    ):
        raise ValueError("runtime batch sizes must be integers")
    batch_size = batch_size_raw
    effective_cfg_batch_size = effective_cfg_batch_size_raw
    if batch_size != 1 or effective_cfg_batch_size != 2:
        raise ValueError("primary PixelGen runs require real batch=1 and CFG batch=2")
    records = load_manifest(args.manifest)
    validate_manifest(
        records,
        world_size=args.world_size,
        batch_size=batch_size,
    )
    configured_denoiser = dict(dict(config["model"])["denoiser"])
    configured_denoiser_args = dict(configured_denoiser["init_args"])
    configured_resolution = int(configured_denoiser_args.get("input_size", 256))
    manifest_sidecar = Path(args.manifest).with_suffix(
        Path(args.manifest).suffix + ".meta.json"
    )
    validate_manifest_sidecar(
        args.manifest,
        records,
        world_size=args.world_size,
        batch_size=batch_size,
        generator_device="cpu",
        noise_dtype="float32",
        noise_shape=(3, configured_resolution, configured_resolution),
    )
    config_origin_dir = (
        Path(args.config_origin_dir).resolve(strict=True)
        if args.config_origin_dir
        else config_path.parent
    )
    if not config_origin_dir.is_dir():
        raise NotADirectoryError(config_origin_dir)
    checkpoint = _resolve_checkpoint(
        str(config["checkpoint"]), config_path, config_origin_dir
    )
    checkpoint_info = checkpoint_identity(checkpoint)
    output_root = Path(args.output_root).resolve()
    for relative in ("samples", "metadata", "summaries", "logs"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    _archive_input(config_path, output_root / "config_resolved.yaml")
    _archive_input(Path(args.manifest), output_root / "input_manifest.jsonl")
    _archive_input(
        manifest_sidecar, output_root / "input_manifest.jsonl.meta.json"
    )

    model_config = dict(config["model"])
    denoiser = dict(model_config["denoiser"])
    denoiser_args = dict(denoiser["init_args"])
    sampler = dict(model_config["diffusion_sampler"])
    sampler_args = dict(sampler["init_args"])
    expected_denoiser = "speca_style.pixelgen_model.SpeCaPixelGenJiT"
    expected_sampler = "speca_style.pixelgen_sampler.SpeCaHeunSamplerJiT"
    if denoiser.get("class_path") != expected_denoiser:
        raise ValueError(f"PixelGen SpeCa denoiser must be {expected_denoiser}")
    if sampler.get("class_path") != expected_sampler:
        raise ValueError(f"PixelGen SpeCa sampler must be {expected_sampler}")
    if not bool(sampler_args.get("exact_henu")):
        raise ValueError("PixelGen SpeCa generation requires exact_henu=true")
    num_steps = sampler_args.get("num_steps")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("PixelGen sampler num_steps must be a positive integer")
    model_config_hash = canonical_hash(
        {
            **{
                key: value
                for key, value in model_config.items()
                if key not in {"denoiser", "diffusion_sampler", "compile_mode"}
            },
            "denoiser": {
                "class_path": denoiser["class_path"],
                "init_args": {
                    key: value
                    for key, value in denoiser_args.items()
                    if not key.startswith("speca_") and key != "compile_mode"
                },
            },
        }
    )
    sampler_config_hash = canonical_hash(sampler)
    resolution = int(denoiser_args.get("input_size", 256))
    config_hash = canonical_hash(config)
    speca_config_hash = canonical_hash(speca)
    metadata_path = output_root / "metadata" / f"rank_{args.shard_id}.jsonl"
    existing_metadata = (
        load_metadata_jsonl(metadata_path) if metadata_path.exists() else {}
    )
    pending_groups, skipped_group_ids = resumable_batch_groups(
        records,
        args.shard_id,
        output_root / "samples",
        existing_metadata,
        manifest_sha256=sha256_file(args.manifest),
        config_hash=config_hash,
        speca_config_hash=speca_config_hash,
        speca_config={field: speca[field] for field in SPECA_CONFIG_FIELDS},
        checkpoint_path=str(checkpoint),
        checkpoint_size=int(checkpoint_info["checkpoint_size"]),
        method=str(speca["mode"]),
        interval=speca["interval"],
        max_order=speca["max_order"],
        coordinate_mode=str(speca["coordinate_mode"]),
        resolution=resolution,
    )

    resolved = dict(config)
    resolved.pop("schema_version", None)
    resolved.pop("checkpoint", None)
    resolved.pop("speca", None)
    resolved.pop("runtime", None)
    resolved["trainer"] = dict(resolved.get("trainer", {}))
    resolved["trainer"].update(
        {
            "default_root_dir": str(output_root / "lightning" / f"rank_{args.shard_id}"),
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "logger": False,
            "precision": runtime.get("precision", "bf16-mixed"),
            "use_distributed_sampler": False,
            "callbacks": [
                {
                    "class_path": "speca_style.pixelgen_io.AtomicManifestSaveHook",
                    "init_args": {
                        "output_root": str(output_root),
                        "shard_id": args.shard_id,
                        "resolution": resolution,
                        "resume": args.resume,
                    },
                }
            ],
        }
    )
    resolved["model"] = model_config
    resolved["model"]["compile_mode"] = runtime.get("compile_mode", "matched_eager")
    resolved["model"]["denoiser"] = denoiser
    resolved["model"]["denoiser"]["init_args"] = denoiser_args
    resolved["model"]["denoiser"]["init_args"].update(
        {
            f"speca_{field}": speca[field]
            for field in SPECA_CONFIG_FIELDS
            if field != "scheduler_mode"
        }
    )
    resolved["model"]["denoiser"]["init_args"].update(
        {"speca_mode": speca["mode"], "compile_mode": compile_mode}
    )
    resolved["model"]["diffusion_sampler"] = sampler
    resolved["data"] = dict(resolved["data"])
    resolved["data"].update(
        {
            "train_dataset": None,
            "eval_dataset": None,
            "pred_batch_size": batch_size,
            "pred_num_workers": int(runtime.get("num_workers", 1)),
            "pred_dataset": {
                "class_path": "speca_style.pixelgen_io.ManifestNoiseDataset",
                "init_args": {
                    "manifest_path": str(Path(args.manifest).resolve()),
                    "shard_id": args.shard_id,
                    "world_size": args.world_size,
                    "output_root": str(output_root),
                    "batch_size": batch_size,
                    "config_hash": config_hash,
                    "speca_config_hash": speca_config_hash,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_size": int(checkpoint_info["checkpoint_size"]),
                    "method": speca["mode"],
                    "speca_config": {
                        field: speca[field] for field in SPECA_CONFIG_FIELDS
                    },
                    "resolution": resolution,
                    "noise_scale": noise_scale,
                    "resume": args.resume,
                },
            },
        }
    )
    invocation_id = os.environ.get("SPECA_INVOCATION_ID", "direct")
    if not invocation_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in invocation_id
    ):
        raise ValueError("unsafe SPECA_INVOCATION_ID")
    resolved_path = (
        output_root
        / "metadata"
        / f"rank_{args.shard_id}_resolved_{invocation_id}.yaml"
    )
    _write_yaml(resolved_path, resolved)

    current_git_revision = git_revision(PIXARC_ROOT)
    run_config = {
        "model": "PixelGen-JiT",
        "model_config_hash": model_config_hash,
        "ema": "ema_denoiser",
        "input_config_hash": config_hash,
        "port_source_sha256": source_tree_sha256(LOCAL_ROOT),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_sidecar_sha256": sha256_file(manifest_sidecar),
        "manifest_records_sha256": manifest_records_sha256(records),
        "initial_noise_protocol": "per-sample torch.Generator(device='cpu') standard Gaussian, dataset noise_scale",
        "rng_device": "cpu",
        "generator_type": "torch.Generator.manual_seed per sample",
        "initial_noise_dtype": "float32",
        "initial_noise_shape": [3, resolution, resolution],
        "noise_scale": noise_scale,
        "sampler": _sampler_name(sampler),
        "sampler_config_hash": sampler_config_hash,
        "steps": int(sampler_args["num_steps"]),
        "cfg_scale": float(sampler_args["guidance"]),
        "guidance_interval": [
            float(sampler_args.get("guidance_interval_min", 0.0)),
            float(sampler_args.get("guidance_interval_max", 1.0)),
        ],
        "timeshift": float(sampler_args.get("timeshift", 1.0)),
        "dtype": str(runtime.get("precision", "bf16-mixed")),
        "resolution": resolution,
        "image_postprocessing": "PixelGen fp2uint8 after identity PixelAE decode; RGB uint8 PNG",
        "world_size": args.world_size,
        "batch_size": batch_size,
        "real_batch_size": batch_size,
        "effective_cfg_batch_size": effective_cfg_batch_size,
        "batch_grouping": "manifest batch_group_id/position_in_batch",
        "cfg_execution": "single combined [unconditional, conditional] effective-2B forward",
        "gate_protocol": "one batch-global scheduler over the complete combined 2B tensor",
        "compile_mode": runtime.get("compile_mode", "matched_eager"),
        "speca_config_hash": speca_config_hash,
    }
    run_config.update({field: speca[field] for field in SPECA_CONFIG_FIELDS})
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
        value = build_run_metadata(
            model="PixelGen-JiT",
            method=str(speca["mode"]),
            config=run_config,
            checkpoint=checkpoint,
            manifest_sha256=run_config["manifest_sha256"],
            git_commit=current_git_revision,
            cache4diffusion_commit=CACHE4DIFFUSION_COMMIT,
            taylorseer_commit=TAYLORSEER_COMMIT,
        )
        value.update({field: speca[field] for field in SPECA_CONFIG_FIELDS})
        value["input_config_hash"] = config_hash
        value["speca_config_hash"] = speca_config_hash
        atomic_create_json(run_manifest_path, value)
    existing_run = load_json(run_manifest_path)
    expected_existing = {
        "config": run_config,
        "manifest_sha256": run_config["manifest_sha256"],
        "manifest_sidecar_sha256": run_config["manifest_sidecar_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": int(checkpoint_info["checkpoint_size"]),
        "method": str(speca["mode"]),
        "speca_config_hash": speca_config_hash,
        "git_commit": current_git_revision,
        "cache4diffusion_commit": CACHE4DIFFUSION_COMMIT,
        "taylorseer_commit": TAYLORSEER_COMMIT,
        "pytorch_version": torch.__version__,
    }
    expected_existing.update({field: speca[field] for field in SPECA_CONFIG_FIELDS})
    for key, expected_value in expected_existing.items():
        if existing_run.get(key) != expected_value:
            raise RuntimeError(f"existing run_manifest mismatch: {key}")

    if not pending_groups:
        integer_fields = (
            "total_nfe", "full_nfe", "taylor_nfe", "network_forward_count",
            "verified_taylor_nfe", "unverified_taylor_nfe",
            "verification_pass_count", "verification_fail_count",
            "verification_block_calls", "full_due_first_enhance",
            "full_due_max_taylor", "full_due_previous_error",
        )
        time_fields = (
            "predictor_time_ms", "history_update_time_ms", "verification_time_ms",
            "verification_block_time_ms", "metric_reduction_time_ms",
            "error_reduction_time_ms", "scalar_sync_time_ms", "scheduler_time_ms",
            "cache_io_time_ms",
        )
        sum_fields = (*integer_fields, *time_fields)
        maximum_fields = {
            "max_speculative_span": "max_speculative_span",
            "max_forecast_horizon": "max_forecast_horizon",
            "max_available_order": "max_available_order",
            "cache_bytes": "max_cache_bytes",
            "cache_allocated_bytes": "max_cache_allocated_bytes",
            "cache_tensor_count": "max_cache_tensor_count",
            "peak_memory_allocated": "max_peak_memory_allocated",
            "peak_memory_reserved": "max_peak_memory_reserved",
        }
        counts: dict[str, int | float | str | None] = {
            "generated": len(existing_metadata),
            "generated_this_invocation": 0,
            "skipped_groups": len(skipped_group_ids),
            "trajectory_count": 0,
            "shard_id": args.shard_id,
            "invocation_id": os.environ.get("SPECA_INVOCATION_ID"),
        }
        counts.update({field: 0 for field in sum_fields})
        counts.update({field: 0 for field in maximum_fields.values()})
        existing_groups: dict[str, dict[str, object]] = {}
        for row in existing_metadata.values():
            existing_groups.setdefault(str(row["batch_group_id"]), row)
        for row in existing_groups.values():
            counts["trajectory_count"] = int(counts["trajectory_count"]) + 1
            for field in integer_fields:
                counts[field] = int(counts[field]) + int(
                    row.get(f"trajectory_{field}", 0) or 0
                )
            for field in time_fields:
                counts[field] = float(counts[field]) + float(
                    row.get(f"trajectory_{field}", 0) or 0
                )
            for source, destination in maximum_fields.items():
                counts[destination] = max(
                    float(counts[destination]),
                    float(row.get(f"trajectory_{source}", 0) or 0),
                )
        summary_path = (
            output_root
            / "summaries"
            / f"rank_{args.shard_id}_summary.json"
        )
        atomic_write_json(summary_path, counts)
        print(json.dumps(counts, indent=2, sort_keys=True))
        return

    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(UPSTREAM_ROOT), str(LOCAL_ROOT), existing_pythonpath]
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("pixelgen_main.py")),
        "predict",
        "-c",
        str(resolved_path),
        "--ckpt_path",
        str(checkpoint),
    ]
    completed = subprocess.run(command, cwd=UPSTREAM_ROOT, env=environment)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
