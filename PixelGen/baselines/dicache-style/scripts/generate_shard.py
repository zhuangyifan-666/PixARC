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

from dicache_style.manifest import (
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
    validate_manifest_sidecar,
)
from dicache_style.image_io import load_metadata_jsonl, resumable_batch_groups
from dicache_style.metadata import (
    DICACHE_CONFIG_FIELDS,
    UNRELEASED_RELEASE_GATE,
    archived_release_gate_sha256,
    atomic_create_json,
    atomic_write_json,
    build_run_metadata,
    canonical_hash,
    checkpoint_identity,
    git_revision,
    load_json,
    source_tree_sha256,
    validate_dicache_config as validate_common_dicache_config,
)
from dicache_style.source_identity import release_source_bindings


PIXARC_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = PIXARC_ROOT / "third-party" / "PixelGen"
LOCAL_ROOT = Path(__file__).resolve().parents[1]
DICACHE_COMMIT = "fdbe20b669c9174bbed5ec994de073fd881c8010"
PIXELGEN_TREE_ID = "3043acf90f255a264f1445bda9ea8d468ba91a58"


def _verify_worker_release(
    *, gate: str, expected_sha256: str, config: Path, manifest: Path,
    output_root: Path,
) -> dict[str, object]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("release_gate.py")),
            "worker-verify",
            "--model-family", "PixelGen",
            "--gate", gate,
            "--expected-gate-sha256", expected_sha256,
            "--config", str(config),
            "--manifest", str(manifest),
            "--output-root", str(output_root),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env=environment,
    )
    report = json.loads(completed.stdout)
    if report.get("passed") is not True:
        raise RuntimeError("worker release-gate semantic verification did not pass")
    if report.get("release_gate_sha256") != expected_sha256:
        raise RuntimeError("worker release-gate verifier returned a different digest")
    return report

# Only these common configuration fields are constructor arguments.  The
# remaining released-profile invariants are validated before construction and
# retained in run metadata, but must not leak into the upstream JiT kwargs.
_DENOISER_FIELDS = (
    "profile",
    "probe_depth",
    "error_choice",
    "rel_l1_thresh",
    "ret_ratio",
    "gamma_min",
    "gamma_max",
    "force_last_full",
    "numeric_mode",
    "epsilon",
    "nonfinite_policy",
    "gamma_nonfinite_policy",
    "gate_mode",
    "cache_dtype",
    "trace_mode",
)

_SUMMARY_SUM_FIELDS = (
    "total_nfe",
    "total_stream_calls",
    "direct_full_count",
    "resumed_full_count",
    "reuse_count",
    "network_forward_count",
    "probe_count",
    "dcta_count",
    "zero_order_fallback_count",
    "gamma_clip_min_count",
    "gamma_clip_max_count",
    "gamma_nonfinite_count",
    "probe_time_ms",
    "gate_time_ms",
    "scalar_sync_time_ms",
    "dcta_time_ms",
    "suffix_time_ms",
    "cache_io_time_ms",
)
_SUMMARY_MAX_FIELDS = {
    "cache_bytes": "max_cache_bytes",
    "cache_tensor_count": "max_cache_tensor_count",
    "peak_memory_allocated": "max_peak_memory_allocated",
    "peak_memory_reserved": "max_peak_memory_reserved",
}

_PORT_ONLY_TOP_LEVEL_KEYS = (
    "schema_version",
    "checkpoint",
    "dicache",
    "runtime",
    "selection_provenance",
)


def _lightning_cli_base_config(config: dict[str, object]) -> dict[str, object]:
    """Remove port-only metadata before handing the config to LightningCLI."""

    resolved = dict(config)
    for key in _PORT_ONLY_TOP_LEVEL_KEYS:
        resolved.pop(key, None)
    return resolved


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


def _validate_dicache_config(value: dict[str, object]) -> None:
    validate_common_dicache_config(value, require_resolved=True)


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
    parser.add_argument("--release-gate")
    parser.add_argument("--release-gate-sha256")
    parser.add_argument("--nonfinal-proxy", action="store_true")
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.shard_id < args.world_size:
        raise ValueError("shard-id must satisfy 0 <= shard-id < world-size")
    config_path = Path(args.config).resolve(strict=True)
    manifest_path = Path(args.manifest).resolve(strict=True)
    output_root = Path(args.output_root).resolve(strict=not args.nonfinal_proxy)
    worker_release: dict[str, object] | None = None
    if args.nonfinal_proxy:
        if args.release_gate is not None or args.release_gate_sha256 is not None:
            raise ValueError("non-final proxy workers must not accept release-gate identity")
    else:
        if not args.release_gate or not args.release_gate_sha256:
            raise RuntimeError(
                "final shard workers require --release-gate and --release-gate-sha256"
            )
        worker_release = _verify_worker_release(
            gate=args.release_gate,
            expected_sha256=args.release_gate_sha256,
            config=config_path,
            manifest=manifest_path,
            output_root=output_root,
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    if config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError("unsupported config schema; expected pixarc-dicache-config-v1")
    runtime = dict(config["runtime"])
    dicache = dict(config["dicache"])
    _validate_dicache_config(dicache)
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
    if compile_mode == "upstream" and dicache["mode"] != "upstream_full":
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
    records = load_manifest(manifest_path)
    if args.nonfinal_proxy:
        if len(records) >= 50000:
            raise ValueError("non-final proxy workers require fewer than 50000 records")
    elif len(records) != 50000:
        raise ValueError("release-gated shard workers require exactly 50000 records")
    if worker_release is not None:
        if sha256_file(config_path) != worker_release["config_sha256"]:
            raise RuntimeError("archived config changed after worker gate verification")
        if sha256_file(manifest_path) != worker_release["manifest_sha256"]:
            raise RuntimeError("archived manifest changed after worker gate verification")
    validate_manifest(
        records,
        world_size=args.world_size,
        batch_size=batch_size,
    )
    configured_denoiser = dict(dict(config["model"])["denoiser"])
    configured_denoiser_args = dict(configured_denoiser["init_args"])
    configured_resolution = int(configured_denoiser_args.get("input_size", 256))
    manifest_sidecar = manifest_path.with_suffix(
        manifest_path.suffix + ".meta.json"
    )
    if worker_release is not None and (
        sha256_file(manifest_sidecar) != worker_release["manifest_sidecar_sha256"]
    ):
        raise RuntimeError("manifest sidecar changed after worker gate verification")
    validate_manifest_sidecar(
        manifest_path,
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
    if worker_release is not None and (
        str(checkpoint) != worker_release["checkpoint_path"]
        or int(checkpoint_info["checkpoint_size"]) != worker_release["checkpoint_size"]
        or checkpoint_info["checkpoint_sha256"] != worker_release["checkpoint_sha256"]
    ):
        raise RuntimeError("checkpoint changed after worker release-gate verification")
    release_gate_sha256 = archived_release_gate_sha256(output_root)
    if args.nonfinal_proxy:
        if release_gate_sha256 != UNRELEASED_RELEASE_GATE:
            raise ValueError("non-final proxy output root must not contain a release gate")
    elif release_gate_sha256 != args.release_gate_sha256:
        raise RuntimeError("archived release gate changed after worker verification")
    if not args.acknowledge_gpu_job or os.environ.get("DICACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("GPU generation is deferred; explicit safety acknowledgement is required")
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible) != 1:
        raise RuntimeError("one PixelGen shard process must see exactly one CUDA GPU")
    for relative in ("samples", "metadata", "summaries", "logs"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    _archive_input(config_path, output_root / "config_resolved.yaml")
    _archive_input(manifest_path, output_root / "input_manifest.jsonl")
    _archive_input(
        manifest_sidecar, output_root / "input_manifest.jsonl.meta.json"
    )

    model_config = dict(config["model"])
    denoiser = dict(model_config["denoiser"])
    denoiser_args = dict(denoiser["init_args"])
    sampler = dict(model_config["diffusion_sampler"])
    sampler_args = dict(sampler["init_args"])
    expected_denoiser = "dicache_style.pixelgen_model.DiCachePixelGenJiT"
    expected_sampler = "dicache_style.pixelgen_sampler.DiCacheHeunSamplerJiT"
    if denoiser.get("class_path") != expected_denoiser:
        raise ValueError(f"PixelGen DiCache denoiser must be {expected_denoiser}")
    if sampler.get("class_path") != expected_sampler:
        raise ValueError(f"PixelGen DiCache sampler must be {expected_sampler}")
    if not bool(sampler_args.get("exact_henu")):
        raise ValueError("PixelGen DiCache generation requires exact_henu=true")
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
                    if not key.startswith("dicache_") and key != "compile_mode"
                },
            },
        }
    )
    sampler_config_hash = canonical_hash(sampler)
    resolution = int(denoiser_args.get("input_size", 256))
    config_hash = canonical_hash(config)
    dicache_config_hash = canonical_hash(dicache)
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
        dicache_config_hash=dicache_config_hash,
        release_gate_sha256=release_gate_sha256,
        dicache_config={field: dicache[field] for field in DICACHE_CONFIG_FIELDS},
        checkpoint_path=str(checkpoint),
        checkpoint_size=int(checkpoint_info["checkpoint_size"]),
        method=str(dicache["mode"]),
        resolution=resolution,
    )

    resolved = _lightning_cli_base_config(config)
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
                    "class_path": "dicache_style.pixelgen_io.AtomicManifestSaveHook",
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
            f"dicache_{field}": dicache[field]
            for field in _DENOISER_FIELDS
        }
    )
    resolved["model"]["denoiser"]["init_args"].update(
        {"dicache_mode": dicache["mode"], "compile_mode": compile_mode}
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
                "class_path": "dicache_style.pixelgen_io.ManifestNoiseDataset",
                "init_args": {
                    "manifest_path": str(Path(args.manifest).resolve()),
                    "shard_id": args.shard_id,
                    "world_size": args.world_size,
                    "output_root": str(output_root),
                    "batch_size": batch_size,
                    "config_hash": config_hash,
                    "dicache_config_hash": dicache_config_hash,
                    "release_gate_sha256": release_gate_sha256,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_size": int(checkpoint_info["checkpoint_size"]),
                    "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                    "method": dicache["mode"],
                    "dicache_config": {
                        field: dicache[field] for field in DICACHE_CONFIG_FIELDS
                    },
                    "resolution": resolution,
                    "noise_scale": noise_scale,
                    "resume": args.resume,
                },
            },
        }
    )
    invocation_id = os.environ.get("DICACHE_INVOCATION_ID", "direct")
    if not invocation_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in invocation_id
    ):
        raise ValueError("unsafe DICACHE_INVOCATION_ID")
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
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "input_config_hash": config_hash,
        "release_gate_sha256": release_gate_sha256,
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
        "dicache_config_hash": dicache_config_hash,
    }
    run_config.update({field: dicache[field] for field in DICACHE_CONFIG_FIELDS})
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
            method=str(dicache["mode"]),
            config=run_config,
            checkpoint=checkpoint,
            manifest_sha256=run_config["manifest_sha256"],
            git_commit=current_git_revision,
            dicache_commit=DICACHE_COMMIT,
            pixelgen_tree_id=PIXELGEN_TREE_ID,
        )
        value.update({field: dicache[field] for field in DICACHE_CONFIG_FIELDS})
        value["input_config_hash"] = config_hash
        value["dicache_config_hash"] = dicache_config_hash
        value["release_gate_sha256"] = release_gate_sha256
        atomic_create_json(run_manifest_path, value)
    existing_run = load_json(run_manifest_path)
    expected_existing = {
        "config": run_config,
        "manifest_sha256": run_config["manifest_sha256"],
        "manifest_sidecar_sha256": run_config["manifest_sidecar_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": int(checkpoint_info["checkpoint_size"]),
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "method": str(dicache["mode"]),
        "dicache_config_hash": dicache_config_hash,
        "release_gate_sha256": release_gate_sha256,
        "git_commit": current_git_revision,
        "dicache_commit": DICACHE_COMMIT,
        "pixelgen_tree_id": PIXELGEN_TREE_ID,
        "pytorch_version": torch.__version__,
    }
    expected_existing.update({field: dicache[field] for field in DICACHE_CONFIG_FIELDS})
    for key, expected_value in expected_existing.items():
        if existing_run.get(key) != expected_value:
            raise RuntimeError(f"existing run_manifest mismatch: {key}")

    if not pending_groups:
        counts: dict[str, int | float | str | None] = {
            "generated": len(existing_metadata),
            "generated_this_invocation": 0,
            "skipped_groups": len(skipped_group_ids),
            "trajectory_count": 0,
            "shard_id": args.shard_id,
            "invocation_id": os.environ.get("DICACHE_INVOCATION_ID"),
        }
        counts.update({field: 0 for field in _SUMMARY_SUM_FIELDS})
        counts.update({field: 0 for field in _SUMMARY_MAX_FIELDS.values()})
        existing_groups: dict[str, dict[str, object]] = {}
        for row in existing_metadata.values():
            existing_groups.setdefault(str(row["batch_group_id"]), row)
        for row in existing_groups.values():
            counts["trajectory_count"] = int(counts["trajectory_count"]) + 1
            for field in _SUMMARY_SUM_FIELDS:
                counts[field] = float(counts[field]) + float(
                    row.get(f"trajectory_{field}", 0) or 0
                )
            for source, destination in _SUMMARY_MAX_FIELDS.items():
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

    if worker_release is not None:
        current_source = release_source_bindings(LOCAL_ROOT, UPSTREAM_ROOT)
        if (
            current_source["port"]["sha256"]
            != worker_release["port_source_sha256"]
            or current_source["upstream"]["sha256"]
            != worker_release["upstream_source_sha256"]
        ):
            raise RuntimeError(
                "release-critical source changed after worker gate verification"
            )
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
