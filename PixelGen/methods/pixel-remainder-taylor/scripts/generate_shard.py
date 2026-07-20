#!/usr/bin/env python3
"""Resolve and execute one PixelGen Pixel-Remainder shard on one visible GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parents[1]
PIXELGEN_ROOT = Path(__file__).resolve().parents[3]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = PIXARC_ROOT / "third-party" / "PixelGen"
BASELINE_ROOT = PIXELGEN_ROOT / "baselines" / "taylorseer-style"
SHARED_ROOT = PIXARC_ROOT / "JiT" / "methods" / "pixel-remainder-taylor"
for item in (METHOD_ROOT, SHARED_ROOT, BASELINE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pixel_remainder_taylor.config import load_config  # noqa: E402
from taylorseer_style.image_io import load_metadata_jsonl, resumable_batch_groups  # noqa: E402
from taylorseer_style.manifest import (  # noqa: E402
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
    validate_manifest_sidecar,
)
from taylorseer_style.metadata import (  # noqa: E402
    atomic_create_json,
    atomic_write_json,
    canonical_hash,
    checkpoint_identity,
    git_revision,
    load_json,
)


def _checkpoint(value: str, origin: Path) -> Path:
    candidate = Path(value).expanduser()
    choices = [candidate] if candidate.is_absolute() else [origin / candidate, PIXARC_ROOT / candidate]
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"checkpoint not found; checked {choices}")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable run input differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_yaml(path: Path, value: object) -> None:
    content = yaml.safe_dump(value, sort_keys=False).encode("utf-8")
    _atomic_bytes(path, content)


def _trace_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-origin-dir")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acknowledge-gpu-job", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.world_size:
        raise ValueError("shard-id must satisfy 0 <= shard-id < world-size")
    if not args.acknowledge_gpu_job or os.environ.get("PIXEL_REMAINDER_GPU_RUN_ALLOWED") != "1":
        raise RuntimeError("set PIXEL_REMAINDER_GPU_RUN_ALLOWED=1 and acknowledge the GPU job")
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible) != 1:
        raise RuntimeError("one shard process must see exactly one CUDA GPU")

    config_path = Path(args.config).resolve(strict=True)
    config = load_config(config_path)
    runtime = dict(config["runtime"]); method = dict(config["method"])
    model_config = dict(config["model"])
    sampler = dict(model_config["diffusion_sampler"])
    sampler_args = dict(sampler["init_args"])
    if sampler_args.get("exact_henu") is not True or int(sampler_args.get("num_steps", 0)) != 50:
        raise ValueError("official runs require exact 50-step Heun")
    origin = Path(args.config_origin_dir).resolve(strict=True) if args.config_origin_dir else config_path.parent
    checkpoint = _checkpoint(str(config["checkpoint"]), origin)
    checkpoint_info = checkpoint_identity(checkpoint)
    denoiser = dict(model_config["denoiser"]); denoiser_args = dict(denoiser["init_args"])
    resolution = int(denoiser_args.get("input_size", 256)); batch_size = int(runtime["batch_size"])
    records = load_manifest(args.manifest)
    validate_manifest(records, world_size=args.world_size, batch_size=batch_size)
    sidecar = Path(args.manifest).with_suffix(Path(args.manifest).suffix + ".meta.json")
    validate_manifest_sidecar(
        args.manifest, records, world_size=args.world_size, batch_size=batch_size,
        generator_device="cpu", noise_dtype="float32", noise_shape=(3, resolution, resolution),
    )
    output = Path(args.output_root).resolve()
    if output == PIXARC_ROOT or PIXARC_ROOT in output.parents:
        raise ValueError("generation output must be outside the PixARC checkout")
    for name in ("samples", "metadata", "summaries", "traces", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    _write_yaml(output / "config_resolved.yaml", config)
    _atomic_bytes(output / "input_manifest.jsonl", Path(args.manifest).resolve().read_bytes())
    _atomic_bytes(output / "input_manifest.jsonl.meta.json", sidecar.resolve().read_bytes())

    config_hash = canonical_hash(config); manifest_hash = sha256_file(args.manifest)
    metadata_path = output / "metadata" / f"rank_{args.shard_id}.jsonl"
    trace_path = output / "traces" / f"rank_{args.shard_id}.jsonl"
    if (metadata_path.exists() or trace_path.exists()) and not args.resume:
        raise FileExistsError("rank outputs already exist; use resume only after validation")
    existing = load_metadata_jsonl(metadata_path) if metadata_path.exists() else {}
    existing_groups = {str(row["batch_group_id"]) for row in existing.values()}
    if _trace_count(trace_path) != len(existing_groups):
        raise RuntimeError("durable trajectory trace count does not match completed batch groups")
    pending, skipped = resumable_batch_groups(
        records, args.shard_id, output / "samples", existing,
        manifest_sha256=manifest_hash, config_hash=config_hash,
        checkpoint_path=str(checkpoint), checkpoint_size=int(checkpoint_info["checkpoint_size"]),
        method=str(method["mode"]), interval=int(method["max_taylor_span"]) + 1,
        max_order=2, coordinate_mode="pixel_remainder_dynamic_v1", resolution=resolution,
    )
    run_value = {
        "schema_version": config["schema_version"], "model": "PixelGen-JiT",
        "method": method["mode"], "tau": method.get("tau"),
        "max_taylor_span": method["max_taylor_span"], "config_hash": config_hash,
        "manifest_sha256": manifest_hash, "manifest_records_sha256": manifest_records_sha256(records),
        "checkpoint_path": str(checkpoint), "checkpoint_size": int(checkpoint_info["checkpoint_size"]),
        "git_commit": git_revision(PIXARC_ROOT), "world_size": args.world_size,
        "batch_size": batch_size, "expected_nfe_per_trajectory": 99,
        "expected_network_forwards_per_trajectory": 99,
        "forward_contract": "one combined 2B CFG forward per NFE",
    }
    run_manifest = output / "run_manifest.json"
    if not run_manifest.exists():
        atomic_create_json(run_manifest, run_value)
    if load_json(run_manifest) != run_value:
        raise RuntimeError("existing run_manifest does not match this invocation")
    if not pending:
        result = {
            "generated": len(existing), "generated_this_invocation": 0,
            "skipped_groups": len(skipped), "shard_id": args.shard_id,
            "tau": method.get("tau"), "max_taylor_span": method["max_taylor_span"],
        }
        atomic_write_json(output / "summaries" / f"rank_{args.shard_id}_summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True)); return

    resolved = {
        key: value for key, value in config.items()
        if key not in {"schema_version", "template_only", "checkpoint", "method", "runtime"}
    }
    resolved["trainer"] = dict(resolved.get("trainer", {}))
    resolved["trainer"].update({
        "default_root_dir": str(output / "lightning" / f"rank_{args.shard_id}"),
        "accelerator": "gpu", "devices": 1, "strategy": "auto", "logger": False,
        "precision": runtime.get("precision", "bf16-mixed"), "use_distributed_sampler": False,
        "callbacks": [{
            "class_path": "pixel_remainder_taylor.pixelgen_io.AtomicManifestSaveHook",
            "init_args": {"output_root": str(output), "shard_id": args.shard_id,
                          "resolution": resolution, "resume": args.resume},
        }],
    })
    debug = dict(method.get("debug", {}))
    denoiser_args.update({
        "method_mode": method["mode"], "method_tau": method.get("tau"),
        "method_max_taylor_span": method["max_taylor_span"],
        "method_cache_dtype": method.get("cache_dtype", "inherit"),
        "method_trace_mode": method.get("trace_mode", "full"),
        "debug_fixed_interval": debug.get("interval"), "debug_fixed_order": debug.get("order"),
        "compile_mode": runtime.get("compile_mode", "matched_eager"),
    })
    denoiser["init_args"] = denoiser_args
    model_config["denoiser"] = denoiser; model_config["diffusion_sampler"] = sampler
    model_config["compile_mode"] = runtime.get("compile_mode", "matched_eager")
    resolved["model"] = model_config
    resolved["data"] = dict(resolved["data"])
    resolved["data"].update({
        "train_dataset": None, "eval_dataset": None, "pred_batch_size": batch_size,
        "pred_num_workers": int(runtime.get("num_workers", 1)),
        "pred_dataset": {
            "class_path": "pixel_remainder_taylor.pixelgen_io.ManifestNoiseDataset",
            "init_args": {
                "manifest_path": str(Path(args.manifest).resolve()), "shard_id": args.shard_id,
                "world_size": args.world_size, "output_root": str(output), "batch_size": batch_size,
                "config_hash": config_hash, "checkpoint_path": str(checkpoint),
                "checkpoint_size": int(checkpoint_info["checkpoint_size"]),
                "method": method["mode"], "interval": int(method["max_taylor_span"]) + 1,
                "max_order": 2, "coordinate_mode": "pixel_remainder_dynamic_v1",
                "resolution": resolution, "noise_scale": float(runtime.get("noise_scale", 1.0)),
                "resume": args.resume,
            },
        },
    })
    invocation = os.environ.get("PIXEL_REMAINDER_INVOCATION_ID", "direct")
    if not invocation or not all(char.isalnum() or char in "._-" for char in invocation):
        raise ValueError("unsafe PIXEL_REMAINDER_INVOCATION_ID")
    resolved_path = output / "metadata" / f"rank_{args.shard_id}_resolved_{invocation}.yaml"
    _write_yaml(resolved_path, resolved)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(UPSTREAM_ROOT), str(METHOD_ROOT), str(SHARED_ROOT), str(BASELINE_ROOT), environment.get("PYTHONPATH", "")]
    )
    command = [
        sys.executable, str(Path(__file__).with_name("pixelgen_main.py")), "predict",
        "-c", str(resolved_path), "--ckpt_path", str(checkpoint),
    ]
    completed = subprocess.run(command, cwd=UPSTREAM_ROOT, env=environment)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
