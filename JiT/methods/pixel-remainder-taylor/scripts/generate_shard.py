#!/usr/bin/env python3
"""Generate one deterministic JiT Pixel-Remainder shard on one visible GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parents[1]
JIT_ROOT = Path(__file__).resolve().parents[3]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = JIT_ROOT / "baselines" / "taylorseer-style"
UPSTREAM_ROOT = PIXARC_ROOT / "third-party" / "JiT"
for item in (UPSTREAM_ROOT, METHOD_ROOT, BASELINE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pixel_remainder_taylor.config import load_config  # noqa: E402
from pixel_remainder_taylor.finite_difference import (  # noqa: E402
    MAX_INTERPOLATION_WEIGHT_L1,
)
from pixel_remainder_taylor.jit_denoiser import PixelRemainderTaylorDenoiser  # noqa: E402
from pixel_remainder_taylor.protocol import (  # noqa: E402
    executable_tree_sha256,
    validate_compatible_manifest_sidecar,
)
from taylorseer_style.image_io import (  # noqa: E402
    atomic_write_png,
    image_path,
    load_metadata_jsonl,
    resumable_batch_groups,
)
from taylorseer_style.manifest import (  # noqa: E402
    initial_noise,
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
    validate_manifest_sidecar,
)
from taylorseer_style.metadata import (  # noqa: E402
    atomic_create_json,
    atomic_write_json,
    build_run_metadata,
    canonical_hash,
    checkpoint_identity,
    git_revision,
    load_json,
    source_tree_sha256,
)


TAYLORSEER_COMMIT = "704ee98c74f7f04da443daa3c0aa2cc7803d86e3"


def _args(config: dict[str, object]) -> SimpleNamespace:
    model = dict(config["model"])
    sampling = dict(config["sampling"])
    extra = dict(model.get("args", {}))
    interval = sampling.get("guidance_interval", [0.1, 1.0])
    return SimpleNamespace(
        model=model["variant"], img_size=int(model.get("image_size", 256)),
        class_num=int(model.get("num_classes", 1000)),
        attn_dropout=float(extra.get("attn_dropout", 0.0)),
        proj_dropout=float(extra.get("proj_dropout", 0.0)),
        label_drop_prob=float(extra.get("label_drop_prob", 0.1)),
        P_mean=float(extra.get("P_mean", -0.8)), P_std=float(extra.get("P_std", 0.8)),
        t_eps=float(extra.get("t_eps", 0.05)),
        noise_scale=float(sampling.get("noise_scale", 1.0)),
        ema_decay1=float(extra.get("ema_decay1", 0.9999)),
        ema_decay2=float(extra.get("ema_decay2", 0.9996)),
        sampling_method=str(sampling["method"]),
        num_sampling_steps=int(sampling["steps"]), cfg=float(sampling["cfg_scale"]),
        interval_min=float(interval[0]), interval_max=float(interval[1]),
    )


def _checkpoint(value: str, config_path: Path, origin: Path) -> Path:
    candidate = Path(value).expanduser()
    choices = [candidate] if candidate.is_absolute() else [origin / candidate, PIXARC_ROOT / candidate]
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"checkpoint not found; checked {choices}")


def _load_ema1(model: PixelRemainderTaylorDenoiser, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if not all(key in state for key in ("model", "model_ema1")):
        raise KeyError("JiT checkpoint must contain model and model_ema1")
    model.load_state_dict(state["model"], strict=True)
    ema = state["model_ema1"]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema:
                raise KeyError(f"EMA1 is missing parameter {name}")
            parameter.copy_(ema[name])


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


def _append(handle, value: dict[str, object]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush(); os.fsync(handle.fileno())


def _assert_external(path: Path) -> None:
    if path == PIXARC_ROOT or PIXARC_ROOT in path.parents:
        raise ValueError("generation output must be outside the PixARC checkout")


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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one shard process must see exactly one CUDA GPU")

    config_path = Path(args.config).resolve(strict=True)
    config = load_config(config_path)
    runtime = dict(config["runtime"]); method = dict(config["method"])
    sampling = dict(config["sampling"]); model_config = dict(config["model"])
    debug = dict(method.get("debug", {}))
    fixed_parity = method["mode"] == "fixed_schedule_parity"
    identity_interval = (
        int(debug["interval"])
        if fixed_parity
        else int(method["max_taylor_span"]) + 1
    )
    identity_max_order = int(debug["order"]) if fixed_parity else 2
    coordinate_mode = (
        "pixel_remainder_legacy_parity_v1"
        if fixed_parity
        else "pixel_remainder_nonuniform_v1"
    )
    if sampling.get("method") != "heun" or sampling.get("exact_heun") is not True:
        raise ValueError("official runs require exact 50-step Heun")
    if int(sampling.get("steps", 0)) != 50:
        raise ValueError("official runs require 50 sampling steps")
    origin = Path(args.config_origin_dir).resolve(strict=True) if args.config_origin_dir else config_path.parent
    checkpoint = _checkpoint(str(model_config["checkpoint"]), config_path, origin)
    identity = checkpoint_identity(checkpoint)
    records = load_manifest(args.manifest)
    batch_size = int(runtime["batch_size"]); resolution = int(model_config.get("image_size", 256))
    validate_manifest(records, world_size=args.world_size, batch_size=batch_size)
    sidecar, _sidecar_metadata = validate_compatible_manifest_sidecar(
        args.manifest, records, world_size=args.world_size, batch_size=batch_size,
        validator=validate_manifest_sidecar,
        generator_device="cuda", noise_dtype="float32", noise_shape=(3, resolution, resolution),
    )
    output = Path(args.output_root).resolve(); _assert_external(output)
    for name in ("samples", "metadata", "summaries", "traces"):
        (output / name).mkdir(parents=True, exist_ok=True)
    resolved_yaml = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    _atomic_bytes(output / "config_resolved.yaml", resolved_yaml)
    _atomic_bytes(output / "input_manifest.jsonl", Path(args.manifest).resolve().read_bytes())
    _atomic_bytes(output / "input_manifest.jsonl.meta.json", sidecar.resolve().read_bytes())

    config_hash = canonical_hash(config); manifest_hash = sha256_file(args.manifest)
    metadata_path = output / "metadata" / f"rank_{args.shard_id}.jsonl"
    trace_path = output / "traces" / f"rank_{args.shard_id}.jsonl"
    if (metadata_path.exists() or trace_path.exists()) and not args.resume:
        raise FileExistsError("rank outputs already exist; use resume only after validation")
    prior = load_metadata_jsonl(metadata_path) if metadata_path.exists() else {}
    prior_groups = {str(row["batch_group_id"]) for row in prior.values()}
    if _trace_count(trace_path) != len(prior_groups):
        raise RuntimeError("durable trajectory trace count does not match completed batch groups")
    # The legacy resume checker needs fixed fields. They are compatibility-only;
    # config_hash plus the explicit fields below define the adaptive run identity.
    groups, skipped = resumable_batch_groups(
        records, args.shard_id, output / "samples", prior,
        manifest_sha256=manifest_hash, config_hash=config_hash,
        checkpoint_path=str(checkpoint), checkpoint_size=int(identity["checkpoint_size"]),
        method=str(method["mode"]), interval=identity_interval,
        max_order=identity_max_order, coordinate_mode=coordinate_mode, resolution=resolution,
    )
    run_config = {
        "model": model_config["variant"],
        "model_config_hash": canonical_hash(
            {**model_config, "checkpoint": str(checkpoint)}
        ),
        "ema": "EMA1",
        "input_config_hash": config_hash,
        "port_source_sha256": source_tree_sha256(BASELINE_ROOT),
        "method_source_sha256": executable_tree_sha256(METHOD_ROOT),
        "manifest_sha256": manifest_hash,
        "manifest_sidecar_sha256": sha256_file(sidecar),
        "manifest_records_sha256": manifest_records_sha256(records),
        "initial_noise_protocol": "per-sample torch.Generator(device='cuda') standard Gaussian; JiT noise_scale applied by config",
        "rng_device": "cuda",
        "generator_type": "torch.Generator.manual_seed per sample",
        "initial_noise_dtype": "float32",
        "initial_noise_shape": [3, resolution, resolution],
        "noise_scale": float(sampling.get("noise_scale", 1.0)),
        "sampler": str(sampling["method"]),
        "sampler_config_hash": canonical_hash(sampling),
        "steps": int(sampling["steps"]),
        "cfg_scale": float(sampling["cfg_scale"]),
        "guidance_interval": list(sampling["guidance_interval"]),
        "timeshift": None,
        "dtype": str(sampling["dtype"]),
        "resolution": resolution,
        "image_postprocessing": "(x+1)/2; clip; round(x*255); RGB uint8 PNG",
        "world_size": args.world_size,
        "batch_size": batch_size,
        "batch_grouping": "manifest batch_group_id/position_in_batch",
        "compile_mode": runtime.get("compile_mode", "matched_eager"),
        "coordinate_mode": coordinate_mode,
    }
    manifest_value = build_run_metadata(
        model=model_config["variant"],
        method=str(method["mode"]),
        config=run_config,
        checkpoint=checkpoint,
        manifest_sha256=manifest_hash,
        git_commit=git_revision(PIXARC_ROOT),
        taylorseer_commit=TAYLORSEER_COMMIT,
    )
    manifest_value.update({
        "method_schema_version": config["schema_version"],
        "input_config_hash": config_hash,
        "method_source_sha256": run_config["method_source_sha256"],
        "tau": method.get("tau"),
        "max_taylor_span": method["max_taylor_span"],
        "expected_nfe_per_trajectory": 99,
        "expected_network_forwards_per_trajectory": 198,
        "forward_contract": "two separate B-sized CFG forwards per NFE",
        "identity_interval": identity_interval,
        "identity_max_order": identity_max_order,
        "coordinate_mode": coordinate_mode,
        "predictor_backend": (
            "legacy_recursive" if fixed_parity else "nonuniform_polynomial"
        ),
        "max_interpolation_weight_l1": MAX_INTERPOLATION_WEIGHT_L1,
        "trace_mode": method.get("trace_mode", "full"),
        "cuda_version": torch.version.cuda,
        "real_batch_size": batch_size,
        "cfg_execution": "two separate B-sized conditional/unconditional forwards",
    })
    run_manifest = output / "run_manifest.json"
    if not run_manifest.exists():
        atomic_create_json(run_manifest, manifest_value)
    existing_run = load_json(run_manifest)
    for key in (
        "config", "input_config_hash", "manifest_sha256",
        "manifest_sidecar_sha256", "checkpoint_path", "checkpoint_size",
        "method", "tau", "max_taylor_span", "git_commit",
        "pytorch_version", "python_version", "cuda_version",
        "method_source_sha256", "predictor_backend",
        "max_interpolation_weight_l1", "trace_mode",
        "expected_nfe_per_trajectory", "expected_network_forwards_per_trajectory",
        "real_batch_size", "cfg_execution",
    ):
        if existing_run.get(key) != manifest_value.get(key):
            raise RuntimeError(f"existing run_manifest mismatch: {key}")

    if not groups:
        result = {
            "generated": len(prior),
            "generated_this_invocation": 0,
            "skipped_groups": len(skipped),
            "shard_id": args.shard_id,
            "tau": method.get("tau"),
            "max_taylor_span": method["max_taylor_span"],
        }
        summary_path = output / "summaries" / f"rank_{args.shard_id}_summary.json"
        if not summary_path.exists():
            atomic_write_json(summary_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    denoiser_args = _args(config)
    model = PixelRemainderTaylorDenoiser(
        denoiser_args, mode=str(method["mode"]), tau=method.get("tau"),
        max_taylor_span=int(method["max_taylor_span"]),
        cache_dtype=str(method.get("cache_dtype", "inherit")),
        trace_mode=str(method.get("trace_mode", "full")),
        compile_mode=str(runtime.get("compile_mode", "matched_eager")),
        debug_fixed_interval=debug.get("interval"), debug_fixed_order=debug.get("order"),
    )
    if str(runtime.get("compile_mode", "matched_eager")) != "matched_eager":
        model.net.compile()
    _load_ema1(model, checkpoint)
    model = model.cuda().eval()
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(sampling.get("dtype")))
    if autocast_dtype is None:
        raise ValueError("sampling.dtype must be bfloat16 or float16")

    counts: dict[str, int | float] = {
        "generated": len(prior), "generated_this_invocation": 0,
        "skipped_groups": len(skipped), "trajectory_count_this_invocation": 0,
        "total_nfe_this_invocation": 0, "full_nfe_this_invocation": 0,
        "taylor_nfe_this_invocation": 0, "network_forwards_this_invocation": 0,
    }
    metadata_mode = "a" if args.resume else "x"; trace_mode = "a" if args.resume else "x"
    with metadata_path.open(metadata_mode, encoding="utf-8") as metadata_handle, trace_path.open(trace_mode, encoding="utf-8") as trace_handle:
        for group in groups:
            labels = torch.tensor([row.class_id for row in group], device="cuda", dtype=torch.long)
            noise = initial_noise([row.seed for row in group], (3, resolution, resolution), device="cuda", dtype=torch.float32)
            trajectory_id = f"shard-{args.shard_id}-group-{group[0].batch_group_id}"
            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                images = model.generate(labels, noise=noise, sample_ids=[row.sample_id for row in group], trajectory_id=trajectory_id)
            if not bool(torch.isfinite(images).all()):
                raise FloatingPointError("JiT generated non-finite pixels")
            summary = model._last_pixel_remainder_summary
            if not isinstance(summary, dict) or summary.get("call_count_valid") is not True:
                raise RuntimeError("complete dynamic trajectory summary is missing")
            if int(summary["network_forward_count"]) != 198:
                raise RuntimeError("JiT must execute exactly 198 model forwards per trajectory")
            arrays = np.round(np.clip((images.float().cpu().numpy() + 1.0) / 2.0, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)
            trajectory = {
                "total_nfe": summary["total_nfe"], "full_nfe": summary["full_nfe"],
                "taylor_nfe": summary["taylor_nfe"], "network_forwards": summary["network_forward_count"],
                "history_update_time_ms": summary.get("history_update_time_ms", 0.0),
                "forecast_time_ms": summary.get("forecast_time_ms", 0.0),
                "scheduler_time_ms": summary.get("controller_time_ms", 0.0),
                "max_forecast_horizon": summary.get("max_planned_span", 0),
                "cache_bytes": summary.get("cache_bytes", 0), "cache_tensor_count": summary.get("cache_tensor_count", 0),
                "peak_memory_allocated": summary.get("peak_memory_allocated", 0),
                "peak_memory_reserved": summary.get("peak_memory_reserved", 0),
            }
            for row, array in zip(group, arrays, strict=True):
                atomic_write_png(array, image_path(output / "samples", row.sample_id), resolution=resolution)
                _append(metadata_handle, {
                    "sample_id": row.sample_id, "class_id": row.class_id, "seed": row.seed,
                    "batch_group_id": row.batch_group_id, "position_in_batch": row.position_in_batch,
                    "manifest_sha256": manifest_hash, "config_hash": config_hash,
                    "checkpoint_path": str(checkpoint), "checkpoint_size": int(identity["checkpoint_size"]),
                    "method": method["mode"], "interval": identity_interval,
                    "max_order": identity_max_order, "coordinate_mode": coordinate_mode,
                    "tau": method.get("tau"), "max_taylor_span": method["max_taylor_span"],
                    **{f"trajectory_{key}": value for key, value in trajectory.items()}, "status": "ok",
                })
                counts["generated"] += 1; counts["generated_this_invocation"] += 1
            _append(trace_handle, summary)
            counts["trajectory_count_this_invocation"] += 1
            counts["total_nfe_this_invocation"] += int(summary["total_nfe"])
            counts["full_nfe_this_invocation"] += int(summary["full_nfe"])
            counts["taylor_nfe_this_invocation"] += int(summary["taylor_nfe"])
            counts["network_forwards_this_invocation"] += int(summary["network_forward_count"])
    result = {**counts, "shard_id": args.shard_id, "tau": method.get("tau"), "max_taylor_span": method["max_taylor_span"]}
    atomic_write_json(output / "summaries" / f"rank_{args.shard_id}_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
