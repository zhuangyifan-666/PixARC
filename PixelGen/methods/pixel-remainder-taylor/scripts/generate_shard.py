#!/usr/bin/env python3
"""Resolve and execute one PixelGen Pixel-Remainder shard on one visible GPU."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
PIXELGEN_ROOT = Path(__file__).resolve().parents[3]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = PIXARC_ROOT / "third-party" / "PixelGen"
BASELINE_ROOT = PIXELGEN_ROOT / "baselines" / "taylorseer-style"
SHARED_ROOT = PIXARC_ROOT / "JiT" / "methods" / "pixel-remainder-taylor"
for item in (METHOD_ROOT, SHARED_ROOT, BASELINE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pixel_remainder_taylor.config import (  # noqa: E402
    canonical_yaml_bytes,
    immutable_write_bytes,
    validate_archived_config_contract,
)
from pixel_remainder_taylor.finite_difference import (  # noqa: E402
    MAX_INTERPOLATION_WEIGHT_L1,
)
from pixel_remainder_taylor.protocol import (  # noqa: E402
    executable_tree_sha256,
    validate_compatible_manifest_sidecar,
)
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
    build_run_metadata,
    canonical_hash,
    checkpoint_identity,
    git_revision,
    load_json,
    source_tree_sha256,
)


TAYLORSEER_COMMIT = "704ee98c74f7f04da443daa3c0aa2cc7803d86e3"


def _write_yaml(path: Path, value: object) -> None:
    immutable_write_bytes(path, canonical_yaml_bytes(value))


def build_resolved_cli_config(
    config: dict[str, object],
    *,
    trainer_updates: dict[str, object],
    denoiser_updates: dict[str, object],
    dataset_init_args: dict[str, object],
    pred_batch_size: int,
    pred_num_workers: int,
    compile_mode: str,
) -> dict[str, object]:
    """Build the exact LightningCLI prediction config used by a rank."""

    resolved = copy.deepcopy({
        key: value for key, value in config.items()
        if key not in {"schema_version", "template_only", "checkpoint", "method", "runtime"}
    })
    resolved["trainer"] = dict(resolved.get("trainer", {}))
    resolved["trainer"].update(trainer_updates)
    model = dict(resolved["model"])
    denoiser = dict(model["denoiser"])
    denoiser_args = dict(denoiser["init_args"])
    denoiser_args.update(denoiser_updates)
    denoiser["init_args"] = denoiser_args
    model["denoiser"] = denoiser
    model["compile_mode"] = compile_mode
    resolved["model"] = model
    data = dict(resolved["data"])
    data.update({
        "train_dataset": None,
        "eval_dataset": None,
        "pred_batch_size": pred_batch_size,
        "pred_num_workers": pred_num_workers,
        "pred_dataset": {
            "class_path": "pixel_remainder_taylor.pixelgen_io.ManifestNoiseDataset",
            "init_args": copy.deepcopy(dataset_init_args),
        },
    })
    resolved["data"] = data
    return resolved


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

    output = Path(args.output_root).resolve()
    if output == PIXARC_ROOT or PIXARC_ROOT in output.parents:
        raise ValueError("generation output must be outside the PixARC checkout")
    for name in ("samples", "metadata", "summaries", "traces", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve(strict=True)
    config, config_identity = validate_archived_config_contract(
        output, config_path, model="PixelGen"
    )
    runtime = dict(config["runtime"]); method = dict(config["method"])
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
    model_config = dict(config["model"])
    sampler = dict(model_config["diffusion_sampler"])
    sampler_args = dict(sampler["init_args"])
    if sampler_args.get("exact_henu") is not True or int(sampler_args.get("num_steps", 0)) != 50:
        raise ValueError("official runs require exact 50-step Heun")
    checkpoint = Path(str(config["checkpoint"])).resolve(strict=True)
    if not checkpoint.is_absolute():
        raise ValueError("resolved PixelGen checkpoint must be absolute")
    checkpoint_info = checkpoint_identity(checkpoint)
    denoiser = dict(model_config["denoiser"]); denoiser_args = dict(denoiser["init_args"])
    resolution = int(denoiser_args.get("input_size", 256)); batch_size = int(runtime["batch_size"])
    manifest_path = Path(args.manifest).resolve(strict=True)
    if manifest_path != (output / "input_manifest.jsonl").resolve(strict=True):
        raise ValueError("production generator requires launcher-owned input_manifest.jsonl")
    records = load_manifest(manifest_path)
    validate_manifest(records, world_size=args.world_size, batch_size=batch_size)
    sidecar, _sidecar_metadata = validate_compatible_manifest_sidecar(
        manifest_path, records, world_size=args.world_size, batch_size=batch_size,
        validator=validate_manifest_sidecar,
        generator_device="cpu", noise_dtype="float32", noise_shape=(3, resolution, resolution),
    )
    if sidecar.resolve() != (output / "input_manifest.jsonl.meta.json").resolve(strict=True):
        raise ValueError("production generator requires launcher-owned manifest sidecar")

    config_hash = config_identity["semantic_config_hash"]
    manifest_hash = sha256_file(manifest_path)
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
        method=str(method["mode"]), interval=identity_interval,
        max_order=identity_max_order, coordinate_mode=coordinate_mode, resolution=resolution,
    )
    normalized_model = {
        key: value
        for key, value in model_config.items()
        if key not in {"denoiser", "diffusion_sampler", "compile_mode"}
    }
    normalized_model["diffusion_trainer"] = {
        "class_path": "taylorseer_style.pixelgen_lightning.InferenceOnlyTrainer",
        "init_args": {},
    }
    normalized_model["denoiser"] = {
        "class_path": "taylorseer_style.pixelgen_model.TaylorSeerPixelGenJiT",
        "init_args": {
            key: value
            for key, value in denoiser_args.items()
            if not key.startswith(("method_", "debug_")) and key != "compile_mode"
        },
    }
    normalized_sampler = {
        **sampler,
        "class_path": "taylorseer_style.pixelgen_sampler.TaylorSeerHeunSamplerJiT",
    }
    method_source_hash = canonical_hash({
        "shared": executable_tree_sha256(SHARED_ROOT),
        "pixelgen_adapter": executable_tree_sha256(METHOD_ROOT),
    })
    run_config = {
        "model": "PixelGen-JiT",
        "model_config_hash": canonical_hash(normalized_model),
        "ema": "ema_denoiser",
        **config_identity,
        # Compatibility field: this remains the semantic mapping hash.
        "input_config_hash": config_hash,
        "port_source_sha256": source_tree_sha256(BASELINE_ROOT),
        "method_source_sha256": method_source_hash,
        "manifest_sha256": manifest_hash,
        "manifest_sidecar_sha256": sha256_file(sidecar),
        "manifest_records_sha256": manifest_records_sha256(records),
        "initial_noise_protocol": "per-sample torch.Generator(device='cpu') standard Gaussian, dataset noise_scale",
        "rng_device": "cpu",
        "generator_type": "torch.Generator.manual_seed per sample",
        "initial_noise_dtype": "float32",
        "initial_noise_shape": [3, resolution, resolution],
        "noise_scale": float(runtime.get("noise_scale", 1.0)),
        "sampler": "exact_heun",
        "sampler_config_hash": canonical_hash(normalized_sampler),
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
        "batch_grouping": "manifest batch_group_id/position_in_batch",
        "compile_mode": runtime.get("compile_mode", "matched_eager"),
        "coordinate_mode": coordinate_mode,
    }
    run_value = build_run_metadata(
        model="PixelGen-JiT",
        method=str(method["mode"]),
        config=run_config,
        checkpoint=checkpoint,
        manifest_sha256=manifest_hash,
        git_commit=git_revision(PIXARC_ROOT),
        taylorseer_commit=TAYLORSEER_COMMIT,
    )
    run_value.update({
        "method_schema_version": config["schema_version"],
        **config_identity,
        "input_config_hash": config_hash,
        "method_source_sha256": method_source_hash,
        "tau": method.get("tau"),
        "max_taylor_span": method["max_taylor_span"],
        "expected_nfe_per_trajectory": 99,
        "expected_network_forwards_per_trajectory": 99,
        "forward_contract": "one combined 2B CFG forward per NFE",
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
        "cfg_execution": "one combined [unconditional, conditional] 2B forward",
    })
    run_manifest = output / "run_manifest.json"
    if not run_manifest.exists():
        atomic_create_json(run_manifest, run_value)
    existing_run = load_json(run_manifest)
    for key in (
        "config", "input_config_hash", "input_config_sha256",
        "resolved_config_sha256", "semantic_config_hash", "manifest_sha256",
        "manifest_sidecar_sha256", "checkpoint_path", "checkpoint_size",
        "method", "tau", "max_taylor_span", "git_commit",
        "pytorch_version", "python_version", "cuda_version",
        "method_source_sha256", "predictor_backend",
        "max_interpolation_weight_l1", "trace_mode",
        "expected_nfe_per_trajectory", "expected_network_forwards_per_trajectory",
        "real_batch_size", "cfg_execution",
    ):
        if existing_run.get(key) != run_value.get(key):
            raise RuntimeError(f"existing run_manifest mismatch: {key}")
    if not pending:
        result = {
            "generated": len(existing), "generated_this_invocation": 0,
            "skipped_groups": len(skipped), "shard_id": args.shard_id,
            "tau": method.get("tau"), "max_taylor_span": method["max_taylor_span"],
        }
        summary_path = output / "summaries" / f"rank_{args.shard_id}_summary.json"
        if not summary_path.exists():
            atomic_write_json(summary_path, result)
        print(json.dumps(result, indent=2, sort_keys=True)); return

    resolved = build_resolved_cli_config(
        config,
        trainer_updates={
        "default_root_dir": str(output / "lightning" / f"rank_{args.shard_id}"),
        "accelerator": "gpu", "devices": 1, "strategy": "auto", "logger": False,
        "precision": runtime.get("precision", "bf16-mixed"), "use_distributed_sampler": False,
        "callbacks": [{
            "class_path": "pixel_remainder_taylor.pixelgen_io.AtomicManifestSaveHook",
            "init_args": {"output_root": str(output), "shard_id": args.shard_id,
                          "resolution": resolution, "resume": args.resume},
        }],
        },
        denoiser_updates={
        "method_mode": method["mode"], "method_tau": method.get("tau"),
        "method_max_taylor_span": method["max_taylor_span"],
        "method_cache_dtype": method.get("cache_dtype", "inherit"),
        "method_trace_mode": method.get("trace_mode", "full"),
        "debug_fixed_interval": debug.get("interval"), "debug_fixed_order": debug.get("order"),
        "compile_mode": runtime.get("compile_mode", "matched_eager"),
        },
        dataset_init_args={
            "manifest_path": str(manifest_path), "shard_id": args.shard_id,
            "world_size": args.world_size, "output_root": str(output), "batch_size": batch_size,
            "config_hash": config_hash, "checkpoint_path": str(checkpoint),
            **config_identity,
            "checkpoint_size": int(checkpoint_info["checkpoint_size"]),
            "method": method["mode"], "interval": identity_interval,
            "max_order": identity_max_order, "coordinate_mode": coordinate_mode,
            "resolution": resolution, "noise_scale": float(runtime.get("noise_scale", 1.0)),
            "resume": args.resume,
        },
        pred_batch_size=batch_size,
        pred_num_workers=int(runtime.get("num_workers", 1)),
        compile_mode=str(runtime.get("compile_mode", "matched_eager")),
    )
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
