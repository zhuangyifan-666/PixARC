from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pixel_remainder_taylor.config import (
    canonical_yaml_bytes,
    immutable_write_bytes,
    materialize_config,
    semantic_config_sha256,
    validate_archived_config_contract,
    validate_resolved_config,
)
from preflight_run import preflight_run
from snapshot_input import atomic_snapshot


JIT_CONFIGS = (
    "jit_b16_256_instrumented_full.yaml",
    "jit_b16_256_fixed_i3_k2.yaml",
    "jit_b16_256_prt_t0p01_h3.yaml",
    "jit_b16_256_prt_t0p02_h3.yaml",
    "jit_b16_256_prt_t0p04_h3.yaml",
)


def _config_source_root() -> Path:
    checkout = Path(__file__).resolve().parents[4]
    configured = os.environ.get("PIXEL_REMAINDER_CONFIG_SOURCE")
    candidates = [Path(configured)] if configured else []
    candidates.extend((checkout, checkout.parent / "PixARC"))
    for candidate in candidates:
        checkpoint = candidate / "JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth"
        if checkpoint.is_file():
            return candidate.resolve()
    raise FileNotFoundError("a checkout containing the real JiT checkpoint is required")


@pytest.mark.parametrize("name", JIT_CONFIGS)
def test_real_jit_config_end_to_end_is_idempotent_and_portable(
    tmp_path: Path, name: str
):
    source = _config_source_root() / "JiT/methods/pixel-remainder-taylor/configs" / name
    run = tmp_path / "run"
    atomic_snapshot(source, run / "input_config.yaml")
    config = materialize_config(
        source, run / "config_resolved.yaml", model="JiT"
    )
    assert source.read_bytes() != (run / "config_resolved.yaml").read_bytes()
    assert config["template_only"] is False
    assert Path(config["model"]["checkpoint"]).is_absolute()
    assert Path(config["model"]["checkpoint"]).is_file()
    first = (run / "config_resolved.yaml").read_bytes()
    materialize_config(source, run / "config_resolved.yaml", model="JiT")
    assert (run / "config_resolved.yaml").read_bytes() == first
    loaded, identity = validate_archived_config_contract(
        run, run / "config_resolved.yaml", model="JiT"
    )
    assert loaded == config
    assert identity["semantic_config_hash"] == semantic_config_sha256(config)
    isolated = tmp_path / "isolated/config_resolved.yaml"
    immutable_write_bytes(isolated, first)
    assert validate_resolved_config(isolated, model="JiT") == config
    manifest = (
        Path(__file__).resolve().parents[4]
        / "results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl"
    )
    preflight = preflight_run(
        model="JiT",
        input_config=source,
        manifest=manifest,
        expected_count=1000,
        require_clean=False,
    )
    assert preflight["status"] == "PASS"
    assert preflight["gpu_queries"] == 0


def _minimal_pixelgen_parent(checkpoint: Path, *, tau: float = 0.01) -> dict:
    return {
        "schema_version": "pixarc-pixel-remainder-taylor-v1",
        "template_only": True,
        "checkpoint": str(checkpoint),
        "method": {
            "mode": "pixel_remainder_taylor",
            "tau": tau,
            "max_taylor_span": 3,
            "stored_feature_order": 2,
            "pixel_max_order": 3,
            "warmup_full_nfe": 3,
            "pool_kernel": 8,
            "batch_reduction": "mean",
            "cache_dtype": "inherit",
            "trace_mode": "full",
        },
        "runtime": {"batch_size": 4},
        "trainer": {},
        "model": {},
        "data": {},
    }


def test_raw_bytes_and_parent_semantics_are_both_immutable(tmp_path: Path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    configs = tmp_path / "configs"
    configs.mkdir()
    parent = configs / "base.yaml"
    child = configs / "child.yaml"
    parent.write_bytes(canonical_yaml_bytes(_minimal_pixelgen_parent(checkpoint)))
    child.write_text(
        "extends: base.yaml\ntemplate_only: false\nmethod: {tau: 0.02}\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"
    atomic_snapshot(child, run / "input_config.yaml")
    materialize_config(child, run / "config_resolved.yaml", model="PixelGen")

    original_child = child.read_bytes()
    child.write_bytes(original_child + b"# formatting-only change\n")
    with pytest.raises(FileExistsError, match="immutable run input differs"):
        atomic_snapshot(child, run / "input_config.yaml")
    child.write_bytes(original_child)

    changed_parent = _minimal_pixelgen_parent(checkpoint, tau=0.04)
    changed_parent["runtime"]["batch_size"] = 8
    parent.write_bytes(canonical_yaml_bytes(changed_parent))
    with pytest.raises(FileExistsError, match="immutable run input differs"):
        materialize_config(child, run / "config_resolved.yaml", model="PixelGen")


def test_tampered_resolved_archive_fails_closed(tmp_path: Path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "config.yaml"
    config = _minimal_pixelgen_parent(checkpoint)
    config["template_only"] = False
    source.write_bytes(canonical_yaml_bytes(config))
    run = tmp_path / "run"
    atomic_snapshot(source, run / "input_config.yaml")
    materialize_config(source, run / "config_resolved.yaml", model="PixelGen")
    parsed = yaml.safe_load((run / "config_resolved.yaml").read_text())
    parsed["method"]["tau"] = 0.04
    (run / "config_resolved.yaml").write_bytes(canonical_yaml_bytes(parsed))
    with pytest.raises(FileExistsError, match="immutable run input differs"):
        materialize_config(source, run / "config_resolved.yaml", model="PixelGen")
