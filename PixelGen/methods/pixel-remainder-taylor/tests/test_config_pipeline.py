from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pixel_remainder_taylor.config import (
    immutable_write_bytes,
    materialize_config,
    validate_archived_config_contract,
    validate_resolved_config,
)
from snapshot_input import atomic_snapshot


PIXELGEN_CONFIGS = (
    "pixelgen_xl_256_instrumented_full.yaml",
    "pixelgen_xl_256_fixed_i3_k2.yaml",
    "pixelgen_xl_256_prt_t0p01_h3.yaml",
    "pixelgen_xl_256_prt_t0p02_h3.yaml",
    "pixelgen_xl_256_prt_t0p04_h3.yaml",
)


def _config_source_root() -> Path:
    checkout = Path(__file__).resolve().parents[4]
    configured = os.environ.get("PIXEL_REMAINDER_CONFIG_SOURCE")
    candidates = [Path(configured)] if configured else []
    candidates.extend((checkout, checkout.parent / "PixARC"))
    for candidate in candidates:
        checkpoint = candidate / "PixelGen/checkpoints/PixelGen_XL_160ep.ckpt"
        if checkpoint.is_file():
            return candidate.resolve()
    raise FileNotFoundError("a checkout containing the real PixelGen checkpoint is required")


@pytest.mark.parametrize("name", PIXELGEN_CONFIGS)
def test_real_pixelgen_extends_materializes_portably(tmp_path: Path, name: str):
    source = (
        _config_source_root()
        / "PixelGen/methods/pixel-remainder-taylor/configs"
        / name
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert raw["extends"] == "pixelgen_xl_256_base.yaml"
    run = tmp_path / "run"
    atomic_snapshot(source, run / "input_config.yaml")
    config = materialize_config(
        source, run / "config_resolved.yaml", model="PixelGen"
    )
    resolved_raw = yaml.safe_load(
        (run / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    assert "extends" not in resolved_raw
    assert config["template_only"] is False
    assert {
        "trainer", "model", "data", "runtime", "method", "checkpoint"
    } <= set(config)
    assert Path(config["checkpoint"]).is_absolute()
    assert Path(config["checkpoint"]).is_file()
    first = (run / "config_resolved.yaml").read_bytes()
    materialize_config(source, run / "config_resolved.yaml", model="PixelGen")
    assert (run / "config_resolved.yaml").read_bytes() == first
    loaded, identity = validate_archived_config_contract(
        run, run / "config_resolved.yaml", model="PixelGen"
    )
    assert loaded == config
    assert set(identity) == {
        "input_config_sha256",
        "resolved_config_sha256",
        "semantic_config_hash",
    }
    isolated = tmp_path / "arbitrary-directory/config_resolved.yaml"
    immutable_write_bytes(isolated, first)
    assert validate_resolved_config(isolated, model="PixelGen") == config
