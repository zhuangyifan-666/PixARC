"""Manifest I/O plus durable full dynamic traces."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_PIXELGEN_ROOT = Path(__file__).resolve().parents[3]
_TAYLOR_BASE = _PIXELGEN_ROOT / "baselines" / "taylorseer-style"
if str(_TAYLOR_BASE) not in sys.path:
    sys.path.insert(0, str(_TAYLOR_BASE))

from taylorseer_style.pixelgen_io import (  # noqa: E402
    AtomicManifestSaveHook as _BaseSaveHook,
    ManifestNoiseDataset as _BaseManifestNoiseDataset,
)


class ManifestNoiseDataset(_BaseManifestNoiseDataset):
    """Add explicit raw/resolved configuration identities to every sample."""

    def __init__(
        self,
        *,
        input_config_sha256: str,
        resolved_config_sha256: str,
        semantic_config_hash: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config_identity = {
            "input_config_sha256": str(input_config_sha256),
            "resolved_config_sha256": str(resolved_config_sha256),
            "semantic_config_hash": str(semantic_config_hash),
        }

    def __getitem__(self, index: int):
        noise, class_id, metadata = super().__getitem__(index)
        metadata.update(self._config_identity)
        return noise, class_id, metadata


class AtomicManifestSaveHook(_BaseSaveHook):
    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        summary = getattr(pl_module.diffusion_sampler, "last_taylorseer_summary", None)
        super().on_predict_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        if not isinstance(summary, dict):
            raise RuntimeError("dynamic trajectory summary is missing")
        trace_dir = self.output_root / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"rank_{self.shard_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


__all__ = ["AtomicManifestSaveHook", "ManifestNoiseDataset"]
