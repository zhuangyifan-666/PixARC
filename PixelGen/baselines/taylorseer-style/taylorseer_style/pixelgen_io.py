"""Manifest-backed PixelGen prediction dataset and atomic save callback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from lightning.pytorch import Callback
from torch.utils.data import Dataset

from .image_io import (
    atomic_write_png,
    image_path,
    load_metadata_jsonl,
    resumable_batch_groups,
)
from .manifest import (
    initial_noise,
    load_manifest,
    records_for_shard,
    sha256_file,
    validate_manifest,
)
from .metadata import atomic_write_json


class ManifestNoiseDataset(Dataset):
    """One fixed shard, in fixed batch-group order, with per-sample RNG."""

    def __init__(
        self,
        *,
        manifest_path: str,
        shard_id: int,
        world_size: int,
        output_root: str,
        batch_size: int,
        config_hash: str,
        checkpoint_path: str,
        checkpoint_size: int,
        method: str,
        interval: int,
        max_order: int,
        coordinate_mode: str,
        resolution: int = 256,
        noise_scale: float = 1.0,
        resume: bool = False,
    ) -> None:
        super().__init__()
        records = load_manifest(manifest_path)
        validate_manifest(records, world_size=world_size, batch_size=batch_size)
        shard = records_for_shard(records, shard_id)
        if not shard:
            raise ValueError(f"manifest shard {shard_id} is empty")
        if any(
            record.position_in_batch >= batch_size
            for record in shard
        ):
            raise ValueError("runtime batch_size is incompatible with manifest grouping")
        root = Path(output_root)
        manifest_digest = sha256_file(manifest_path)
        sample_dir = root / "samples"
        metadata_path = root / "metadata" / f"rank_{shard_id}.jsonl"
        existing_metadata = (
            load_metadata_jsonl(metadata_path) if metadata_path.exists() else {}
        )
        if not resume:
            existing = [
                record.sample_id
                for record in shard
                if image_path(sample_dir, record.sample_id).exists()
            ]
            if existing or metadata_path.exists():
                raise FileExistsError(
                    f"rank {shard_id} already has outputs; pass resume only after validation"
                )
        pending, skipped = resumable_batch_groups(
            records,
            shard_id,
            sample_dir,
            existing_metadata,
            manifest_sha256=manifest_digest,
            config_hash=config_hash,
            checkpoint_path=checkpoint_path,
            checkpoint_size=checkpoint_size,
            method=method,
            interval=interval,
            max_order=max_order,
            coordinate_mode=coordinate_mode,
            resolution=resolution,
        )
        self.records = [record for group in pending for record in group]
        self.skipped_group_ids = tuple(skipped)
        self.shard_id = int(shard_id)
        self.output_root = str(root)
        self.resolution = int(resolution)
        self.noise_scale = float(noise_scale)
        self.config_hash = str(config_hash)
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_size = int(checkpoint_size)
        self.method = str(method)
        self.interval = int(interval)
        self.max_order = int(max_order)
        self.coordinate_mode = str(coordinate_mode)
        self.manifest_sha256 = manifest_digest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        noise = initial_noise(
            [record.seed],
            (3, self.resolution, self.resolution),
            device="cpu",
            dtype=torch.float32,
        )[0]
        noise = noise * self.noise_scale
        metadata = {
            "sample_id": record.sample_id,
            "class_id": record.class_id,
            "seed": record.seed,
            "batch_group_id": record.batch_group_id,
            "position_in_batch": record.position_in_batch,
            "manifest_sha256": self.manifest_sha256,
            "config_hash": self.config_hash,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_size": self.checkpoint_size,
        }
        metadata.update(
            {
                "method": self.method,
                "interval": self.interval,
                "max_order": self.max_order,
                "coordinate_mode": self.coordinate_mode,
            }
        )
        return noise, record.class_id, metadata


class AtomicManifestSaveHook(Callback):
    """Write numeric PNGs and JSONL metadata without all-gather or NPZ copies."""

    def __init__(
        self,
        *,
        output_root: str,
        shard_id: int,
        resolution: int = 256,
        resume: bool = False,
    ) -> None:
        super().__init__()
        self.output_root = Path(output_root)
        self.shard_id = int(shard_id)
        self.resolution = int(resolution)
        self.resume = bool(resume)
        self.invocation_id = os.environ.get("TAYLORSEER_INVOCATION_ID")
        self._metadata_handle = None
        self._counts = {
            "generated": 0,
            "generated_this_invocation": 0,
            "trajectory_count": 0,
            "total_nfe": 0,
            "full_nfe": 0,
            "taylor_nfe": 0,
            "network_forwards": 0,
            "history_update_time_ms": 0.0,
            "forecast_time_ms": 0.0,
            "scheduler_time_ms": 0.0,
            "max_forecast_horizon": 0.0,
            "max_cache_bytes": 0,
            "max_cache_tensor_count": 0,
            "max_peak_memory_allocated": 0,
            "max_peak_memory_reserved": 0,
        }

    def on_predict_epoch_start(self, trainer, pl_module) -> None:
        if trainer.world_size != 1:
            raise RuntimeError(
                "AtomicManifestSaveHook expects one process per GPU; do not wrap it in DDP"
            )
        for relative in ("samples", "metadata", "summaries", "logs"):
            (self.output_root / relative).mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_root / "metadata" / f"rank_{self.shard_id}.jsonl"
        existing_metadata = (
            load_metadata_jsonl(metadata_path) if metadata_path.exists() else {}
        )
        self._counts = {
            "generated": len(existing_metadata),
            "generated_this_invocation": 0,
            "trajectory_count": 0,
            "total_nfe": 0,
            "full_nfe": 0,
            "taylor_nfe": 0,
            "network_forwards": 0,
            "history_update_time_ms": 0.0,
            "forecast_time_ms": 0.0,
            "scheduler_time_ms": 0.0,
            "max_forecast_horizon": 0.0,
            "max_cache_bytes": 0,
            "max_cache_tensor_count": 0,
            "max_peak_memory_allocated": 0,
            "max_peak_memory_reserved": 0,
        }
        existing_groups: dict[str, dict[str, Any]] = {}
        for row in existing_metadata.values():
            existing_groups.setdefault(str(row["batch_group_id"]), row)
        for row in existing_groups.values():
            self._counts["trajectory_count"] += 1
            for field in ("total_nfe", "full_nfe", "taylor_nfe", "network_forwards"):
                self._counts[field] += int(row[f"trajectory_{field}"])
            for field in ("history_update_time_ms", "forecast_time_ms", "scheduler_time_ms"):
                self._counts[field] += float(row[f"trajectory_{field}"])
            self._counts["max_forecast_horizon"] = max(
                float(self._counts["max_forecast_horizon"]),
                float(row["trajectory_max_forecast_horizon"]),
            )
            self._counts["max_cache_bytes"] = max(
                int(self._counts["max_cache_bytes"]), int(row["trajectory_cache_bytes"])
            )
            self._counts["max_cache_tensor_count"] = max(
                int(self._counts["max_cache_tensor_count"]),
                int(row["trajectory_cache_tensor_count"]),
            )
            self._counts["max_peak_memory_allocated"] = max(
                int(self._counts["max_peak_memory_allocated"]),
                int(row["trajectory_peak_memory_allocated"]),
            )
            self._counts["max_peak_memory_reserved"] = max(
                int(self._counts["max_peak_memory_reserved"]),
                int(row["trajectory_peak_memory_reserved"]),
            )
        mode = "a" if self.resume else "x"
        self._metadata_handle = metadata_path.open(mode, encoding="utf-8")

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if self._metadata_handle is None:
            raise RuntimeError("metadata writer was not initialized")
        _noise, _labels, metadata = batch
        metadata_rows = _unbatch_metadata(metadata, int(outputs.shape[0]))
        if len(metadata_rows) != int(outputs.shape[0]):
            raise ValueError("PixelGen output and metadata batch sizes differ")
        arrays = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()
        if arrays.dtype != np.uint8:
            raise ValueError(f"PixelGen callback expected uint8 output, got {arrays.dtype}")
        summary = getattr(pl_module.diffusion_sampler, "last_taylorseer_summary", None)
        if not summary:
            raise RuntimeError("PixelGen sampler did not expose a trajectory summary")
        trajectory_summary = {
            "total_nfe": int(summary["total_nfe"]),
            "full_nfe": int(summary["full_nfe"]),
            "taylor_nfe": int(summary["taylor_nfe"]),
            "network_forwards": int(summary["network_forward_count"]),
            "history_update_time_ms": float(summary.get("history_update_time_ms", 0.0)),
            "forecast_time_ms": float(summary.get("forecast_time_ms", 0.0)),
            "scheduler_time_ms": float(summary.get("scheduler_time_ms", 0.0)),
            "max_forecast_horizon": float(summary.get("max_forecast_horizon", 0.0)),
            "cache_bytes": int(summary.get("cache_bytes", 0)),
            "cache_tensor_count": int(summary.get("cache_tensor_count", 0)),
            "peak_memory_allocated": int(summary.get("peak_memory_allocated", 0)),
            "peak_memory_reserved": int(summary.get("peak_memory_reserved", 0)),
        }
        for array, row in zip(arrays, metadata_rows, strict=True):
            sample_id = int(row["sample_id"])
            atomic_write_png(
                array,
                image_path(self.output_root / "samples", sample_id),
                resolution=self.resolution,
            )
            value = {key: _python_scalar(item) for key, item in row.items()}
            value.update(
                {
                    f"trajectory_{key}": item
                    for key, item in trajectory_summary.items()
                }
            )
            value["status"] = "ok"
            self._metadata_handle.write(json.dumps(value, sort_keys=True) + "\n")
            self._metadata_handle.flush()
            os.fsync(self._metadata_handle.fileno())
            self._counts["generated"] += 1
            self._counts["generated_this_invocation"] += 1
        self._counts["trajectory_count"] += 1
        for key in (
            "total_nfe",
            "full_nfe",
            "taylor_nfe",
            "network_forwards",
            "history_update_time_ms",
            "forecast_time_ms",
            "scheduler_time_ms",
        ):
            self._counts[key] += trajectory_summary[key]
        self._counts["max_forecast_horizon"] = max(
            float(self._counts["max_forecast_horizon"]),
            float(trajectory_summary["max_forecast_horizon"]),
        )
        self._counts["max_cache_bytes"] = max(
            int(self._counts["max_cache_bytes"]), int(trajectory_summary["cache_bytes"])
        )
        self._counts["max_cache_tensor_count"] = max(
            int(self._counts["max_cache_tensor_count"]),
            int(trajectory_summary["cache_tensor_count"]),
        )
        self._counts["max_peak_memory_allocated"] = max(
            int(self._counts["max_peak_memory_allocated"]),
            int(trajectory_summary["peak_memory_allocated"]),
        )
        self._counts["max_peak_memory_reserved"] = max(
            int(self._counts["max_peak_memory_reserved"]),
            int(trajectory_summary["peak_memory_reserved"]),
        )

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        if self._metadata_handle is not None:
            self._metadata_handle.close()
            self._metadata_handle = None
        dataset = trainer.datamodule.pred_dataset
        self._counts["skipped_groups"] = len(
            getattr(dataset, "skipped_group_ids", ())
        )
        atomic_write_json(
            self.output_root / "summaries" / f"rank_{self.shard_id}_summary.json",
            {
                **self._counts,
                "shard_id": self.shard_id,
                "invocation_id": self.invocation_id,
            },
        )

    def teardown(self, trainer, pl_module, stage: str) -> None:
        if self._metadata_handle is not None:
            self._metadata_handle.close()
            self._metadata_handle = None


def _python_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("metadata tensors must be scalar")
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unbatch_metadata(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(metadata, Mapping):
        rows = []
        for index in range(batch_size):
            row = {}
            for key, value in metadata.items():
                if torch.is_tensor(value):
                    if value.shape[0] != batch_size:
                        raise ValueError(f"batched metadata field {key!r} has wrong length")
                    row[key] = value[index]
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    if len(value) != batch_size:
                        raise ValueError(f"batched metadata field {key!r} has wrong length")
                    row[key] = value[index]
                elif batch_size == 1:
                    row[key] = value
                else:
                    raise ValueError(f"metadata field {key!r} is not batched")
            rows.append(row)
        return rows
    if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        rows = list(metadata)
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("metadata sequence must contain mappings")
        return [dict(row) for row in rows]
    raise ValueError("metadata must be a mapping of batches or a sequence of rows")


__all__ = ["AtomicManifestSaveHook", "ManifestNoiseDataset"]
