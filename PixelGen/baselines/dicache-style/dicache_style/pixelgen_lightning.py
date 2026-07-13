"""Minimal PixelGen Lightning integration for batch-scoped DiCache histories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
from torch import nn


try:
    from src.callbacks.simple_ema import SimpleEMA
    from src.diffusion.base.sampling import BaseSampler
    from src.diffusion.base.training import BaseTrainer
    from src.lightning_model import LRSchedulerCallable, OptimizerCallable
    from src.lightning_model import LightningModel as _UpstreamLightningModel
    from src.models.autoencoder.base import BaseAE, fp2uint8
    from src.models.conditioner.base import BaseConditioner
except Exception as exc:  # pragma: no cover - depends on PixelGen's environment.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc
    _UpstreamLightningModel = nn.Module
    BaseAE = Any
    BaseConditioner = Any
    BaseSampler = Any
    BaseTrainer = nn.Module
    LRSchedulerCallable = Any
    OptimizerCallable = Any
    SimpleEMA = Any
    fp2uint8 = None
else:
    _UPSTREAM_IMPORT_ERROR = None


_COMPILE_MODES = frozenset({"upstream", "matched_eager", "blockwise"})


def _require_finite_tensor(value: torch.Tensor, *, stage: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{stage} output must be a tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"non-finite PixelGen {stage} output")


def _validate_compile_mode(mode: str) -> str:
    value = str(mode)
    if value not in _COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of {sorted(_COMPILE_MODES)}, got {mode!r}"
        )
    return value


def _integer_sample_id(value: Any, *, index: int) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"sample_id at index {index} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or (
        isinstance(value, float) and not value.is_integer()
    ):
        raise ValueError(f"sample_id at index {index} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"sample_id at index {index} is not integer-compatible: {value!r}"
        ) from exc


def sample_ids_from_metadata(
    metadata: Any, *, batch_size: int, batch_idx: int
) -> list[int]:
    """Extract the manifest identity for each item without positional fallback."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if isinstance(metadata, Mapping):
        if "sample_id" not in metadata:
            raise ValueError(
                f"batch {batch_idx} metadata is missing manifest field 'sample_id'"
            )
        values = metadata["sample_id"]
        if torch.is_tensor(values):
            values = tuple(values.reshape(-1))
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            values = tuple(values)
        elif batch_size == 1:
            values = (values,)
        else:
            raise ValueError("batched sample_id metadata must be a sequence")
        if len(values) != batch_size:
            raise ValueError(
                f"batch {batch_idx} has {len(values)} sample_ids for B={batch_size}"
            )
        return [
            _integer_sample_id(value, index=index)
            for index, value in enumerate(values)
        ]

    if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        if len(metadata) != batch_size:
            raise ValueError(
                f"batch {batch_idx} has {len(metadata)} metadata rows for B={batch_size}"
            )
        result: list[int] = []
        for index, row in enumerate(metadata):
            if not isinstance(row, Mapping) or "sample_id" not in row:
                raise ValueError(
                    f"batch {batch_idx} metadata row {index} lacks 'sample_id'"
                )
            result.append(_integer_sample_id(row["sample_id"], index=index))
        return result
    raise ValueError(f"batch {batch_idx} metadata has unsupported structure")


def batch_group_id_from_metadata(
    metadata: Any, *, batch_size: int, batch_idx: int
) -> str:
    """Return the one manifest batch-group identity represented by a batch.

    PixelGen's four-GPU launcher starts four independent single-device
    Lightning processes, so ``global_rank`` and ``batch_idx`` are not globally
    unique (and ``batch_idx`` restarts on resume).  The manifest
    ``batch_group_id`` already contains the shard identity and is stable across
    resume, making it the only safe trajectory key.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if isinstance(metadata, Mapping):
        if "batch_group_id" not in metadata:
            raise ValueError(
                f"batch {batch_idx} metadata is missing manifest field "
                "'batch_group_id'"
            )
        values = metadata["batch_group_id"]
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            values = tuple(values)
        elif batch_size == 1:
            values = (values,)
        else:
            raise ValueError("batched batch_group_id metadata must be a sequence")
    elif isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        if len(metadata) != batch_size:
            raise ValueError(
                f"batch {batch_idx} has {len(metadata)} metadata rows for B={batch_size}"
            )
        values = tuple(
            row.get("batch_group_id") if isinstance(row, Mapping) else None
            for row in metadata
        )
    else:
        raise ValueError(f"batch {batch_idx} metadata has unsupported structure")

    if len(values) != batch_size:
        raise ValueError(
            f"batch {batch_idx} has {len(values)} batch_group_ids for B={batch_size}"
        )
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(
            f"batch {batch_idx} has a missing or non-string batch_group_id"
        )
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(
            f"batch {batch_idx} mixes manifest batch groups: {sorted(unique)}"
        )
    return next(iter(unique))


def trajectory_id_from_metadata(
    metadata: Any, *, batch_size: int, batch_idx: int
) -> str:
    """Build a globally unique, resume-stable trajectory identity."""

    group_id = batch_group_id_from_metadata(
        metadata, batch_size=batch_size, batch_idx=batch_idx
    )
    return f"manifest-group:{group_id}"


class InferenceOnlyTrainer(BaseTrainer):
    """Parameter-free trainer placeholder for prediction-only configurations."""

    def __init__(self) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover - runtime guard.
            raise ImportError("PixelGen's `src` package is required") from _UPSTREAM_IMPORT_ERROR
        super().__init__(null_condition_p=0.0)

    def _impl_trainstep(self, *args: Any, **kwargs: Any):
        raise RuntimeError("InferenceOnlyTrainer cannot be used for training")


class DiCachePixelGenLightning(_UpstreamLightningModel):
    """Preserve upstream EMA semantics and scope one runtime to one batch."""

    def __init__(
        self,
        vae: BaseAE,
        conditioner: BaseConditioner,
        denoiser: nn.Module,
        diffusion_trainer: BaseTrainer,
        diffusion_sampler: BaseSampler,
        ema_tracker: SimpleEMA = None,
        optimizer: OptimizerCallable = None,
        lr_scheduler: LRSchedulerCallable = None,
        eval_original_model: bool = False,
        compile_mode: str = "matched_eager",
    ) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover - runtime guard.
            raise ImportError(
                "PixelGen's `src` package is required to construct "
                "DiCachePixelGenLightning"
            ) from _UPSTREAM_IMPORT_ERROR
        compile_mode = _validate_compile_mode(compile_mode)
        super().__init__(
            vae=vae,
            conditioner=conditioner,
            denoiser=denoiser,
            diffusion_trainer=diffusion_trainer,
            diffusion_sampler=diffusion_sampler,
            ema_tracker=ema_tracker,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            eval_original_model=eval_original_model,
        )
        if eval_original_model:
            raise ValueError(
                "DiCache evaluation is pinned to ema_denoiser; "
                "eval_original_model=true would invalidate run provenance"
            )
        object.__setattr__(self, "_dicache_compile_mode", compile_mode)

        # Upstream created ema_denoiser via deepcopy above.  The common runtime
        # defines __deepcopy__ to produce an empty independent state machine.
        for name, model in (
            ("denoiser", self.denoiser),
            ("ema_denoiser", self.ema_denoiser),
        ):
            model_mode = getattr(model, "compile_mode", None)
            if model_mode != compile_mode:
                raise ValueError(
                    f"{name}.compile_mode={model_mode!r} does not match "
                    f"Lightning compile_mode={compile_mode!r}"
                )
        source_runtime = getattr(self.denoiser, "dicache_runtime", None)
        ema_runtime = getattr(self.ema_denoiser, "dicache_runtime", None)
        if source_runtime is None or ema_runtime is None:
            raise TypeError("both PixelGen denoisers must expose dicache_runtime")
        if source_runtime is ema_runtime:
            raise RuntimeError("denoiser and EMA unexpectedly share DiCache runtime")
        if ema_runtime.active or ema_runtime.tensor_count() != 0:
            raise RuntimeError("deepcopied EMA DiCache runtime must start empty")

    @property
    def dicache_compile_mode(self) -> str:
        return self._dicache_compile_mode

    def configure_model(self) -> None:
        # Upstream retains responsibility for parameter copy, no-grad setup,
        # and invoking compile on both models.  The adapter model's compile()
        # keeps scheduler/history mutation outside compiled regions.
        for name, model in (
            ("denoiser", self.denoiser),
            ("ema_denoiser", self.ema_denoiser),
        ):
            if getattr(model, "compile_mode", None) != self._dicache_compile_mode:
                raise RuntimeError(f"{name} compile_mode changed after construction")
        return super().configure_model()

    def predict_step(self, batch, batch_idx):
        x_t, _labels, metadata = batch
        net = self.ema_denoiser
        runtime = getattr(net, "dicache_runtime", None)
        setter = getattr(
            self.diffusion_sampler, "set_dicache_batch_context", None
        )
        clearer = getattr(
            self.diffusion_sampler, "clear_dicache_batch_context", None
        )
        if runtime is None:
            raise TypeError("PixelGen prediction model has no DiCache runtime")
        if setter is None:
            raise TypeError(
                "DiCache PixelGen model requires DiCacheHeunSamplerJiT"
            )

        batch_size = int(x_t.shape[0])
        sample_ids = sample_ids_from_metadata(
            metadata, batch_size=batch_size, batch_idx=int(batch_idx)
        )
        trajectory_id = trajectory_id_from_metadata(
            metadata, batch_size=batch_size, batch_idx=int(batch_idx)
        )
        try:
            setter(sample_ids=sample_ids, trajectory_id=trajectory_id)
            with torch.no_grad():
                condition, uncondition = self.conditioner(_labels)
                samples = self.diffusion_sampler(
                    net, x_t, condition, uncondition
                )
                _require_finite_tensor(samples, stage="sampler-float")
                decoded = self.vae.decode(samples)
                _require_finite_tensor(decoded, stage="decoded-float")
                summary = getattr(
                    self.diffusion_sampler, "last_dicache_summary", None
                )
                if isinstance(summary, dict):
                    summary["raw_sample_finite"] = True
                    summary["decoded_sample_finite"] = True
                    runtime.last_summary = dict(summary)
                return fp2uint8(decoded)
        finally:
            # The sampler also clears in its finally.  This protects failures in
            # conditioner/Lightning code before the sampler body begins.
            if clearer is not None:
                clearer()


__all__ = [
    "batch_group_id_from_metadata",
    "InferenceOnlyTrainer",
    "DiCachePixelGenLightning",
    "sample_ids_from_metadata",
    "trajectory_id_from_metadata",
    "_require_finite_tensor",
]
