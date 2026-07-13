"""Minimal PixelGen Lightning integration for batch-scoped Taylor histories."""

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
    from src.models.autoencoder.base import BaseAE
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
else:
    _UPSTREAM_IMPORT_ERROR = None


_COMPILE_MODES = frozenset({"upstream", "matched_eager", "blockwise"})


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


class InferenceOnlyTrainer(BaseTrainer):
    """Parameter-free trainer placeholder for prediction-only configurations."""

    def __init__(self) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover - runtime guard.
            raise ImportError("PixelGen's `src` package is required") from _UPSTREAM_IMPORT_ERROR
        super().__init__(null_condition_p=0.0)

    def _impl_trainstep(self, *args: Any, **kwargs: Any):
        raise RuntimeError("InferenceOnlyTrainer cannot be used for training")


class TaylorSeerPixelGenLightning(_UpstreamLightningModel):
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
                "TaylorSeerPixelGenLightning"
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
        object.__setattr__(self, "_taylorseer_compile_mode", compile_mode)

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
        source_runtime = getattr(self.denoiser, "taylor_runtime", None)
        ema_runtime = getattr(self.ema_denoiser, "taylor_runtime", None)
        if source_runtime is None or ema_runtime is None:
            raise TypeError("both PixelGen denoisers must expose taylor_runtime")
        if source_runtime is ema_runtime:
            raise RuntimeError("denoiser and EMA unexpectedly share Taylor runtime")
        if ema_runtime.active or ema_runtime.tensor_count() != 0:
            raise RuntimeError("deepcopied EMA Taylor runtime must start empty")

    @property
    def taylorseer_compile_mode(self) -> str:
        return self._taylorseer_compile_mode

    def configure_model(self) -> None:
        # Upstream retains responsibility for parameter copy, no-grad setup,
        # and invoking compile on both models.  The adapter model's compile()
        # keeps scheduler/history mutation outside compiled regions.
        for name, model in (
            ("denoiser", self.denoiser),
            ("ema_denoiser", self.ema_denoiser),
        ):
            if getattr(model, "compile_mode", None) != self._taylorseer_compile_mode:
                raise RuntimeError(f"{name} compile_mode changed after construction")
        return super().configure_model()

    def predict_step(self, batch, batch_idx):
        x_t, _labels, metadata = batch
        net = self.denoiser if self.eval_original_model else self.ema_denoiser
        runtime = getattr(net, "taylor_runtime", None)
        setter = getattr(
            self.diffusion_sampler, "set_taylorseer_batch_context", None
        )
        clearer = getattr(
            self.diffusion_sampler, "clear_taylorseer_batch_context", None
        )
        if runtime is None or runtime.mode == "upstream_full":
            return super().predict_step(batch, batch_idx)
        if setter is None:
            raise TypeError(
                "TaylorSeer PixelGen model requires TaylorSeerHeunSamplerJiT"
            )

        batch_size = int(x_t.shape[0])
        sample_ids = sample_ids_from_metadata(
            metadata, batch_size=batch_size, batch_idx=int(batch_idx)
        )
        trainer = getattr(self, "_trainer", None)
        rank = int(getattr(trainer, "global_rank", 0))
        epoch = int(getattr(self, "current_epoch", 0))
        global_step = int(getattr(self, "global_step", 0))
        trajectory_id = (
            f"rank-{rank}:epoch-{epoch}:step-{global_step}:batch-{batch_idx}"
        )
        try:
            setter(sample_ids=sample_ids, trajectory_id=trajectory_id)
            return super().predict_step(batch, batch_idx)
        finally:
            # The sampler also clears in its finally.  This protects failures in
            # conditioner/Lightning code before the sampler body begins.
            if clearer is not None:
                clearer()


__all__ = [
    "InferenceOnlyTrainer",
    "TaylorSeerPixelGenLightning",
    "sample_ids_from_metadata",
]
