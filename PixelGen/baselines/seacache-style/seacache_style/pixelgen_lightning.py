"""Lightning integration that scopes one SeaCache trajectory to one batch."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .pixelgen_model import configure_pixelgen_compile_mode


try:
    from src.callbacks.simple_ema import SimpleEMA
    from src.diffusion.base.sampling import BaseSampler
    from src.diffusion.base.training import BaseTrainer
    from src.lightning_model import LRSchedulerCallable, OptimizerCallable
    from src.lightning_model import LightningModel as _UpstreamLightningModel
    from src.models.autoencoder.base import BaseAE
    from src.models.conditioner.base import BaseConditioner
    from src.utils.copy import copy_params as _copy_params
    from src.utils.no_grad import no_grad as _no_grad
except Exception as exc:  # pragma: no cover - depends on PixelGen's runtime.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc
    _UpstreamLightningModel = nn.Module
    BaseAE = Any
    BaseConditioner = Any
    BaseSampler = Any
    BaseTrainer = Any
    LRSchedulerCallable = Any
    OptimizerCallable = Any
    SimpleEMA = Any
    _copy_params = None
    _no_grad = None
else:
    _UPSTREAM_IMPORT_ERROR = None


_VALID_COMPILE_MODES = frozenset({"matched_eager", "blockwise", "upstream"})


def _validate_compile_mode(value: str) -> str:
    mode = str(value)
    if mode not in _VALID_COMPILE_MODES:
        raise ValueError(
            "compile_mode must be one of "
            f"{sorted(_VALID_COMPILE_MODES)}, got {value!r}"
        )
    return mode


def _scalar_at(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        flattened = value.reshape(-1)
        if flattened.numel() != batch_size:
            raise ValueError(
                "metadata['sample_id'] must contain exactly one value per batch item; "
                f"got {flattened.numel()} values for batch_size={batch_size}"
            )
        item = flattened[index]
        return item.detach().cpu().item()
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(
                "metadata['sample_id'] must contain exactly one value per batch item; "
                f"got {len(value)} values for batch_size={batch_size}"
            )
        return value[index]
    if batch_size != 1:
        raise ValueError(
            "scalar metadata['sample_id'] is valid only for batch_size=1"
        )
    return value


def _as_manifest_sample_id(value: Any, *, index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"sample_id at batch index {index} must be an integer, not bool"
        )
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"sample_id at batch index {index} must be losslessly convertible to int"
        )
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"sample_id at batch index {index} must be a manifest-stable integer; "
            f"got {value!r}"
        ) from exc
    return converted


def sample_ids_from_metadata(
    metadata: Any, batch_size: int, batch_idx: int
) -> List[int]:
    """Read the manifest's stable integer ``sample_id`` for every item.

    Filename/seed strings and batch-position fallbacks are intentionally not
    accepted: they are not stable manifest identities and can silently collide
    across ranks or resumed jobs.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if isinstance(metadata, Mapping):
        if "sample_id" not in metadata:
            raise ValueError(
                f"batch {batch_idx} metadata must provide manifest-stable "
                "integer 'sample_id' values"
            )
        return [
            _as_manifest_sample_id(
                _scalar_at(metadata["sample_id"], index, batch_size),
                index=index,
            )
            for index in range(batch_size)
        ]

    if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        if len(metadata) != batch_size:
            raise ValueError(
                f"batch {batch_idx} metadata must contain {batch_size} items; "
                f"got {len(metadata)}"
            )
        result: List[int] = []
        for index, item in enumerate(metadata):
            if not isinstance(item, Mapping) or "sample_id" not in item:
                raise ValueError(
                    f"batch {batch_idx} metadata item {index} must provide "
                    "integer 'sample_id'"
                )
            result.append(_as_manifest_sample_id(item["sample_id"], index=index))
        return result

    raise ValueError(
        f"batch {batch_idx} metadata must provide manifest-stable integer "
        "'sample_id' values"
    )


_InferenceTrainerBase = BaseTrainer if _UPSTREAM_IMPORT_ERROR is None else nn.Module


class InferenceOnlyTrainer(_InferenceTrainerBase):
    """Parameter-free placeholder required by the upstream prediction module.

    Prediction never calls the diffusion trainer. Keeping this local component
    avoids importing training-only LPIPS/DINO modules or invoking torch.hub.
    """

    def __init__(self) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover - runtime guard.
            raise ImportError("PixelGen's `src` package is required") from _UPSTREAM_IMPORT_ERROR
        super().__init__(null_condition_p=0.0)

    def _impl_trainstep(self, *args: Any, **kwargs: Any):
        raise RuntimeError("InferenceOnlyTrainer cannot be used for training")


class SeaCacheLightningModel(_UpstreamLightningModel):
    """PixelGen LightningModel with batch-local sampler context and cleanup."""

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
        compile_mode = _validate_compile_mode(compile_mode)
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover - runtime guard.
            raise ImportError(
                "PixelGen's `src` package is required to construct SeaCacheLightningModel"
            ) from _UPSTREAM_IMPORT_ERROR
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
        object.__setattr__(self, "_seacache_compile_mode", compile_mode)
        object.__setattr__(self.denoiser, "_seacache_compile_mode", compile_mode)
        object.__setattr__(self.ema_denoiser, "_seacache_compile_mode", compile_mode)
        denoiser_unwrapped = configure_pixelgen_compile_mode(
            self.denoiser, compile_mode
        )
        ema_unwrapped = configure_pixelgen_compile_mode(
            self.ema_denoiser, compile_mode
        )
        if denoiser_unwrapped != ema_unwrapped:
            raise RuntimeError("denoiser/EMA compile wrapper counts differ")
        object.__setattr__(
            self, "_seacache_compile_wrappers_unwrapped", denoiser_unwrapped
        )

    @property
    def seacache_compile_mode(self) -> str:
        return self._seacache_compile_mode

    def configure_model(self) -> None:
        mode = _validate_compile_mode(self._seacache_compile_mode)
        denoiser_mode = getattr(self.denoiser, "_seacache_compile_mode", None)
        ema_mode = getattr(self.ema_denoiser, "_seacache_compile_mode", None)
        if denoiser_mode != mode or ema_mode != mode:
            raise RuntimeError(
                "denoiser and ema_denoiser must use the same compile_mode as "
                "SeaCacheLightningModel"
            )

        if mode == "upstream":
            return super().configure_model()

        assert _copy_params is not None and _no_grad is not None
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            trainer = self.trainer
        trainer.strategy.barrier()
        _copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)
        _no_grad(self.conditioner)
        _no_grad(self.vae)
        _no_grad(self.ema_denoiser)

        # Do not outer-compile either denoiser: controller state and gate
        # decisions stay eager. PixelGen's JiTBlock.forward is decorated with
        # @torch.compile upstream; blockwise preserves those wrappers, whereas
        # matched_eager has already rebound the original callables per instance.

    def predict_step(self, batch, batch_idx):
        x_t, _condition, metadata = batch
        setter = getattr(self.diffusion_sampler, "set_seacache_batch_context", None)
        clearer = getattr(self.diffusion_sampler, "clear_seacache_batch_context", None)
        if setter is None:
            return super().predict_step(batch, batch_idx)

        net = self.denoiser if self.eval_original_model else self.ema_denoiser
        controller = getattr(net, "seacache_controller", None)
        if controller is not None and getattr(controller, "mode", None) == "full":
            return super().predict_step(batch, batch_idx)

        batch_size = int(x_t.shape[0])
        sample_ids = sample_ids_from_metadata(metadata, batch_size, batch_idx)
        rank = int(getattr(self.trainer, "global_rank", 0))
        epoch = int(getattr(self, "current_epoch", 0))
        global_step = int(getattr(self, "global_step", 0))
        stream_id = "combined_cfg"
        trajectory_id = (
            f"rank-{rank}:epoch-{epoch}:step-{global_step}:batch-{batch_idx}"
        )

        try:
            setter(
                sample_ids=sample_ids,
                trajectory_id=trajectory_id,
                stream_id=stream_id,
            )
            return super().predict_step(batch, batch_idx)
        finally:
            # The sampler also clears in its own finally block. Keeping this
            # outer cleanup protects failures before the sampler body starts.
            if clearer is not None:
                clearer()


PixelGenSeaCacheLightningModel = SeaCacheLightningModel


__all__ = [
    "InferenceOnlyTrainer",
    "PixelGenSeaCacheLightningModel",
    "SeaCacheLightningModel",
    "sample_ids_from_metadata",
]
