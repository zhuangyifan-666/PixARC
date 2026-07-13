from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seacache_style.pixelgen_model import PixelGenSeaCacheModelMixin
from seacache_style.controller import SeaCacheController
from seacache_style.pixelgen_lightning import SeaCacheLightningModel
import seacache_style.pixelgen_lightning as pixelgen_lightning
from seacache_style.pixelgen_benchmark import _benchmark_denoiser_spec


class RuntimeController(nn.Module):
    """An nn.Module controller catches accidental parameter-tree registration."""

    def __init__(self, mode="seacache", threshold=0.0, trace_mode="off"):
        super().__init__()
        self.mode = mode
        self.threshold = threshold
        self.trace_mode = trace_mode
        self.register_buffer("runtime_tensor", torch.tensor([1.0]))
        self.runtime = {"calls": []}


class ToyModel(PixelGenSeaCacheModelMixin, nn.Module):
    def __init__(self, controller_factory=RuntimeController):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("persistent", torch.tensor([3.0]))
        self._init_seacache_runtime(
            "seacache",
            0.125,
            "off",
            controller_factory=controller_factory,
        )
        self.set_seacache_call_context(
            "original-stream", solver_stage="euler", macro_step=0
        )


class CompileSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.compile_calls = 0

    def compile(self, *args, **kwargs):
        self.compile_calls += 1


class DeepcopyStateTest(unittest.TestCase):
    def test_benchmark_spec_injects_selected_threshold_without_mutation(self):
        source = {
            "class_path": "example.Model",
            "init_args": {"hidden_size": 8},
        }

        configured = _benchmark_denoiser_spec(source, 0.125)

        self.assertEqual(source["init_args"], {"hidden_size": 8})
        self.assertEqual(configured["init_args"]["seacache_mode"], "seacache")
        self.assertEqual(configured["init_args"]["seacache_threshold"], 0.125)
        self.assertEqual(configured["init_args"]["seacache_trace_mode"], "summary")
        with self.assertRaises(ValueError):
            _benchmark_denoiser_spec(source, True)

    def test_runtime_controller_is_absent_from_state_dict(self):
        model = ToyModel()

        self.assertEqual(set(model.state_dict()), {"weight", "persistent"})
        self.assertNotIn("_seacache_controller", model._modules)
        self.assertFalse(
            any(key.startswith("_seacache") for key in model.state_dict())
        )

    def test_deepcopy_owns_independent_runtime_state(self):
        model = ToyModel()
        clone = copy.deepcopy(model)

        self.assertIsNot(clone.seacache_controller, model.seacache_controller)
        self.assertIsNot(
            clone.seacache_controller.runtime,
            model.seacache_controller.runtime,
        )
        self.assertIsNot(
            clone._seacache_call_context,
            model._seacache_call_context,
        )

        clone.seacache_controller.runtime["calls"].append("clone-only")
        clone.seacache_controller.runtime_tensor.add_(5)
        clone._seacache_call_context["stream_id"] = "clone-stream"

        self.assertEqual(model.seacache_controller.runtime["calls"], [])
        torch.testing.assert_close(
            model.seacache_controller.runtime_tensor, torch.tensor([1.0])
        )
        self.assertEqual(
            model._seacache_call_context["stream_id"], "original-stream"
        )
        self.assertEqual(set(clone.state_dict()), {"weight", "persistent"})

    def test_real_controller_deepcopy_drops_live_trajectory(self):
        model = ToyModel(controller_factory=SeaCacheController)
        controller = model.seacache_controller
        controller.begin_trajectory(
            "source", "active-trajectory", total_calls=2, sample_ids=[31]
        )
        tokens = torch.zeros(1, 1, 1)
        controller.compute(
            "source",
            tokens,
            tokens,
            torch.zeros(1),
            (1, 1),
            lambda value: value + 1,
        )

        clone = copy.deepcopy(model)

        self.assertTrue(controller.state("source").active)
        self.assertEqual(clone.seacache_controller._states, {})
        self.assertEqual(clone.seacache_controller._traces, {})
        self.assertIsNot(clone.seacache_controller, controller)
        self.assertEqual(set(clone.state_dict()), {"weight", "persistent"})

    def test_non_upstream_compile_modes_skip_outer_compile_symmetrically(self):
        for mode in ("matched_eager", "blockwise"):
            with self.subTest(mode=mode):
                lightning = object.__new__(SeaCacheLightningModel)
                nn.Module.__init__(lightning)
                lightning.denoiser = CompileSpy()
                lightning.ema_denoiser = CompileSpy()
                lightning.conditioner = nn.Linear(1, 1)
                lightning.vae = nn.Linear(1, 1)
                object.__setattr__(lightning, "_seacache_compile_mode", mode)
                object.__setattr__(lightning.denoiser, "_seacache_compile_mode", mode)
                object.__setattr__(
                    lightning.ema_denoiser, "_seacache_compile_mode", mode
                )

                barrier_calls = []
                copy_calls = []
                no_grad_calls = []
                object.__setattr__(
                    lightning,
                    "_trainer",
                    SimpleNamespace(
                        strategy=SimpleNamespace(
                            barrier=lambda: barrier_calls.append("barrier")
                        )
                    ),
                )

                def copy_params(*, src_model, dst_model):
                    copy_calls.append((src_model, dst_model))

                def no_grad(module):
                    no_grad_calls.append(module)

                with mock.patch.object(
                    pixelgen_lightning, "_copy_params", copy_params
                ), mock.patch.object(pixelgen_lightning, "_no_grad", no_grad):
                    lightning.configure_model()

                self.assertEqual(barrier_calls, ["barrier"])
                self.assertEqual(
                    copy_calls, [(lightning.denoiser, lightning.ema_denoiser)]
                )
                self.assertEqual(
                    no_grad_calls,
                    [lightning.conditioner, lightning.vae, lightning.ema_denoiser],
                )
                self.assertEqual(lightning.denoiser.compile_calls, 0)
                self.assertEqual(lightning.ema_denoiser.compile_calls, 0)

    def test_compile_mode_validation_and_symmetry_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "compile_mode"):
            SeaCacheLightningModel(
                None, None, None, None, None, compile_mode="invalid"
            )

        lightning = object.__new__(SeaCacheLightningModel)
        nn.Module.__init__(lightning)
        lightning.denoiser = CompileSpy()
        lightning.ema_denoiser = CompileSpy()
        object.__setattr__(lightning, "_seacache_compile_mode", "blockwise")
        object.__setattr__(lightning.denoiser, "_seacache_compile_mode", "blockwise")
        object.__setattr__(
            lightning.ema_denoiser, "_seacache_compile_mode", "matched_eager"
        )
        with self.assertRaisesRegex(RuntimeError, "same compile_mode"):
            lightning.configure_model()


if __name__ == "__main__":
    unittest.main()
