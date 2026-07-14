from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seacache_style.pixelgen_model import (
    PixelGenSeaCacheModelMixin,
    configure_pixelgen_compile_mode,
)
import seacache_style.pixelgen_model as pixelgen_model
from seacache_style.pixelgen_lightning import sample_ids_from_metadata
from seacache_style.controller import SeaCacheController
from seacache_style.pixelgen_sampler import (
    PixelGenSeaCacheSamplerMixin,
    combined_cfg_sample_ids,
    expected_model_calls,
)


class FakeController:
    def __init__(self, mode="seacache", threshold=0.0, trace_mode="off"):
        self.mode = mode
        self.begins = []
        self.computes = []
        self.ends = []
        self.resets = []

    def begin_trajectory(self, stream_id, trajectory_id, total_calls, sample_ids):
        self.begins.append((stream_id, trajectory_id, total_calls, tuple(sample_ids)))

    def compute(
        self,
        stream_id,
        body_input,
        probe_raw,
        t,
        grid_shape,
        body_fn,
        solver_stage="",
        macro_step=None,
        force_full_reason=None,
    ):
        self.computes.append(
            {
                "stream_id": stream_id,
                "batch": body_input.shape[0],
                "dtype": body_input.dtype,
                "stage": solver_stage,
                "macro_step": macro_step,
                "force_full_reason": force_full_reason,
            }
        )
        return body_fn(body_input)

    def end_trajectory(self, stream_id, require_complete=True):
        expected = self.begins[-1][2]
        if require_complete and len(self.computes) != expected:
            raise AssertionError((len(self.computes), expected))
        summary = {"stream_id": stream_id, "calls": len(self.computes)}
        self.ends.append(summary)
        return summary

    def reset(self, stream_id):
        self.resets.append(stream_id)


class FakeNet:
    def __init__(self):
        self.seacache_controller = FakeController()
        self.context = None

    def set_seacache_call_context(self, stream_id, **kwargs):
        self.context = {"stream_id": stream_id, **kwargs}

    def clear_seacache_call_context(self):
        self.context = None

    def __call__(self, x, t, y):
        if self.seacache_controller.mode == "full":
            return x + 1
        assert self.context is not None
        context = self.context
        return self.seacache_controller.compute(
            context["stream_id"],
            x,
            x,
            t,
            (1, x.shape[-1]),
            lambda value: value + 1,
            solver_stage=context["solver_stage"],
            macro_step=context["macro_step"],
            force_full_reason=context["force_full_reason"],
        )


class ControllerBackedNet:
    """Tiny CPU net exercising the real shared controller contract."""

    def __init__(self):
        self.seacache_controller = SeaCacheController(
            mode="seacache", threshold=1.0, trace_mode="off"
        )
        self.context = None

    def set_seacache_call_context(self, stream_id, **kwargs):
        self.context = {"stream_id": stream_id, **kwargs}

    def clear_seacache_call_context(self):
        self.context = None

    def __call__(self, x, t, y):
        del y
        assert self.context is not None
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        output = self.seacache_controller.compute(
            self.context["stream_id"],
            tokens,
            tokens,
            t,
            (height, width),
            lambda value: value + 1,
            solver_stage=self.context["solver_stage"],
            macro_step=self.context["macro_step"],
            force_full_reason=self.context["force_full_reason"],
        )
        return output.transpose(1, 2).reshape(batch, channels, height, width)


class ToyEulerBase:
    def __init__(self, num_steps=3):
        self.num_steps = num_steps

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch = noise.shape[0]
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        trajectory = [x]
        velocities = []
        for step in range(self.num_steps):
            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = torch.full((2 * batch,), float(step))
            out = net(cfg_x, cfg_t, cfg_condition)
            x = out[:batch]
            trajectory.append(x)
            velocities.append(x)
        return trajectory, velocities


class ToySampler(PixelGenSeaCacheSamplerMixin, ToyEulerBase):
    pass


class FailingBase(ToyEulerBase):
    def _impl_sampling(self, net, noise, condition, uncondition):
        cfg_x = torch.cat([noise, noise], dim=0)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        net(cfg_x, torch.zeros(cfg_x.shape[0]), cfg_condition)
        raise RuntimeError("intentional")


class FailingSampler(PixelGenSeaCacheSamplerMixin, FailingBase):
    pass


class ShortBase(ToyEulerBase):
    def _impl_sampling(self, net, noise, condition, uncondition):
        cfg_x = torch.cat([noise, noise], dim=0)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        net(cfg_x, torch.zeros(cfg_x.shape[0]), cfg_condition)
        return [noise], []


class ShortSampler(PixelGenSeaCacheSamplerMixin, ShortBase):
    pass


class _TimeEmbed(nn.Module):
    def forward(self, t):
        return t.reshape(-1, 1)


class _PatchEmbed(nn.Module):
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class _ThreeTokenEmbed(nn.Module):
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)[:, :3]


class _ZeroMod(nn.Module):
    def forward(self, condition):
        return torch.zeros(condition.shape[0], condition.shape[1] * 6)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dtypes = []
        self.adaLN_modulation = _ZeroMod()
        self.norm1 = nn.Identity()

    def forward(self, x, condition, rope):
        self.input_dtypes.append(x.dtype)
        return x + 1


class _Final(nn.Module):
    def forward(self, x, condition):
        return x


class ToyDiagnosticModel(PixelGenSeaCacheModelMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embedder = _TimeEmbed()
        self.y_embedder = nn.Embedding(4, 1)
        nn.init.zeros_(self.y_embedder.weight)
        self.x_embedder = _PatchEmbed()
        self.register_buffer("pos_embed", torch.zeros(1, 4, 1))
        self.blocks = nn.ModuleList([_Block()])
        self.in_context_len = 0
        self.in_context_start = 1
        self.feat_rope = nn.Identity()
        self.feat_rope_incontext = nn.Identity()
        self.final_layer = _Final()
        self.patch_size = 1
        self._init_seacache_runtime(
            "seacache", 1.0, "off", controller_factory=FakeController
        )

    def unpatchify(self, x, patch_size):
        batch, tokens, channels = x.shape
        side = int(tokens**0.5)
        return x.transpose(1, 2).reshape(batch, channels, side, side)


class CombinedCFGStateTest(unittest.TestCase):
    def test_matched_eager_unwraps_only_the_model_instance(self):
        def original(module, value):
            return value + 1

        def compiled(module, value):
            return value + 2

        compiled._torchdynamo_orig_callable = original

        class Wrapped(nn.Module):
            forward = compiled

        class Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Wrapped()])
                self.final_layer = Wrapped()

        first = Container()
        second = Container()
        self.assertEqual(configure_pixelgen_compile_mode(first, "matched_eager"), 2)
        self.assertEqual(first.blocks[0](torch.tensor(1)).item(), 2)
        self.assertEqual(second.blocks[0](torch.tensor(1)).item(), 3)

    def test_position_add_preserves_low_precision_tokens(self):
        model = ToyDiagnosticModel()
        model.set_seacache_call_context("dtype", solver_stage="euler", macro_step=0)
        with mock.patch.object(
            pixelgen_model,
            "_pixelgen_modulate",
            lambda value, shift, scale: value,
        ):
            model._seacache_forward(
                torch.zeros(2, 1, 2, 2, dtype=torch.bfloat16),
                torch.zeros(2),
                torch.zeros(2, dtype=torch.long),
            )
        self.assertEqual(
            model.seacache_controller.computes[-1]["dtype"], torch.bfloat16
        )

    def test_context_position_add_preserves_low_precision_tokens(self):
        model = ToyDiagnosticModel()
        model.in_context_len = 1
        model.in_context_start = 0
        model.in_context_posemb = nn.Parameter(torch.zeros(1, 1, 1))
        diagnostic = {"feat": None, "last_out": None, "y_emb": torch.zeros(
            2, 1, dtype=torch.bfloat16
        )}
        body = torch.zeros(2, 4, 1, dtype=torch.bfloat16)
        result = model._pixelgen_body(
            body,
            torch.zeros(2, 1, dtype=torch.bfloat16),
            return_layer=None,
            return_last=False,
            diagnostic=diagnostic,
        )
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(model.blocks[0].input_dtypes, [torch.bfloat16])

    def test_combined_ids_keep_upstream_2b_order(self):
        self.assertEqual(combined_cfg_sample_ids([10, 20], 2), (10, 20, 10, 20))
        with self.assertRaisesRegex(ValueError, "manifest-stable"):
            combined_cfg_sample_ids(None, 2)
        with self.assertRaisesRegex(ValueError, "identical ID halves"):
            combined_cfg_sample_ids([10, 20, 20, 10], 2)

    def test_one_state_tracks_each_combined_cfg_call(self):
        sampler = ToySampler(num_steps=3)
        net = FakeNet()
        sampler.set_seacache_batch_context(
            sample_ids=[10, 20], trajectory_id="traj", stream_id="stream"
        )
        noise = torch.zeros(2, 1, 1, 1)
        condition = torch.ones(2, 1)
        uncondition = torch.zeros(2, 1)

        sampler._impl_sampling(net, noise, condition, uncondition)

        self.assertEqual(net.seacache_controller.begins[0][2], 3)
        self.assertEqual(
            net.seacache_controller.begins[0][3], (10, 20, 10, 20)
        )
        self.assertEqual([item["batch"] for item in net.seacache_controller.computes], [4, 4, 4])
        self.assertEqual([item["macro_step"] for item in net.seacache_controller.computes], [0, 1, 2])
        self.assertIsNone(net.context)
        self.assertIsNone(getattr(sampler, "_seacache_batch_context"))
        self.assertEqual(sampler.last_seacache_summary["calls"], 3)

    def test_real_controller_lifecycle_uses_one_combined_cfg_stream(self):
        sampler = ToySampler(num_steps=3)
        net = ControllerBackedNet()
        sampler.set_seacache_batch_context(
            sample_ids=[101, 102], trajectory_id="real-traj", stream_id="real"
        )
        sampler._impl_sampling(
            net,
            torch.zeros(2, 1, 1, 1),
            torch.ones(2, 1),
            torch.zeros(2, 1),
        )

        self.assertEqual(
            sampler.last_seacache_summary["sample_ids"], [101, 102, 101, 102]
        )
        self.assertEqual(sampler.last_seacache_summary["total_calls"], 3)
        state = net.seacache_controller.state("real")
        self.assertFalse(state.active)
        self.assertIsNone(state.previous_probe)
        self.assertIsNone(state.previous_body_residual)

    def test_failed_batch_resets_and_clears(self):
        sampler = FailingSampler(num_steps=3)
        net = FakeNet()
        sampler.set_seacache_batch_context(sample_ids=[0], stream_id="failed")
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            sampler._impl_sampling(
                net,
                torch.zeros(1, 1, 1, 1),
                torch.ones(1, 1),
                torch.zeros(1, 1),
            )
        self.assertEqual(net.seacache_controller.resets, ["failed"])
        self.assertIsNone(net.context)
        self.assertIsNone(getattr(sampler, "_seacache_batch_context"))

    def test_incomplete_call_plan_resets_after_end_failure(self):
        sampler = ShortSampler(num_steps=3)
        net = FakeNet()
        sampler.set_seacache_batch_context(sample_ids=[7], stream_id="short")
        with self.assertRaises(AssertionError):
            sampler._impl_sampling(
                net,
                torch.zeros(1, 1, 1, 1),
                torch.ones(1, 1),
                torch.zeros(1, 1),
            )
        self.assertEqual(net.seacache_controller.resets, ["short"])
        self.assertIsNone(net.context)
        self.assertIsNone(getattr(sampler, "_seacache_batch_context"))

    def test_full_mode_has_no_controller_lifecycle(self):
        sampler = ToySampler(num_steps=2)
        net = FakeNet()
        net.seacache_controller.mode = "full"
        sampler.set_seacache_batch_context(sample_ids=[4], stream_id="full")
        sampler._impl_sampling(
            net,
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1),
            torch.zeros(1, 1),
        )
        self.assertEqual(net.seacache_controller.begins, [])
        self.assertEqual(net.seacache_controller.computes, [])
        self.assertEqual(net.seacache_controller.ends, [])
        self.assertEqual(net.seacache_controller.resets, [])
        self.assertIsNone(getattr(sampler, "_seacache_batch_context"))
        self.assertEqual(sampler.last_seacache_summary["full_calls"], 2)
        self.assertEqual(sampler.last_seacache_summary["reuse_calls"], 0)

    def test_expected_heun_calls(self):
        class ToyHeun:
            num_steps = 5
            exact_henu = False

        class ToyExactHeun:
            num_steps = 5
            exact_henu = True

        self.assertEqual(expected_model_calls(ToyHeun()), 5)
        self.assertEqual(expected_model_calls(ToyExactHeun()), 9)

    def test_diagnostic_tuple_forces_full(self):
        model = ToyDiagnosticModel()
        model.set_seacache_call_context("diag", solver_stage="euler", macro_step=0)
        with mock.patch.object(
            pixelgen_model,
            "_pixelgen_modulate",
            lambda value, shift, scale: value * (1 + scale.unsqueeze(1))
            + shift.unsqueeze(1),
        ):
            result = model._seacache_forward(
                torch.zeros(2, 1, 2, 2),
                torch.zeros(2),
                torch.zeros(2, dtype=torch.long),
                return_layer=0,
                return_last=True,
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            model.seacache_controller.computes[-1]["force_full_reason"],
            "diagnostic_return",
        )

    def test_patch_grid_validation_precedes_controller(self):
        model = ToyDiagnosticModel()
        model.patch_size = 2
        with self.assertRaisesRegex(ValueError, "divisible"):
            model._seacache_forward(
                torch.zeros(1, 1, 3, 2),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.long),
            )
        self.assertEqual(model.seacache_controller.computes, [])

        model.patch_size = 1
        model.x_embedder = _ThreeTokenEmbed()
        model.pos_embed = torch.zeros(1, 3, 1)
        with self.assertRaisesRegex(ValueError, "embedded token shape"):
            model._seacache_forward(
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.long),
            )
        self.assertEqual(model.seacache_controller.computes, [])

    def test_metadata_requires_manifest_integer_ids(self):
        self.assertEqual(
            sample_ids_from_metadata(
                {"sample_id": torch.tensor([11, 12])}, 2, batch_idx=3
            ),
            [11, 12],
        )
        self.assertEqual(
            sample_ids_from_metadata(
                [{"sample_id": "13"}, {"sample_id": 14}], 2, batch_idx=3
            ),
            [13, 14],
        )
        with self.assertRaisesRegex(ValueError, "manifest-stable"):
            sample_ids_from_metadata(
                {"filename": ["image-a"], "seed": [0]}, 1, batch_idx=3
            )
        with self.assertRaisesRegex(ValueError, "manifest-stable"):
            sample_ids_from_metadata({"sample_id": ["image-a:0"]}, 1, batch_idx=3)


if __name__ == "__main__":
    unittest.main()
