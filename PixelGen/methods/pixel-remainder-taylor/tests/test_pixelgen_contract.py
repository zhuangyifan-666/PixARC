from __future__ import annotations

import copy
import runpy
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from generate_shard import build_resolved_cli_config
from pixel_remainder_taylor.controller import plan_segment
from pixel_remainder_taylor.pixelgen_sampler import (
    PixelRemainderTaylorHeunSampler,
    real_batch_sample_ids,
)
from pixel_remainder_taylor.runtime import PixelRemainderRuntime
from pixel_remainder_taylor.scheduler import expected_network_forward_count


def test_shared_core_resolves_in_pixelgen_package():
    plan = plan_segment(
        [
            torch.ones(2, 3, 16, 16),
            torch.zeros(2, 3, 16, 16),
            torch.full((2, 3, 16, 16), 0.01),
        ],
        feature_available_order_min=1,
        nfe_index=2,
        total_nfe=99,
        tau=1.0,
        max_taylor_span=3,
    )
    assert plan.selected_span == 3


def test_combined_cfg_real_ids_are_not_duplicated_for_controller():
    assert real_batch_sample_ids([3, 4], 2) == (3, 4)
    assert real_batch_sample_ids([3, 4, 3, 4], 2) == (3, 4)
    with pytest.raises(ValueError):
        real_batch_sample_ids([3, 4, 4, 3], 2)
    with pytest.raises(TypeError):
        real_batch_sample_ids([3.5], 1)
    with pytest.raises(ValueError):
        real_batch_sample_ids([], 0)


def test_pixelgen_no_extra_forward_contract():
    class Schedule:
        @staticmethod
        def sigma(t):
            return torch.ones_like(t).reshape(-1, 1, 1, 1)

        @staticmethod
        def dalpha_over_alpha(t):
            return torch.ones_like(t).reshape(-1, 1, 1, 1)

        @staticmethod
        def dsigma_mul_sigma(t):
            return torch.zeros_like(t).reshape(-1, 1, 1, 1)

    class Net:
        def __init__(self, runtime):
            self.pixel_remainder_runtime = runtime
            self.calls = 0

        def forward_taylor(self, x, t, condition, *, stream_id):
            self.calls += 1
            shaped_t = t.reshape(-1, 1, 1, 1).to(x)
            value = self.pixel_remainder_runtime.branch(
                stream_id=stream_id,
                layer_idx=0,
                module_name="dummy",
                exact_fn=lambda: x + (1.0 - shaped_t),
            )
            self.pixel_remainder_runtime.mark_stream_complete(stream_id)
            return value

    sampler = object.__new__(PixelRemainderTaylorHeunSampler)
    sampler.num_steps = 50
    sampler.exact_henu = True
    sampler.timesteps = torch.linspace(0.0, 1.0, 51)
    sampler.scheduler = Schedule()
    sampler.w_scheduler = None
    sampler.step_fn = lambda x, v, dt, **kwargs: x + dt * v
    sampler.last_step_fn = sampler.step_fn
    sampler.guidance = 2.0
    sampler.guidance_interval_min = 0.1
    sampler.guidance_interval_max = 0.9
    sampler.guidance_fn = lambda velocity, scale: (
        velocity[: velocity.shape[0] // 2]
        + scale
        * (
            velocity[velocity.shape[0] // 2 :]
            - velocity[: velocity.shape[0] // 2]
        )
    )
    sampler.t_eps = 1.0e-6
    object.__setattr__(
        sampler,
        "_taylorseer_batch_context",
        {"sample_ids": [3, 4], "trajectory_id": "pixelgen-count"},
    )
    runtime = PixelRemainderRuntime(
        mode="instrumented_full", tau=0.0, max_taylor_span=3
    )
    net = Net(runtime)
    trajectories, _velocities = sampler._impl_sampling(
        net,
        torch.zeros(2, 3, 8, 8),
        torch.ones(2, 1),
        torch.zeros(2, 1),
    )
    assert trajectories[-1].shape == (2, 3, 8, 8)
    assert net.calls == expected_network_forward_count(
        model_family="pixelgen", sampler="heun", num_steps=50
    ) == 99
    assert sampler._last_taylorseer_summary["total_nfe"] == 99
    assert sampler._last_taylorseer_summary["combined_cfg_batch_size"] == 4
    assert sampler._last_taylorseer_summary["stage_statistics"] == {
        "corrector": {"full_nfe": 49, "taylor_nfe": 0},
        "final_euler": {"full_nfe": 1, "taylor_nfe": 0},
        "predictor": {"full_nfe": 49, "taylor_nfe": 0},
    }
    assert not runtime.active


def test_guided_velocity_uses_sampler_epsilon_and_preserves_cfg_order():
    sampler = object.__new__(PixelRemainderTaylorHeunSampler)
    sampler.t_eps = 0.25
    sampler.guidance = 2.0
    sampler.guidance_interval_min = 0.1
    sampler.guidance_interval_max = 0.9
    sampler.guidance_fn = lambda velocity, scale: (
        velocity[:1] + scale * (velocity[1:] - velocity[:1])
    )
    cfg_x = torch.zeros(2, 1, 1, 1)
    raw = torch.tensor([[[[1.0]]], [[[3.0]]]])
    cfg_t = torch.tensor([0.9, 0.9])

    guided = sampler._guided_velocity(raw, cfg_x, cfg_t, torch.tensor([0.5]))

    # 1-t=0.1 is clamped to t_eps=0.25, so [uncond, cond]=[4,12].
    assert torch.equal(guided, torch.tensor([[[[20.0]]]]))


def test_pixelgen_main_binds_model_and_upstream_datamodule(monkeypatch):
    captured = {}
    model_class = type("Model", (), {})
    data_class = type("DataModule", (), {})

    def fake_cli(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    lightning_cli = types.ModuleType("lightning.pytorch.cli")
    lightning_cli.LightningCLI = fake_cli
    lightning_pytorch = types.ModuleType("lightning.pytorch")
    lightning = types.ModuleType("lightning")
    src_data = types.ModuleType("src.lightning_data")
    src_data.DataModule = data_class
    src = types.ModuleType("src")
    model_module = types.ModuleType("pixel_remainder_taylor.pixelgen_lightning")
    model_module.PixelRemainderTaylorLightning = model_class
    for name, module in {
        "lightning": lightning,
        "lightning.pytorch": lightning_pytorch,
        "lightning.pytorch.cli": lightning_cli,
        "src": src,
        "src.lightning_data": src_data,
        "pixel_remainder_taylor.pixelgen_lightning": model_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    entrypoint = Path(__file__).resolve().parents[1] / "scripts" / "pixelgen_main.py"
    runpy.run_path(str(entrypoint), run_name="__main__")
    assert captured["args"] == (model_class, data_class)
    assert captured["kwargs"] == {
        "auto_configure_optimizers": False,
        "save_config_callback": None,
    }


def test_generated_resolved_yaml_routes_prediction_dataset(tmp_path):
    config = {
        "schema_version": "ignored-here",
        "template_only": False,
        "checkpoint": "/checkpoint.ckpt",
        "method": {},
        "runtime": {},
        "trainer": {"logger": True},
        "model": {
            "denoiser": {"class_path": "Denoiser", "init_args": {}},
            "diffusion_sampler": {"class_path": "Sampler", "init_args": {}},
        },
        "data": {"pred_dataset": None},
    }
    resolved = build_resolved_cli_config(
        config,
        trainer_updates={"accelerator": "gpu", "devices": 1},
        denoiser_updates={"method_tau": 0.02, "method_trace_mode": "full"},
        dataset_init_args={
            "manifest_path": "/frozen/manifest_1k.jsonl",
            "shard_id": 2,
            "world_size": 4,
        },
        pred_batch_size=4,
        pred_num_workers=1,
        compile_mode="matched_eager",
    )
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["data"]["pred_dataset"]["class_path"] == (
        "pixel_remainder_taylor.pixelgen_io.ManifestNoiseDataset"
    )
    assert parsed["data"]["pred_dataset"]["init_args"] == {
        "manifest_path": "/frozen/manifest_1k.jsonl",
        "shard_id": 2,
        "world_size": 4,
    }
    assert parsed["data"]["pred_batch_size"] == 4
    assert parsed["model"]["denoiser"]["init_args"]["method_tau"] == 0.02


def test_deepcopy_runtime_is_empty_and_independent():
    runtime = PixelRemainderRuntime(
        mode="pixel_remainder_taylor", tau=0.02, max_taylor_span=3
    )
    clone = copy.deepcopy(runtime)
    assert clone is not runtime
    assert clone.pixel_history.available_order == -1
    assert not clone.active


def test_runtime_is_not_persistent_module_state():
    module = nn.Linear(2, 2)
    runtime = PixelRemainderRuntime(
        mode="pixel_remainder_taylor", tau=0.02, max_taylor_span=3
    )
    object.__setattr__(module, "pixel_remainder_runtime", runtime)
    assert not any("pixel" in key or "remainder" in key for key in module.state_dict())
