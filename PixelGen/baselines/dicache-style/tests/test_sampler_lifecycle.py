import importlib.util
from importlib.machinery import ModuleSpec
import sys
import types
from types import SimpleNamespace

import torch

if importlib.util.find_spec("lightning") is None:
    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    lightning.__spec__ = ModuleSpec("lightning", loader=None, is_package=True)
    pytorch.__spec__ = ModuleSpec(
        "lightning.pytorch", loader=None, is_package=True
    )
    pytorch.Callback = type("Callback", (), {})
    lightning.pytorch = pytorch
    sys.modules["lightning"] = lightning
    sys.modules["lightning.pytorch"] = pytorch

import dicache_style.pixelgen_lightning as pixelgen_lightning
from dicache_style.pixelgen_io import _scalar_summary
from dicache_style.pixelgen_sampler import DiCacheHeunSamplerJiT
from dicache_style.runtime import DiCacheRuntime
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.sampling import ode_step_fn
from src.diffusion.flow_matching.scheduling import LinearScheduler


class _InstrumentedFakeNet:
    def __init__(self):
        self.dicache_runtime = DiCacheRuntime(
            mode="instrumented_full",
            rel_l1_thresh=None,
        )
        self.shapes = []

    def forward_dicache(self, x, t, condition, *, stream_id):
        self.shapes.append((tuple(x.shape), tuple(condition.shape)))
        plan = self.dicache_runtime.plan_stream_call(stream_id, x)
        self.dicache_runtime.complete_full(
            plan=plan,
            body_input=x,
            probe_feature=x + 1,
            exact_body_output=x + 2,
            resumed=False,
        )
        return x


class _UpstreamFakeNet:
    def __init__(self):
        self.dicache_runtime = DiCacheRuntime(
            mode="upstream_full",
            rel_l1_thresh=None,
        )
        self.calls = 0

    def __call__(self, x, _t, _condition):
        self.calls += 1
        return x


def _sampler():
    return DiCacheHeunSamplerJiT(
        scheduler=LinearScheduler(),
        w_scheduler=None,
        exact_henu=True,
        num_steps=3,
        guidance=2.25,
        timeshift=2.0,
        guidance_interval_min=0.1,
        guidance_interval_max=0.9,
        guidance_fn=simple_guidance_fn,
        step_fn=ode_step_fn,
    )


def test_exact_heun_sampler_scopes_one_combined_runtime_per_batch():
    sampler = _sampler()
    net = _InstrumentedFakeNet()
    noise = torch.randn(1, 3, 2, 2)
    condition = torch.randn(1, 4)
    uncondition = torch.randn(1, 4)
    sampler.set_dicache_batch_context(
        sample_ids=(17,), trajectory_id="manifest-group:test"
    )

    output = sampler(net, noise, condition, uncondition)

    assert output.shape == noise.shape
    assert len(net.shapes) == 5
    assert all(x_shape[0] == 2 and condition_shape[0] == 2 for x_shape, condition_shape in net.shapes)
    summary = sampler.last_dicache_summary
    assert summary["total_nfe"] == 5
    assert summary["network_forward_count"] == 5
    assert summary["total_stream_calls"] == 5
    assert summary["direct_full_count"] == 5
    assert summary["effective_cfg_batch_size"] == 2
    assert summary["peak_memory_allocated"] == 0
    assert summary["peak_memory_reserved"] == 0
    assert summary["call_count_valid"] is True
    assert not net.dicache_runtime.active
    assert net.dicache_runtime.tensor_count() == 0
    assert getattr(sampler, "_dicache_batch_context", None) is None


def test_upstream_full_predict_step_persists_manifest_identity_for_save_hook(
    monkeypatch,
):
    sampler = _sampler()
    net = _UpstreamFakeNet()
    module = SimpleNamespace(
        ema_denoiser=net,
        diffusion_sampler=sampler,
        conditioner=lambda labels: (
            torch.zeros(labels.shape[0], 4),
            torch.ones(labels.shape[0], 4),
        ),
        vae=SimpleNamespace(decode=lambda value: value),
    )
    noise = torch.randn(1, 3, 2, 2)
    labels = torch.tensor([5])
    metadata = {
        "sample_id": torch.tensor([23]),
        "batch_group_id": ["rank-0:batch-7"],
    }

    monkeypatch.setattr(
        pixelgen_lightning,
        "fp2uint8",
        lambda value: torch.clamp((value + 1) * 127.5 + 0.5, 0, 255).to(
            torch.uint8
        ),
    )
    output = pixelgen_lightning.DiCachePixelGenLightning.predict_step(
        module, (noise, labels, metadata), batch_idx=7
    )

    assert output.shape == noise.shape
    assert output.dtype == torch.uint8
    assert net.calls == 5
    summary = _scalar_summary(sampler.last_dicache_summary)
    assert summary["trajectory_id"] == "manifest-group:rank-0:batch-7"
    assert summary["sample_ids"] == [23]
    assert summary["real_batch_size"] == 1
    assert summary["effective_cfg_batch_size"] == 2
    assert summary["call_count_valid"] is True
    assert getattr(sampler, "_dicache_batch_context", None) is None
