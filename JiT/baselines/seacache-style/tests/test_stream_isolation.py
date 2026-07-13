"""CPU-only tests for JiT conditional/unconditional cache isolation."""

from __future__ import annotations

from collections import Counter

import pytest
import torch
import torch.nn as nn

from seacache_style.jit_denoiser import (
    SeaCacheDenoiser,
    expected_model_calls_per_stream,
)


class RecordingNet(nn.Module):
    def __init__(self, fail_at_call=None):
        super().__init__()
        self.calls = []
        self.fail_at_call = fail_at_call

    def forward(
        self,
        z,
        t,
        labels,
        *,
        cache_stream,
        solver_stage="",
        macro_step=None,
    ):
        self.calls.append(
            {
                "stream": cache_stream,
                "labels": labels.detach().clone(),
                "stage": solver_stage,
                "macro_step": macro_step,
            }
        )
        if self.fail_at_call is not None and len(self.calls) == self.fail_at_call:
            raise RuntimeError("injected model failure")
        # x_pred == z produces a zero velocity, making output comparison exact.
        return z


class RecordingController:
    def __init__(self):
        self.begins = []
        self.ends = []
        self.resets = []

    def begin_trajectory(self, **kwargs):
        self.begins.append(kwargs.copy())

    def end_trajectory(self, stream_id, require_complete=True):
        self.ends.append((stream_id, require_complete))

    def reset(self, stream_id):
        self.resets.append(stream_id)


def make_mock_denoiser(*, method="heun", steps=3, mode="seacache", fail_at=None):
    # Bypass the real constructor: upstream VisionRotaryEmbeddingFast creates
    # CUDA tensors during model construction.  This test exercises only the
    # denoiser/sampler lifecycle with an nn.Module mock.
    denoiser = SeaCacheDenoiser.__new__(SeaCacheDenoiser)
    nn.Module.__init__(denoiser)
    denoiser.net = RecordingNet(fail_at_call=fail_at)
    denoiser.img_size = 4
    denoiser.num_classes = 1000
    denoiser.t_eps = 0.05
    denoiser.noise_scale = 1.0
    denoiser.method = method
    denoiser.steps = steps
    denoiser.cfg_scale = 3.0
    denoiser.cfg_interval = (0.1, 1.0)
    controller = RecordingController()
    object.__setattr__(denoiser, "_seacache_controller", controller)
    object.__setattr__(denoiser, "_seacache_mode", mode)
    object.__setattr__(denoiser, "_trajectory_serial", 0)
    object.__setattr__(denoiser, "_active_stream_call_counts", None)
    return denoiser, controller


@pytest.mark.parametrize(
    ("method", "steps", "expected"),
    [("euler", 50, 50), ("heun", 50, 99), ("heun", 1, 1)],
)
def test_expected_model_calls_are_derived_from_sampler(method, steps, expected):
    assert expected_model_calls_per_stream(steps, method) == expected


def test_cond_and_uncond_streams_are_separate_and_ordered(monkeypatch):
    denoiser, controller = make_mock_denoiser(method="heun", steps=3)
    labels = torch.tensor([4, 9], dtype=torch.long)
    noise = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    def unexpected_randn(*args, **kwargs):
        raise AssertionError("explicit noise must bypass torch.randn")

    monkeypatch.setattr(torch, "randn", unexpected_randn)
    output = denoiser.generate(
        labels,
        noise=noise,
        sample_ids=(17, 23),
        trajectory_id="toy-trajectory",
    )

    assert torch.equal(output, noise)
    expected_per_stream = expected_model_calls_per_stream(3, "heun")
    assert Counter(call["stream"] for call in denoiser.net.calls) == {
        "cond": expected_per_stream,
        "uncond": expected_per_stream,
    }
    assert [call["stream"] for call in denoiser.net.calls] == [
        stream
        for _ in range(expected_per_stream)
        for stream in ("cond", "uncond")
    ]

    cond_calls = [call for call in denoiser.net.calls if call["stream"] == "cond"]
    uncond_calls = [
        call for call in denoiser.net.calls if call["stream"] == "uncond"
    ]
    assert all(torch.equal(call["labels"], labels) for call in cond_calls)
    assert all(
        torch.equal(call["labels"], torch.full_like(labels, 1000))
        for call in uncond_calls
    )

    assert [entry["stream_id"] for entry in controller.begins] == [
        "cond",
        "uncond",
    ]
    assert all(entry["total_calls"] == expected_per_stream for entry in controller.begins)
    assert all(entry["trajectory_id"] == "toy-trajectory" for entry in controller.begins)
    assert all(entry["sample_ids"] == (17, 23) for entry in controller.begins)
    assert controller.ends == [("cond", True), ("uncond", True)]
    assert controller.resets == ["cond", "uncond"]


def test_solver_stages_match_heun_and_final_euler():
    denoiser, _controller = make_mock_denoiser(method="heun", steps=2)
    labels = torch.tensor([1], dtype=torch.long)
    noise = torch.zeros(1, 3, 4, 4)

    denoiser.generate(labels, noise=noise)
    cond_stages = [
        call["stage"] for call in denoiser.net.calls if call["stream"] == "cond"
    ]
    assert cond_stages == ["predictor", "corrector", "final_euler"]


def test_exception_resets_both_streams_without_ending_incomplete_trajectory():
    denoiser, controller = make_mock_denoiser(
        method="heun", steps=2, fail_at=2
    )
    labels = torch.tensor([7], dtype=torch.long)
    noise = torch.zeros(1, 3, 4, 4)

    with pytest.raises(RuntimeError, match="injected model failure"):
        denoiser.generate(labels, noise=noise)

    assert [entry["stream_id"] for entry in controller.begins] == [
        "cond",
        "uncond",
    ]
    assert controller.ends == []
    assert controller.resets == ["cond", "uncond"]
    assert denoiser._active_stream_call_counts is None


def test_full_mode_does_not_touch_controller_lifecycle():
    denoiser, controller = make_mock_denoiser(
        method="euler", steps=2, mode="full"
    )
    labels = torch.tensor([2], dtype=torch.long)
    noise = torch.zeros(1, 3, 4, 4)

    denoiser.generate(labels, noise=noise)

    assert controller.begins == []
    assert controller.ends == []
    assert controller.resets == []
    assert denoiser._last_seacache_summaries["cond"]["full_calls"] == 2
    assert denoiser._last_seacache_summaries["uncond"]["full_calls"] == 2
    assert denoiser._last_seacache_summaries["cond"]["reuse_calls"] == 0
