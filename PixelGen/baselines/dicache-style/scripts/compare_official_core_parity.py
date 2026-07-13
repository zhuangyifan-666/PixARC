#!/usr/bin/env python3
"""CPU-only parity check against audited finite FLUX DiCache formulas.

This script intentionally re-expresses only the small observable equations and
branch boundaries; it does not import or copy an official model implementation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_ROOT = PIXARC_ROOT / "baselines" / "DiCache"
OFFICIAL_FLUX = OFFICIAL_ROOT / "FLUX" / "run_flux_dicache.py"
EXPECTED_COMMIT = "fdbe20b669c9174bbed5ec994de073fd881c8010"
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.anchors import AnchorWindow  # noqa: E402
from dicache_style.dcta import estimate_residual  # noqa: E402
from dicache_style.errors import compute_probe_error  # noqa: E402
from dicache_style.gate import (  # noqa: E402
    FULL_RESUME_FROM_PROBE,
    REUSE,
    flux_direct_full_reason,
    strict_accumulated_gate,
)


def _official_revision() -> str:
    result = subprocess.run(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _check_audited_source() -> None:
    source = OFFICIAL_FLUX.read_text(encoding="utf-8")
    required = (
        "self.cnt <= int(self.ret_ratio * self.num_steps)",
        "self.cnt == self.num_steps - 1",
        "delta_x = (hidden_states - self.previous_input).abs().mean()",
        "delta_y = (test_hidden_states - self.previous_probe_states).abs().mean()",
        "self.accumulated_rel_l1_distance < self.rel_l1_thresh",
        ").clip(1, 1.5)",
        "self.residual_window[-2] + gamma * (self.residual_window[-1]",
    )
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise RuntimeError(f"audited released FLUX expressions changed: {missing}")


def _differences(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    absolute = (left.float() - right.float()).abs()
    maximum_absolute = float(absolute.max().item())
    scale = right.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
    maximum_relative = float((absolute / scale).max().item())
    return maximum_absolute, maximum_relative


def _anchor_window(dtype: torch.dtype) -> AnchorWindow:
    window = AnchorWindow()
    for index, (full, probe) in enumerate(((2.0, 1.0), (4.0, 3.0))):
        window.append_exact(
            full_residual=torch.full((2, 3, 4), full, dtype=dtype),
            probe_residual=torch.full((2, 3, 4), probe, dtype=dtype),
            nfe_index=index,
            stream_call_index=index,
            continuous_t=float(index),
            solver_stage="predictor",
        )
    return window


def main() -> None:
    revision = _official_revision()
    if revision != EXPECTED_COMMIT:
        raise RuntimeError(
            f"official DiCache revision changed: {revision} != {EXPECTED_COMMIT}"
        )
    _check_audited_source()
    torch.manual_seed(20260314)
    absolute_errors: dict[str, float] = {}
    relative_errors: dict[str, float] = {}

    def record(name: str, actual: torch.Tensor, reference: torch.Tensor) -> None:
        absolute, relative = _differences(actual, reference)
        absolute_errors[name] = absolute
        relative_errors[name] = relative

    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).removeprefix("torch.")
        previous_body = torch.randn(2, 5, 7, dtype=dtype) + 0.25
        body = previous_body + torch.randn_like(previous_body) * 0.03
        previous_probe = torch.randn(2, 5, 7, dtype=dtype) + 0.5
        probe = previous_probe + torch.randn_like(previous_probe) * 0.05
        local = compute_probe_error(
            body,
            previous_body,
            probe,
            previous_probe,
            error_choice="delta_minus",
            numeric_mode="official_no_epsilon",
        )
        reference_dx = (body - previous_body).abs().mean() / previous_body.abs().mean()
        reference_dy = (probe - previous_probe).abs().mean() / previous_probe.abs().mean()
        record(f"{name}.delta_x", local.delta_x, reference_dx)
        record(f"{name}.delta_y", local.delta_y, reference_dy)
        record(
            f"{name}.delta_minus", local.error,
            (reference_dy - reference_dx).abs(),
        )

        body_input = torch.zeros(2, 3, 4, dtype=dtype)
        current_probe = torch.full_like(body_input, 4.0)
        window = _anchor_window(dtype)
        result = estimate_residual(
            body_input,
            current_probe,
            window,
            gamma_min=1.0,
            gamma_max=1.5,
            numeric_mode="official_no_epsilon",
            gamma_nonfinite_policy="official_propagate",
        )
        old, new = window.last_two
        current_probe_residual = current_probe - body_input
        reference_gamma = (
            (current_probe_residual - old.probe_residual).abs().mean()
            / (new.probe_residual - old.probe_residual).abs().mean()
        ).clamp(1.0, 1.5)
        reference_residual = old.full_residual + reference_gamma * (
            new.full_residual - old.full_residual
        )
        record(f"{name}.gamma", result.gamma, reference_gamma)
        record(
            f"{name}.dcta_residual",
            result.estimated_residual,
            reference_residual,
        )

    gate_cases = (
        (0.0, 0.2, 0.3, REUSE),
        (0.1, 0.2, 0.3, FULL_RESUME_FROM_PROBE),
        (0.2, 0.2, 0.3, FULL_RESUME_FROM_PROBE),
    )
    for accumulator, error, threshold, expected in gate_cases:
        actual = strict_accumulated_gate(
            accumulator, torch.tensor(error), threshold
        ).action
        if actual != expected:
            raise AssertionError((accumulator, error, threshold, actual, expected))

    warmup_cases = {
        "30@0.2": [index for index in range(30) if flux_direct_full_reason(
            call_index=index, total_calls=30, ret_ratio=0.2,
            force_last_full=False,
        )],
        "99@0.2": [index for index in range(99) if flux_direct_full_reason(
            call_index=index, total_calls=99, ret_ratio=0.2,
            force_last_full=False,
        )],
        "30@0": [index for index in range(30) if flux_direct_full_reason(
            call_index=index, total_calls=30, ret_ratio=0.0,
            force_last_full=False,
        )],
        "30@1": [index for index in range(30) if flux_direct_full_reason(
            call_index=index, total_calls=30, ret_ratio=1.0,
            force_last_full=False,
        )],
    }
    expected_warmup_counts = {"30@0.2": 7, "99@0.2": 20, "30@0": 1, "30@1": 30}
    actual_warmup_counts = {key: len(value) for key, value in warmup_cases.items()}
    if actual_warmup_counts != expected_warmup_counts:
        raise AssertionError((actual_warmup_counts, expected_warmup_counts))

    maximum_absolute = max(absolute_errors.values(), default=0.0)
    maximum_relative = max(relative_errors.values(), default=0.0)
    report = {
        "device": "cpu",
        "dicache_commit": revision,
        "audited_source": str(OFFICIAL_FLUX),
        "numeric_mode": "official_no_epsilon",
        "per_check_max_abs_error": absolute_errors,
        "per_check_max_relative_error": relative_errors,
        "overall_max_abs_error": maximum_absolute,
        "overall_max_relative_error": maximum_relative,
        "strict_gate_cases": len(gate_cases),
        "warmup_full_counts": actual_warmup_counts,
        "passed": maximum_absolute == 0.0 and maximum_relative == 0.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if maximum_absolute != 0.0 or maximum_relative != 0.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
