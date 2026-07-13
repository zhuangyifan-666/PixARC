#!/usr/bin/env python3
"""CPU parity fixtures for the audited released DiCache FLUX core formulas.

This intentionally does not import the released FLUX demo because that module
loads Diffusers models at import time.  It pins the local clone revision,
checks the audited source expressions are still present, and evaluates those
expressions directly against this port on deterministic tensors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from dicache_style.anchors import AnchorWindow
from dicache_style.dcta import estimate_residual
from dicache_style.errors import compute_probe_error
from dicache_style.gate import (
    FULL_RESUME_FROM_PROBE,
    REUSE,
    flux_direct_full_reason,
    strict_accumulated_gate,
)


PIXARC_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_ROOT = PIXARC_ROOT / "baselines" / "DiCache"
OFFICIAL_FLUX = OFFICIAL_ROOT / "FLUX" / "run_flux_dicache.py"
EXPECTED_COMMIT = "fdbe20b669c9174bbed5ec994de073fd881c8010"


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


def _append_anchor(
    window: AnchorWindow,
    *,
    full: torch.Tensor,
    probe: torch.Tensor,
    index: int,
) -> None:
    window.append_exact(
        full_residual=full,
        probe_residual=probe,
        nfe_index=index,
        stream_call_index=index,
        continuous_t=float(index),
        solver_stage="fixture",
    )


def _max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Elementwise relative error with a finite machine-epsilon denominator."""

    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    return float(((actual - expected).abs() / denominator).max())


def run_parity() -> dict[str, object]:
    revision = _official_revision()
    if revision != EXPECTED_COMMIT:
        raise RuntimeError(
            f"official DiCache revision changed: {revision} != {EXPECTED_COMMIT}"
        )
    _check_audited_source()

    generator = torch.Generator(device="cpu").manual_seed(20260713)
    previous_x = torch.randn((2, 7, 5), generator=generator)
    current_x = previous_x + 0.1 * torch.randn(previous_x.shape, generator=generator)
    previous_y = torch.randn((2, 7, 5), generator=generator)
    current_y = previous_y + 0.1 * torch.randn(previous_y.shape, generator=generator)
    measured = compute_probe_error(
        current_x,
        previous_x,
        current_y,
        previous_y,
        error_choice="delta_y",
        numeric_mode="official_no_epsilon",
    )
    official_dx = (current_x - previous_x).abs().mean() / previous_x.abs().mean()
    official_dy = (current_y - previous_y).abs().mean() / previous_y.abs().mean()
    error_max_abs = max(
        float((measured.delta_x - official_dx).abs()),
        float((measured.delta_y - official_dy).abs()),
        float((measured.error - official_dy).abs()),
    )
    error_max_rel = max(
        _max_relative_error(measured.delta_x, official_dx),
        _max_relative_error(measured.delta_y, official_dy),
        _max_relative_error(measured.error, official_dy),
    )

    threshold = float(official_dy)
    equality = strict_accumulated_gate(0.0, official_dy, threshold)
    below = strict_accumulated_gate(0.0, official_dy, threshold + 1e-4)
    if equality.action != FULL_RESUME_FROM_PROBE or below.action != REUSE:
        raise AssertionError("strict released gate parity failed")

    warmup_mismatches = 0
    for total_calls in (30, 99):
        for call_index in range(total_calls):
            official_full = (
                call_index <= int(0.2 * total_calls)
                or call_index == total_calls - 1
            )
            port_full = (
                flux_direct_full_reason(
                    call_index=call_index,
                    total_calls=total_calls,
                    ret_ratio=0.2,
                    force_last_full=True,
                )
                is not None
            )
            warmup_mismatches += int(official_full != port_full)

    anchors = AnchorWindow()
    old_full = torch.randn((2, 7, 5), generator=generator)
    new_full = torch.randn((2, 7, 5), generator=generator)
    old_probe = torch.randn((2, 7, 5), generator=generator)
    new_probe = old_probe + torch.randn(old_probe.shape, generator=generator)
    _append_anchor(anchors, full=old_full, probe=old_probe, index=0)
    _append_anchor(anchors, full=new_full, probe=new_probe, index=1)
    body_input = torch.randn((2, 7, 5), generator=generator)
    current_probe_residual = old_probe + 3.0 * (new_probe - old_probe)
    probe_feature = body_input + current_probe_residual
    result = estimate_residual(
        body_input,
        probe_feature,
        anchors,
        gamma_min=1.0,
        gamma_max=1.5,
        numeric_mode="official_no_epsilon",
        gamma_nonfinite_policy="official_propagate",
    )
    official_gamma = (
        (current_probe_residual - old_probe).abs().mean()
        / (new_probe - old_probe).abs().mean()
    ).clip(1, 1.5)
    official_residual = old_full + official_gamma * (new_full - old_full)
    dcta_max_abs = float((result.estimated_residual - official_residual).abs().max())
    dcta_max_rel = _max_relative_error(
        result.estimated_residual, official_residual
    )

    if (
        error_max_abs != 0.0
        or error_max_rel != 0.0
        or dcta_max_abs != 0.0
        or dcta_max_rel != 0.0
        or warmup_mismatches
    ):
        raise AssertionError("released-core parity was not bit-exact on CPU fixtures")
    return {
        "status": "ok",
        "official_commit": revision,
        "official_source": str(OFFICIAL_FLUX),
        "error_max_abs": error_max_abs,
        "error_max_rel": error_max_rel,
        "dcta_max_abs": dcta_max_abs,
        "dcta_max_rel": dcta_max_rel,
        "global_max_abs": max(error_max_abs, dcta_max_abs),
        "global_max_rel": max(error_max_rel, dcta_max_rel),
        "warmup_mismatches": warmup_mismatches,
        "strict_equality_action": equality.action,
        "strict_below_action": below.action,
        "device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    report = run_parity()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
