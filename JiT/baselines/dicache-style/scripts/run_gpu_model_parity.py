#!/usr/bin/env python3
"""Deferred JiT parity: upstream Full vs exact probe/resume Full."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dicache_style.metadata import atomic_write_json
from dicache_style.source_identity import (
    release_source_bindings,
    require_source_identity_current,
)


BASELINE_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = BASELINE_ROOT.parents[2] / "third-party" / "JiT"


def tensor_sequence_metrics(
    candidate: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]
) -> dict[str, float]:
    if len(candidate) != len(reference) or not reference:
        raise ValueError("parity tensor sequences must have equal nonzero length")
    max_abs = 0.0
    absolute_sum = 0.0
    reference_l1 = 0.0
    squared_sum = 0.0
    reference_squared_sum = 0.0
    elements = 0
    for actual, expected in zip(candidate, reference, strict=True):
        if actual.shape != expected.shape:
            raise ValueError(
                f"parity tensor shapes differ: {tuple(actual.shape)} != "
                f"{tuple(expected.shape)}"
            )
        difference = actual.float() - expected.float()
        absolute = difference.abs()
        max_abs = max(max_abs, float(absolute.max().item()))
        absolute_sum += float(absolute.sum().item())
        reference_l1 += float(expected.float().abs().sum().item())
        squared_sum += float(difference.square().sum().item())
        reference_squared_sum += float(expected.float().square().sum().item())
        elements += expected.numel()
    tiny = torch.finfo(torch.float32).tiny
    return {
        "max_absolute_error": max_abs,
        "mean_absolute_error": absolute_sum / elements,
        "relative_l1_error": absolute_sum / max(reference_l1, tiny),
        "relative_l2_error": (squared_sum**0.5)
        / max(reference_squared_sum**0.5, tiny),
    }


def tensor_sequence_allclose(
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> bool:
    if len(candidate) != len(reference) or not reference:
        return False
    return all(
        actual.shape == expected.shape
        and torch.allclose(actual, expected, atol=atol, rtol=rtol)
        for actual, expected in zip(candidate, reference, strict=True)
    )


def evaluate_invariants(
    *,
    upstream: Mapping[str, Any],
    resumed: Mapping[str, Any],
    expected_nfe: int,
    expected_forwards: int,
    probe_feature_allclose: bool,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Evaluate the attachment's nine resume invariants and lifecycle gates."""

    upstream_summary = upstream["summary"]
    resumed_summary = resumed["summary"]
    upstream_diag = upstream["diagnostics"]
    resumed_diag = resumed["diagnostics"]
    depth = int(resumed_diag["depth"])
    expected_block_counts = [expected_forwards] * depth
    all_blocks_once = (
        list(resumed_diag["block_call_counts"]) == expected_block_counts
        and int(resumed_diag["completed_block_orders"]) == expected_forwards
    )
    direct_full_count = int(resumed_summary.get("direct_full_count", -1))
    resumed_full_count = int(resumed_summary.get("resumed_full_count", -1))
    anchor_count = int(resumed_diag["anchor_check_count"])
    anchor_stream_counts = dict(resumed_diag["anchor_stream_counts"])
    nine = {
        "1_probe_blocks_execute_once": all_blocks_once,
        "2_suffix_resumes_from_correct_block": (
            bool(resumed_diag["per_forward_block_order_valid"])
            and all_blocks_once
            and resumed_full_count > 0
        ),
        "3_context_tokens_not_inserted_twice": bool(
            resumed_diag["block_token_layout_valid"]
        ),
        "4_context_tokens_not_removed_early": (
            bool(resumed_diag["block_token_layout_valid"])
            and bool(resumed_diag["final_head_image_tokens_valid"])
        ),
        "5_correct_rope_used": bool(
            upstream_diag["correct_rope_for_every_block"]
        )
        and bool(resumed_diag["correct_rope_for_every_block"]),
        "6_probe_feature_matches_full_same_layer": probe_feature_allclose,
        "7_probe_capture_keeps_inference_gradients_disabled": (
            bool(upstream_diag["gradient_disabled_during_capture"])
            and bool(resumed_diag["gradient_disabled_during_capture"])
            and bool(upstream_diag["gradient_state_restored"])
            and bool(resumed_diag["gradient_state_restored"])
            and bool(upstream_diag["inference_mode_enabled_during_capture"])
            and bool(resumed_diag["inference_mode_enabled_during_capture"])
            and bool(upstream_diag["inference_state_restored"])
            and bool(resumed_diag["inference_state_restored"])
        ),
        "8_final_head_executes_once": (
            int(upstream_diag["final_head_calls"]) == expected_forwards
            and int(resumed_diag["final_head_calls"]) == expected_forwards
            and len(upstream["body_outputs"]) == expected_forwards
            and len(resumed["body_outputs"]) == expected_forwards
        ),
        "9_exact_anchor_uses_current_body_input": (
            anchor_count == direct_full_count + resumed_full_count
            and anchor_count == expected_forwards
            and anchor_stream_counts
            == {"cond": expected_nfe, "uncond": expected_nfe}
            and bool(resumed_diag["all_full_anchors_use_current_body_input"])
            and bool(resumed_diag["all_probe_anchors_use_current_body_input"])
            and int(resumed_diag["distinguishable_anchor_check_count"]) > 0
            and int(resumed_diag["wrong_probe_baseline_match_count"]) == 0
        ),
    }
    operational = {
        "upstream_mode": upstream_summary.get("mode") == "upstream_full",
        "probe_only_mode": resumed_summary.get("mode") == "probe_only_ablation",
        "upstream_nfe_count": int(upstream_summary.get("total_nfe", -1))
        == expected_nfe,
        "probe_only_nfe_count": int(resumed_summary.get("total_nfe", -1))
        == expected_nfe,
        "upstream_stream_call_count": int(
            upstream_summary.get("total_stream_calls", -1)
        )
        == expected_forwards,
        "probe_only_stream_call_count": int(
            resumed_summary.get("total_stream_calls", -1)
        )
        == expected_forwards,
        "upstream_forward_count": int(
            upstream_summary.get("network_forward_count", -1)
        )
        == expected_forwards,
        "probe_only_forward_count": int(
            resumed_summary.get("network_forward_count", -1)
        )
        == expected_forwards,
        "upstream_blocks_execute_once": list(upstream_diag["block_call_counts"])
        == expected_block_counts,
        "probe_only_all_calls_are_exact_full": (
            direct_full_count + resumed_full_count == expected_forwards
            and int(resumed_summary.get("reuse_count", -1)) == 0
        ),
        "probe_path_was_exercised": (
            int(resumed_summary.get("probe_count", -1)) > 0
            and resumed_full_count > 0
        ),
        "upstream_call_count_valid": bool(
            upstream_summary.get("call_count_valid", False)
        ),
        "probe_only_call_count_valid": bool(
            resumed_summary.get("call_count_valid", False)
        ),
        "upstream_runtime_reset": not bool(upstream["runtime_active_after"]),
        "probe_only_runtime_reset": not bool(resumed["runtime_active_after"]),
        "upstream_cache_is_zero": (
            int(upstream["cache_bytes_after"]) == 0
            and int(upstream["cache_tensor_count_after"]) == 0
        ),
        "probe_only_cache_is_zero": (
            int(resumed["cache_bytes_after"]) == 0
            and int(resumed["cache_tensor_count_after"]) == 0
        ),
    }
    return nine, operational


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-group-id", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    arguments = parser.parse_args()
    if (
        not math.isfinite(arguments.atol)
        or not math.isfinite(arguments.rtol)
        or arguments.atol < 0
        or arguments.rtol < 0
    ):
        parser.error("--atol and --rtol must be finite and non-negative")
    if os.environ.get("DICACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("GPU parity requires DICACHE_GPU_TESTS_ALLOWED=1")
    source = release_source_bindings(BASELINE_ROOT, UPSTREAM_ROOT)

    # Deferred import keeps metric/invariant CPU tests independent of the
    # upstream JiT CUDA construction path.
    from dicache_style.jit_benchmark import build_benchmark_spec
    from dicache_style.runtime import expected_forward_count, expected_nfe_count

    spec = build_benchmark_spec(
        {
            "model_config": arguments.model_config,
            "batch_size": 1,
            "sample_ids": [arguments.sample_id],
            "seeds": [arguments.seed],
            "class_ids": [arguments.class_id],
            "manifest": arguments.manifest,
            "batch_group_id": arguments.batch_group_id,
        }
    )
    upstream = spec.full.upstream_body_and_image()  # type: ignore[attr-defined]
    resumed = spec.full.probe_only_body_and_image()  # type: ignore[attr-defined]
    candidate_sanity = spec.dicache.raw_finiteness()  # type: ignore[attr-defined]

    sampler = str(spec.metadata.get("sampler"))
    steps = int(spec.metadata.get("steps", -1))
    exact_heun = bool(spec.metadata.get("exact_heun", False))
    expected_nfe = expected_nfe_count(sampler, steps, exact_heun=exact_heun)
    expected_forwards = expected_forward_count(
        model_family="jit",
        sampler=sampler,
        num_steps=steps,
        exact_heun=exact_heun,
    )
    upstream_summary = upstream["summary"]
    resumed_summary = resumed["summary"]
    if int(upstream_summary.get("total_nfe", -1)) != expected_nfe:
        raise RuntimeError("upstream summary differs from configured sampler NFE")
    if (
        int(upstream_summary.get("expected_network_forward_count", -1))
        != expected_forwards
    ):
        raise RuntimeError("upstream summary differs from configured forward count")
    if (
        int(resumed_summary.get("expected_network_forward_count", -1))
        != expected_forwards
    ):
        raise RuntimeError("probe-only summary differs from configured forward count")

    body_metrics = tensor_sequence_metrics(
        resumed["body_outputs"], upstream["body_outputs"]
    )
    probe_metrics = tensor_sequence_metrics(
        resumed["probe_features"], upstream["probe_features"]
    )
    sample_metrics = tensor_sequence_metrics(
        [resumed["sample"]], [upstream["sample"]]
    )
    body_allclose = tensor_sequence_allclose(
        resumed["body_outputs"],
        upstream["body_outputs"],
        atol=arguments.atol,
        rtol=arguments.rtol,
    )
    probe_allclose = tensor_sequence_allclose(
        resumed["probe_features"],
        upstream["probe_features"],
        atol=arguments.atol,
        rtol=arguments.rtol,
    )
    sample_allclose = tensor_sequence_allclose(
        [resumed["sample"]],
        [upstream["sample"]],
        atol=arguments.atol,
        rtol=arguments.rtol,
    )
    nine, operational = evaluate_invariants(
        upstream=upstream,
        resumed=resumed,
        expected_nfe=expected_nfe,
        expected_forwards=expected_forwards,
        probe_feature_allclose=probe_allclose,
    )
    candidate_summary = candidate_sanity["summary"]
    operational.update(
        {
            "configured_sampler_is_positive_step_exact_heun": (
                sampler == "heun" and exact_heun and steps > 0
            ),
            "upstream_sample_is_finite": bool(
                torch.isfinite(upstream["sample"]).all().item()
            ),
            "probe_only_sample_is_finite": bool(
                torch.isfinite(resumed["sample"]).all().item()
            ),
            "candidate_sample_is_finite_before_uint8": bool(
                candidate_sanity["sample_finite"]
            ),
            "candidate_nfe_count": int(candidate_summary.get("total_nfe", -1))
            == expected_nfe,
            "candidate_stream_call_count": int(
                candidate_summary.get("total_stream_calls", -1)
            )
            == expected_forwards,
            "candidate_forward_count": int(
                candidate_summary.get("network_forward_count", -1)
            )
            == expected_forwards,
            "candidate_call_count_valid": bool(
                candidate_summary.get("call_count_valid", False)
            ),
            "candidate_runtime_reset": not bool(
                candidate_sanity["runtime_active_after"]
            ),
            "candidate_cache_is_zero": (
                int(candidate_sanity["cache_bytes_after"]) == 0
                and int(candidate_sanity["cache_tensor_count_after"]) == 0
            ),
        }
    )
    passed = bool(
        body_allclose
        and probe_allclose
        and sample_allclose
        and all(nine.values())
        and all(operational.values())
    )
    report = {
        "schema_version": "pixarc-jit-dicache-resume-parity-v2",
        "passed": passed,
        "source": source,
        "comparison": "upstream_full_from_scratch_vs_probe_only_ablation",
        "body_final_layer_inputs": body_metrics,
        "probe_depth_features": probe_metrics,
        "final_sample": sample_metrics,
        "body_allclose": body_allclose,
        "probe_feature_allclose": probe_allclose,
        "final_sample_allclose": sample_allclose,
        "nine_resume_invariants": nine,
        "operational_invariants": operational,
        "expected_nfe": expected_nfe,
        "expected_network_forwards": expected_forwards,
        "upstream_diagnostics": upstream["diagnostics"],
        "probe_only_diagnostics": resumed["diagnostics"],
        "upstream_summary": upstream_summary,
        "probe_only_summary": resumed_summary,
        "candidate_raw_finiteness": candidate_sanity,
        "atol": arguments.atol,
        "rtol": arguments.rtol,
        "sample_id": arguments.sample_id,
        "seed": arguments.seed,
        "class_id": arguments.class_id,
    }
    require_source_identity_current(
        source,
        BASELINE_ROOT,
        UPSTREAM_ROOT,
        context="JiT GPU parity evidence generation",
    )
    atomic_write_json(arguments.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("JiT upstream/probe-resume GPU parity failed")


if __name__ == "__main__":
    main()
