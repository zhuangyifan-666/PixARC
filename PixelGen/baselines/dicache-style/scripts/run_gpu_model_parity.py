#!/usr/bin/env python3
"""Deferred real-model parity: upstream Full vs exact probe/resume Full."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dicache_style.manifest import load_manifest
from dicache_style.metadata import atomic_write_json
from dicache_style.source_identity import (
    release_source_bindings,
    require_source_identity_current,
)


BASELINE_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = BASELINE_ROOT.parents[2] / "third-party" / "PixelGen"


def tensor_sequence_metrics(
    candidate: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]
) -> dict[str, float]:
    """Aggregate stable error metrics over an ordered nonempty tensor sequence."""

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
    """Evaluate the attachment's nine resume invariants plus lifecycle gates."""

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
    resumed_full_count = int(resumed_summary.get("resumed_full_count", -1))
    direct_full_count = int(resumed_summary.get("direct_full_count", -1))
    anchor_count = int(resumed_diag["anchor_check_count"])
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
        "7_probe_capture_keeps_gradients_disabled": (
            bool(upstream_diag["gradient_disabled_during_capture"])
            and bool(resumed_diag["gradient_disabled_during_capture"])
            and bool(upstream_diag["gradient_state_restored"])
            and bool(resumed_diag["gradient_state_restored"])
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


def _record_from_arguments(arguments: argparse.Namespace) -> tuple[int, int, int]:
    explicit = (arguments.sample_id, arguments.seed, arguments.class_id)
    if arguments.manifest is not None:
        if any(value is not None for value in explicit):
            raise ValueError("use either --manifest or explicit sample/seed/class IDs")
        records = load_manifest(Path(arguments.manifest).resolve(strict=True))
        if not records:
            raise ValueError("parity manifest is empty")
        record = records[0]
        return record.sample_id, record.seed, record.class_id
    if any(value is None for value in explicit):
        raise ValueError(
            "without --manifest, --sample-id, --seed, and --class-id are required"
        )
    return int(arguments.sample_id), int(arguments.seed), int(arguments.class_id)


def _benchmark_runner_from_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    """Resolve the immutable parity sample and bind the manifest before model work."""

    sample_id, seed, class_id = _record_from_arguments(arguments)
    if arguments.manifest is None:
        raise ValueError("PixelGen model parity requires --manifest")
    manifest = Path(arguments.manifest).resolve(strict=True)
    return {
        "model_config": arguments.model_config,
        "manifest": str(manifest),
        "batch_size": 1,
        "sample_ids": [sample_id],
        "seeds": [seed],
        "class_ids": [class_id],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--class-id", type=int)
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
    runner = _benchmark_runner_from_arguments(arguments)
    sample_id = int(runner["sample_ids"][0])
    seed = int(runner["seeds"][0])
    class_id = int(runner["class_ids"][0])

    # Delay the PixelGen factory import so CPU tests can exercise metric and
    # invariant helpers without importing the upstream GPU dependency tree.
    from dicache_style.pixelgen_benchmark import build_benchmark_spec
    from dicache_style.scheduler import (
        expected_network_forward_count,
        expected_nfe_count,
    )

    spec = build_benchmark_spec(runner)
    upstream = spec.full.upstream_body_and_image()  # type: ignore[attr-defined]
    resumed = spec.full.probe_only_body_and_image()  # type: ignore[attr-defined]
    candidate_sanity = spec.dicache.raw_finiteness()  # type: ignore[attr-defined]

    upstream_summary = upstream["summary"]
    resumed_summary = resumed["summary"]
    sampler_name = str(spec.metadata.get("sampler"))
    sampler_steps = int(spec.metadata.get("steps", -1))
    exact_heun = sampler_name == "exact_heun"
    expected_nfe = expected_nfe_count(
        "heun", sampler_steps, exact_heun=exact_heun
    )
    expected_forwards = expected_network_forward_count(
        model_family="pixelgen",
        sampler="heun",
        num_steps=sampler_steps,
        exact_heun=exact_heun,
    )
    if expected_nfe <= 0 or expected_forwards <= 0:
        raise RuntimeError("sampler configuration did not derive positive parity counts")
    if int(upstream_summary.get("total_nfe", -1)) != expected_nfe:
        raise RuntimeError("upstream summary differs from configured sampler NFE")
    if (
        int(upstream_summary.get("expected_network_forward_count", -1))
        != expected_forwards
    ):
        raise RuntimeError("upstream summary differs from configured forward count")
    if int(resumed_summary.get("expected_network_forward_count", -1)) != expected_forwards:
        raise RuntimeError("upstream/probe-only expected forward counts differ")

    body_metrics = tensor_sequence_metrics(
        resumed["body_outputs"], upstream["body_outputs"]
    )
    probe_metrics = tensor_sequence_metrics(
        resumed["probe_features"], upstream["probe_features"]
    )
    sample_metrics = tensor_sequence_metrics(
        [resumed["sample"]], [upstream["sample"]]
    )
    decoded_metrics = tensor_sequence_metrics(
        [resumed["decoded_image"]], [upstream["decoded_image"]]
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
    decoded_allclose = tensor_sequence_allclose(
        [resumed["decoded_image"]],
        [upstream["decoded_image"]],
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
            "configured_sampler_is_exact_heun": (
                sampler_name == "exact_heun" and sampler_steps > 0
            ),
            "upstream_float_tensors_are_finite": bool(
                torch.isfinite(upstream["sample"]).all().item()
                and torch.isfinite(upstream["decoded_image"]).all().item()
            ),
            "probe_only_float_tensors_are_finite": bool(
                torch.isfinite(resumed["sample"]).all().item()
                and torch.isfinite(resumed["decoded_image"]).all().item()
            ),
            "candidate_sample_is_finite_before_uint8": bool(
                candidate_sanity["sample_finite"]
            ),
            "candidate_decoded_image_is_finite_before_uint8": bool(
                candidate_sanity["decoded_image_finite"]
            ),
            "candidate_nfe_count": int(candidate_summary.get("total_nfe", -1))
            == expected_nfe,
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
        and decoded_allclose
        and all(nine.values())
        and all(operational.values())
    )
    report = {
        "schema_version": "pixarc-pixelgen-dicache-resume-parity-v1",
        "passed": passed,
        "source": source,
        "comparison": "upstream_full_vs_probe_only_ablation",
        "body_final_layer_inputs": body_metrics,
        "probe_depth_features": probe_metrics,
        "final_sample": sample_metrics,
        "decoded_image": decoded_metrics,
        "body_allclose": body_allclose,
        "probe_feature_allclose": probe_allclose,
        "final_sample_allclose": sample_allclose,
        "decoded_image_allclose": decoded_allclose,
        "nine_resume_invariants": nine,
        "operational_invariants": operational,
        "expected_nfe": expected_nfe,
        "expected_combined_forwards": expected_forwards,
        "upstream_diagnostics": upstream["diagnostics"],
        "probe_only_diagnostics": resumed["diagnostics"],
        "upstream_summary": upstream_summary,
        "probe_only_summary": resumed_summary,
        "candidate_raw_finiteness": candidate_sanity,
        "atol": arguments.atol,
        "rtol": arguments.rtol,
        "sample_id": sample_id,
        "seed": seed,
        "class_id": class_id,
    }
    require_source_identity_current(
        source,
        BASELINE_ROOT,
        UPSTREAM_ROOT,
        context="PixelGen GPU parity evidence generation",
    )
    atomic_write_json(arguments.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("PixelGen upstream/probe-resume GPU parity failed")


if __name__ == "__main__":
    main()
