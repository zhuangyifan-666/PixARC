#!/usr/bin/env python3
"""Build the unified speed-quality table from four frozen baselines plus PRT."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


BASELINES = {
    "TaylorSeer": "taylorseer_1k_summary.csv",
    "SeaCache": "seacache_1k_summary.csv",
    "SpeCa": "speca_1k_summary.csv",
    "DiCache": "dicache_1k_summary.csv",
}
METRICS = (
    "fid", "delta_fid", "sfid", "delta_sfid", "inception_score",
    "delta_inception_score", "precision", "delta_precision", "recall",
    "delta_recall", "aggregate_mse", "psnr_from_aggregate_mse",
    "mean_per_image_psnr", "mean_ssim", "mean_lpips",
)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    try:
        return float(value) if value != "" else None
    except ValueError:
        return None


def _pareto(rows: list[dict[str, str]], metric: str) -> set[int]:
    """Non-dominated for greater speed and smaller measured error metric."""
    valid = [(i, _float(row, "speedup_vs_full"), _float(row, metric)) for i, row in enumerate(rows)]
    valid = [(i, speed, quality) for i, speed, quality in valid if speed is not None and quality is not None]
    result: set[int] = set()
    for index, speed, quality in valid:
        dominated = any(
            other_speed >= speed and other_quality <= quality
            and (other_speed > speed or other_quality < quality)
            for other_index, other_speed, other_quality in valid
            if other_index != index
        )
        if not dominated:
            result.add(index)
    return result


def _setting(source: str, row: dict[str, str]) -> str:
    keys = {
        "TaylorSeer": ("interval", "max_order"),
        "SeaCache": ("threshold",),
        "SpeCa": ("max_order", "base_threshold", "decay_rate", "min_taylor_steps", "max_taylor_steps"),
        "DiCache": ("rel_l1_thresh",),
        "Pixel-Remainder Taylor": ("tau", "max_taylor_span"),
    }[source]
    return ";".join(f"{key}={row[key]}" for key in keys if row.get(key, "") != "")


def _selected(source: str, row: dict[str, str], pareto: bool) -> bool:
    run = row.get("run", "")
    if run == "full":
        return source == "TaylorSeer"
    if source == "TaylorSeer":
        return run in {"i3_k1", "i3_k2"}
    if source == "SeaCache":
        return row.get("threshold") in {"0.02", "0.1", "0.4"}
    if source == "SpeCa":
        return run == "speca_ref" or pareto
    if source == "DiCache":
        return run == "t0p01" or pareto
    return True


def normalize(source: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    fid_frontier = _pareto(rows, "delta_fid")
    lpips_frontier = _pareto(rows, "mean_lpips")
    mse_frontier = _pareto(rows, "aggregate_mse")
    output = []
    for index, row in enumerate(rows):
        value: dict[str, Any] = {
            "model": row.get("model", ""),
            "source": source,
            "run": row.get("run", ""),
            "method": row.get("method", ""),
            "setting": _setting(source, row),
            "comparison_selected": int(_selected(source, row, index in fid_frontier)),
            "speed_fid_pareto": int(index in fid_frontier),
            "speed_lpips_pareto": int(index in lpips_frontier),
            "speed_mse_pareto": int(index in mse_frontier),
            "sample_count": row.get("sample_count", ""),
            "elapsed_seconds": row.get("elapsed_seconds", ""),
            "images_per_second": row.get("images_per_second", ""),
            "speedup_vs_full": row.get("speedup_vs_full", ""),
        }
        value.update({metric: row.get(metric, "") for metric in METRICS})
        output.append(value)
    return output


def _write_report(
    path: Path, rows: list[dict[str, Any]], trace_rows: list[dict[str, str]]
) -> None:
    selected = [row for row in rows if row["comparison_selected"]]
    lines = [
        "# Pixel-Remainder Taylor 1K speed-quality report",
        "",
        "This report is generated only from measured CSV values. Empty cells remain unmeasured.",
        "Existing baseline jobs were not rerun.",
        "",
        "| Model | Source | Run | Setting | Speedup | FID | ΔFID | sFID | IS | Precision | Recall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        cells = [
            row["model"], row["source"], row["run"], row["setting"],
            row["speedup_vs_full"], row["fid"], row["delta_fid"], row["sfid"],
            row["inception_score"], row["precision"], row["recall"],
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "The 1K distribution metrics are screening measurements; a small FID change alone is not evidence of a win. Paired LPIPS/MSE, throughput, and the action trace must agree before drawing a conclusion.",
    ])
    for model in ("JiT", "PixelGen"):
        prt = [row for row in rows if row["model"] == model and row["source"] == "Pixel-Remainder Taylor"]
        if not prt:
            continue
        fastest = max(prt, key=lambda row: float(row["speedup_vs_full"]))
        fixed_i3 = next(
            (row for row in rows if row["model"] == model and row["source"] == "TaylorSeer" and row["run"] == "i3_k2"),
            None,
        )
        fixed_i4 = next(
            (row for row in rows if row["model"] == model and row["source"] == "TaylorSeer" and row["run"] == "i4_k2"),
            None,
        )
        lines.extend([
            "",
            f"### {model}",
            "",
            f"Fastest measured PRT point: `{fastest['run']}` at {fastest['speedup_vs_full']}x Full, with ΔFID {fastest['delta_fid']}, mean LPIPS {fastest['mean_lpips']}, and aggregate MSE {fastest['aggregate_mse']}.",
        ])
        if fixed_i3 is not None:
            matches = [
                row for row in prt
                if _float(row, "mean_lpips") is not None
                and _float(row, "aggregate_mse") is not None
                and _float(row, "mean_lpips") <= _float(fixed_i3, "mean_lpips")
                and _float(row, "aggregate_mse") <= _float(fixed_i3, "aggregate_mse")
                and _float(row, "speedup_vs_full") > _float(fixed_i3, "speedup_vs_full")
            ]
            lines.append(
                "- Against TaylorSeer i3/k2, the measured PRT points that are simultaneously faster with no worse mean LPIPS or aggregate MSE are: "
                + (", ".join(f"`{row['run']}`" for row in matches) if matches else "none")
                + "."
            )
        if fixed_i4 is not None:
            safer = [
                row for row in prt
                if _float(row, "delta_fid") is not None
                and _float(row, "mean_lpips") is not None
                and _float(row, "delta_fid") <= _float(fixed_i4, "delta_fid")
                and _float(row, "mean_lpips") <= _float(fixed_i4, "mean_lpips")
            ]
            lines.append(
                "- Relative to TaylorSeer i4/k2's aggressive quality point, PRT points with no worse ΔFID and mean LPIPS are: "
                + (", ".join(f"`{row['run']}`" for row in safer) if safer else "none")
                + "."
            )
        model_traces = [row for row in trace_rows if row.get("model") == model]
        for trace in model_traces:
            span_keys = sorted(key for key in trace if key.startswith("planned_span_") and key.endswith("_ratio"))
            span_text = ", ".join(f"{key.removeprefix('planned_')}={trace[key]}" for key in span_keys)
            elapsed_row = next((row for row in prt if row["run"] == trace.get("run")), None)
            overhead = ""
            if elapsed_row and elapsed_row.get("elapsed_seconds") not in {"", None}:
                fraction = float(trace.get("controller_time_ms", 0.0)) / (float(elapsed_row["elapsed_seconds"]) * 1000.0)
                overhead = f"; controller/wall-clock={fraction:.6f}"
            lines.append(
                f"- `{trace.get('run')}`: Full/Taylor={trace.get('full_ratio')}/{trace.get('taylor_ratio')}, "
                f"order-1/order-2 Taylor={trace.get('order1_taylor_ratio')}/{trace.get('order2_taylor_ratio')}, "
                f"{span_text}, cap-hit={trace.get('span_cap_hit_ratio')}{overhead}."
            )
        if model_traces:
            both_orders = all(
                float(row.get("order1_taylor_nfe", 0)) > 0
                and float(row.get("order2_taylor_nfe", 0)) > 0
                for row in model_traces
            )
            varied = all(
                sum(
                    float(row.get(key, 0)) > 0
                    for key in row
                    if key.startswith("planned_span_")
                    and key.endswith("_ratio")
                    and key != "planned_span_0_ratio"
                ) > 1
                for row in model_traces
            )
            lines.append(
                f"- Trace checks across reported PRT runs: both Taylor orders used in every run={both_orders}; more than one planned span observed in every run={varied}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prt-summary", action="append", default=[])
    parser.add_argument("--prt-trace", action="append", default=[])
    parser.add_argument("--output", default="results/pixel_remainder_taylor_1k_comparison.csv")
    parser.add_argument("--report", default="results/PIXEL_REMAINDER_TAYLOR_1K_REPORT.md")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if not args.prt_summary and not args.baseline_only:
        raise ValueError("at least one --prt-summary is required; use --baseline-only only for validation")
    root = Path(args.results_root)
    combined: list[dict[str, Any]] = []
    for source, name in BASELINES.items():
        combined.extend(normalize(source, _read(root / name)))
    for name in args.prt_summary:
        combined.extend(normalize("Pixel-Remainder Taylor", _read(Path(name))))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    trace_rows = [row for name in args.prt_trace for row in _read(Path(name))]
    _write_report(report, combined, trace_rows)
    print(f"wrote {len(combined)} rows to {output} and {report}")


if __name__ == "__main__":
    main()
