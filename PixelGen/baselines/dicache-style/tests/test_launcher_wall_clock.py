from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launcher_wall_clock", ROOT / "scripts" / "launcher_wall_clock.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    invocation_id: str,
    *,
    start_ns: int,
    end_ns: int,
    baseline: int,
    cumulative: int,
    completed: bool,
) -> dict[str, object]:
    return {
        "invocation_id": invocation_id,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "elapsed_seconds": (end_ns - start_ns) / 1e9,
        "baseline_sample_count": baseline,
        "invocation_sample_count": cumulative - baseline,
        "cumulative_sample_count": cumulative,
        "manifest_sample_count": 10,
        "world_size": 4,
        "completed": completed,
    }


def test_cumulative_wall_clock_includes_all_resume_invocations():
    rows = [
        _row("one", start_ns=1_000_000_000, end_ns=2_000_000_000,
             baseline=0, cumulative=4, completed=False),
        _row("two", start_ns=4_000_000_000, end_ns=6_000_000_000,
             baseline=4, cumulative=10, completed=True),
    ]
    result = MODULE._cumulative_payload(rows, expected=10, world_size=4)
    assert result["completed"] is True
    assert result["invocation_chain_valid"] is True
    assert result["active_elapsed_seconds"] == 3.0
    assert result["end_to_end_elapsed_seconds"] == 5.0
    assert result["images_per_second"] == 10 / 3
    assert result["end_to_end_images_per_second"] == 2.0


def test_cumulative_wall_clock_rejects_noncontiguous_resume_prefix():
    rows = [
        _row("one", start_ns=1, end_ns=2, baseline=0, cumulative=4,
             completed=False),
        _row("two", start_ns=3, end_ns=4, baseline=5, cumulative=10,
             completed=True),
    ]
    result = MODULE._cumulative_payload(rows, expected=10, world_size=4)
    assert result["invocation_chain_valid"] is False
    assert result["completed"] is False
    assert result["images_per_second"] is None
