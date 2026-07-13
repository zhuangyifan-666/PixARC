from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from taylorseer_style.scheduler import FULL, FixedIntervalScheduler


CAL_TYPE = (
    Path(__file__).resolve().parents[4]
    / "baselines"
    / "TaylorSeer"
    / "TaylorSeer-DiT"
    / "cache_functions"
    / "cal_type.py"
)


def _official_cal_type():
    spec = importlib.util.spec_from_file_location("local_official_cal_type", CAL_TYPE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cal_type


@pytest.mark.parametrize("interval", [1, 2, 3, 4, 5])
def test_50_step_schedule_matches_official(interval):
    cal_type = _official_cal_type()
    official_cache = {"first_enhance": 2, "interval": interval, "cache_counter": 0}
    current = {"num_steps": 50, "activated_steps": [49]}
    scheduler = FixedIntervalScheduler(interval=interval, max_order=4, first_enhance=2)
    scheduler.reset(50)
    official_actions = []
    official_counters = []
    for nfe_index in range(50):
        current["step"] = 49 - nfe_index
        cal_type(official_cache, current)
        official_actions.append("FULL" if current["type"] == "full" else "TAYLOR")
        official_counters.append(official_cache["cache_counter"])
        decision = scheduler.decide(
            nfe_index=nfe_index,
            macro_step_index=nfe_index,
            solver_stage="official_dit",
        )
        assert decision.action == official_actions[-1]
        assert decision.cache_counter_after == official_counters[-1]
        assert decision.q == current["step"]
    assert [d.action for d in scheduler.decisions] == official_actions
    assert [d.q for d in scheduler.decisions if d.action == FULL] == current["activated_steps"][1:]
    expected_counts = {1: (50, 0), 2: (26, 24), 3: (18, 32), 4: (14, 36), 5: (11, 39)}
    full_count = sum(action == "FULL" for action in official_actions)
    assert (full_count, 50 - full_count) == expected_counts[interval]
    if interval == 5:
        assert official_actions[-1] == "TAYLOR"
    print(f"official_schedule_parity interval={interval} full={full_count} taylor={50-full_count}")

