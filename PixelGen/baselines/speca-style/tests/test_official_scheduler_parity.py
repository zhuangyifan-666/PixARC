from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from speca_style.scheduler import FULL, TAYLOR, ReleasedCodeSpeCaScheduler


ROOT = Path(__file__).resolve().parents[4]
OFFICIAL = ROOT / "baselines" / "Cache4Diffusion" / "dit" / "speca-dit"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAL = _load("official_speca_cal_type", OFFICIAL / "cache_functions" / "cal_type.py")
INIT = _load("official_speca_cache_init", OFFICIAL / "cache_functions" / "cache_init.py")


def _official_sequence(total: int, params: dict[str, object], injected_error: float):
    kwargs = {
        "max_order": 4,
        "test_FLOPs": False,
        "base_threshold": params["base_threshold"],
        "decay_rate": params["decay_rate"],
        "min_taylor_steps": params["min_taylor_steps"],
        "max_taylor_steps": params["max_taylor_steps"],
        "error_metric": "relative_l1",
    }
    cache, current = INIT.cache_init(kwargs, total)
    records = []
    for q in range(total - 1, -1, -1):
        current["step"] = q
        CAL.cal_type(cache, current)
        action = FULL if current["type"] == "full" else TAYLOR
        threshold = max(
            float(params["base_threshold"])
            * float(params["decay_rate"]) ** ((total - q) / total),
            0.01,
        )
        if action == TAYLOR and cache["check"]:
            current["last_layer_error"] = injected_error
        records.append(
            (
                action,
                bool(cache["check"]),
                int(cache["taylor_step_counter"]),
                int(cache["cache_counter"]),
                current.get("last_layer_error"),
                threshold,
                tuple(current["activated_steps"]),
                int(cache["full_count"]),
            )
        )
    return records


def _port_sequence(total: int, params: dict[str, object], injected_error: float):
    scheduler = ReleasedCodeSpeCaScheduler(**params)
    scheduler.reset(total)
    records = []
    for index in range(total):
        decision = scheduler.decide(
            nfe_index=index,
            macro_step_index=index // 2,
            solver_stage="predictor" if index % 2 == 0 else "corrector",
        )
        error = injected_error if decision.action == TAYLOR and decision.check else None
        scheduler.end_nfe(verification_error=error)
        records.append(
            (
                decision.action,
                decision.check,
                scheduler.taylor_step_counter,
                scheduler.cache_counter,
                scheduler.last_verification_error,
                scheduler.threshold_for_q(decision.q),
                tuple(scheduler.activated_steps),
                int(scheduler.full_count),
            )
        )
    return records


@pytest.mark.parametrize(
    "params,total,error,counts",
    [
        ({"base_threshold": 0.1, "decay_rate": 0.01, "min_taylor_steps": 2,
          "max_taylor_steps": 5, "first_enhance": 3}, 50, 0.0, (9, 41, 24)),
        ({"base_threshold": 0.1, "decay_rate": 0.01, "min_taylor_steps": 2,
          "max_taylor_steps": 5, "first_enhance": 3}, 50, 1e9, (13, 37, 12)),
        ({"base_threshold": 0.3, "decay_rate": 0.05, "min_taylor_steps": 3,
          "max_taylor_steps": 8, "first_enhance": 3}, 99, 0.0, (12, 87, 53)),
        ({"base_threshold": 0.3, "decay_rate": 0.05, "min_taylor_steps": 3,
          "max_taylor_steps": 8, "first_enhance": 3}, 99, 1e9, (21, 78, 19)),
    ],
)
def test_matches_local_released_cal_type(params, total, error, counts):
    official = _official_sequence(total, params, error)
    port = _port_sequence(total, params, error)
    assert len(port) == len(official)
    for expected, actual in zip(official, port):
        assert actual[:5] == expected[:5]
        assert actual[5] == pytest.approx(expected[5], rel=0, abs=1e-15)
        assert actual[6] == expected[6]
        assert actual[7] == expected[7]
    actions = [row[0] for row in port]
    verified = sum(row[0] == TAYLOR and row[1] for row in port)
    assert (actions.count(FULL), actions.count(TAYLOR), verified) == counts


def test_first_enhance_is_full_taylor_full_not_three_fulls():
    params = {"base_threshold": 0.3, "decay_rate": 0.05,
              "min_taylor_steps": 3, "max_taylor_steps": 8,
              "first_enhance": 3}
    assert [row[0] for row in _port_sequence(99, params, 0.0)[:3]] == [FULL, TAYLOR, FULL]


def test_explicit_force_last_full_ablation_overrides_mandatory_post_full_taylor():
    scheduler = ReleasedCodeSpeCaScheduler(
        base_threshold=0.3,
        decay_rate=0.05,
        min_taylor_steps=3,
        max_taylor_steps=8,
        first_enhance=3,
        force_last_full=True,
    )
    scheduler.reset(2)
    first = scheduler.decide(
        nfe_index=0, macro_step_index=0, solver_stage="predictor"
    )
    scheduler.end_nfe(verification_error=None)
    last = scheduler.decide(
        nfe_index=1, macro_step_index=0, solver_stage="final_euler"
    )
    assert first.action == FULL
    assert last.action == FULL
    assert last.full_reason == "force_last_full"
    scheduler.end_nfe(verification_error=None)
