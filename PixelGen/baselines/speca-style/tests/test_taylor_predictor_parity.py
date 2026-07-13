from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from speca_style.finite_difference import taylor_forecast, update_factors
from speca_style.predictor import TaylorDraftPredictor
from speca_style.scheduler import FixedDraftScheduler


ROOT = Path(__file__).resolve().parents[4]
SIBLING = ROOT / "PixelGen" / "baselines" / "taylorseer-style" / "taylorseer_style" / "finite_difference.py"
SIBLING_SCHEDULER = ROOT / "PixelGen" / "baselines" / "taylorseer-style" / "taylorseer_style" / "scheduler.py"
OFFICIAL = ROOT / "baselines" / "Cache4Diffusion" / "dit" / "speca-dit" / "taylor_utils" / "__init__.py"


def _load_sibling():
    spec = importlib.util.spec_from_file_location("local_taylorseer_finite_difference", SIBLING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SIBLING_IMPL = _load_sibling()


def _load_official():
    spec = importlib.util.spec_from_file_location("official_speca_taylor_utils", OFFICIAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL_IMPL = _load_official()


def _load_sibling_scheduler():
    spec = importlib.util.spec_from_file_location(
        "local_taylorseer_scheduler", SIBLING_SCHEDULER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SIBLING_SCHEDULER_IMPL = _load_sibling_scheduler()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coordinates", [(9, 7, 4, 0), (0, 2, 5, 9)])
@pytest.mark.parametrize("max_order", range(5))
def test_factors_and_forecasts_match_local_taylorseer(dtype, coordinates, max_order):
    ours: list[torch.Tensor] = []
    theirs: list[torch.Tensor] = []
    previous = None
    generator = torch.Generator().manual_seed(77)
    for coordinate in coordinates:
        feature = torch.randn(2, 3, dtype=dtype, generator=generator)
        ours = update_factors(
            ours, feature, coordinate=coordinate,
            previous_coordinate=previous, max_order=max_order,
        )
        theirs = SIBLING_IMPL.update_factors(
            theirs, feature, coordinate=coordinate,
            previous_coordinate=previous, max_order=max_order,
        )
        assert len(ours) == len(theirs)
        for left, right in zip(ours, theirs):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        probe = coordinate - 1 if coordinates[0] > coordinates[-1] else coordinate + 1
        torch.testing.assert_close(
            taylor_forecast(ours, coordinate=probe, anchor_coordinate=coordinate),
            SIBLING_IMPL.taylor_forecast(
                theirs, coordinate=probe, anchor_coordinate=coordinate
            ),
            rtol=0,
            atol=0,
        )
        previous = coordinate


def test_predictor_never_writes_forecast_back_to_history():
    predictor = TaylorDraftPredictor(max_order=3)
    predictor.update_exact(torch.tensor([1.0, 2.0]), 8)
    predictor.update_exact(torch.tensor([2.0, 4.0]), 6)
    pointers = tuple(value.data_ptr() for value in predictor.factors)
    factors = [value.clone() for value in predictor.factors]
    _ = predictor.forecast(5)
    assert pointers == tuple(value.data_ptr() for value in predictor.factors)
    for before, after in zip(factors, predictor.factors):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_negative_gap_is_signed():
    first = update_factors([], torch.tensor([1.0]), coordinate=9,
                           previous_coordinate=None, max_order=2)
    second = update_factors(first, torch.tensor([3.0]), coordinate=7,
                            previous_coordinate=9, max_order=2)
    assert second[1].item() == pytest.approx(-1.0)


@pytest.mark.parametrize("max_order", range(5))
def test_factors_and_forecast_match_official_speca_taylor_utils(max_order):
    coordinates = (9, 7, 4, 0)
    generator = torch.Generator().manual_seed(314)
    features = [torch.randn(2, 3, generator=generator) for _ in coordinates]
    cache = {
        "max_order": max_order,
        "first_enhance": 3,
        "cache": {-1: {0: {"attn": {}}}},
    }
    current = {
        "activated_steps": [coordinates[0]],
        "layer": 0,
        "module": "attn",
        "num_steps": 10,
    }
    ours: list[torch.Tensor] = []
    previous = None
    for coordinate, feature in zip(coordinates, features):
        current["step"] = coordinate
        current["activated_steps"].append(coordinate)
        OFFICIAL_IMPL.taylor_cache_init(cache, current)
        OFFICIAL_IMPL.derivative_approximation(cache, current, feature)
        ours = update_factors(
            ours,
            feature,
            coordinate=coordinate,
            previous_coordinate=previous,
            max_order=max_order,
        )
        official = cache["cache"][-1][0]["attn"]
        assert len(ours) == len(official)
        for order, value in enumerate(ours):
            torch.testing.assert_close(value, official[order], rtol=0, atol=0)
        probe = coordinate - 1
        torch.testing.assert_close(
            taylor_forecast(
                ours, coordinate=probe, anchor_coordinate=coordinate
            ),
            OFFICIAL_IMPL.taylor_formula(official, probe - coordinate),
            rtol=0,
            atol=0,
        )
        previous = coordinate


@pytest.mark.parametrize("total_nfe", [50, 99])
@pytest.mark.parametrize("interval", [1, 2, 3, 6])
def test_fixed_draft_schedule_matches_local_taylorseer(total_nfe, interval):
    ours = FixedDraftScheduler(interval=interval, first_enhance=2)
    sibling = SIBLING_SCHEDULER_IMPL.FixedIntervalScheduler(
        interval=interval, max_order=4, first_enhance=2
    )
    ours.reset(total_nfe)
    sibling.reset(total_nfe)
    for index in range(total_nfe):
        arguments = {
            "nfe_index": index,
            "macro_step_index": index // 2,
            "solver_stage": "predictor" if index % 2 == 0 else "corrector",
        }
        left = ours.decide(**arguments)
        right = sibling.decide(**arguments)
        assert (left.action, left.q, left.cache_counter_before, left.cache_counter_after) == (
            right.action,
            right.q,
            right.cache_counter_before,
            right.cache_counter_after,
        )
        ours.end_nfe(verification_error=None)
