from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from taylorseer_style.finite_difference import taylor_forecast, update_factors


PIXARC_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_HELPER = (
    PIXARC_ROOT
    / "baselines"
    / "TaylorSeer"
    / "TaylorSeer-DiT"
    / "taylor_utils"
    / "__init__.py"
)


def _official_module():
    spec = importlib.util.spec_from_file_location("local_official_taylor_utils", OFFICIAL_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("max_order", range(5))
@pytest.mark.parametrize("coordinates", [(12, 10, 7, 3, -2), (-4, -1, 3, 8, 14)])
def test_official_factor_and_forecast_parity(dtype, max_order, coordinates):
    official = _official_module()
    generator = torch.Generator(device="cpu").manual_seed(1234 + max_order)
    features = [torch.randn((2, 3, 4), generator=generator, dtype=dtype) for _ in coordinates]
    cache = {"max_order": max_order, "first_enhance": 2, "cache": {-1: {0: {"attn": {}}}}}
    current = {
        "num_steps": 10_000,
        "activated_steps": [coordinates[0]],
        "layer": 0,
        "module": "attn",
    }
    local_factors: list[torch.Tensor] = []
    previous = None
    max_abs = 0.0
    max_rel = 0.0
    for index, (coordinate, feature) in enumerate(zip(coordinates, features, strict=True)):
        current["step"] = coordinate
        current["activated_steps"].append(coordinate)
        official.derivative_approximation(cache, current, feature)
        local_factors = update_factors(
            local_factors,
            feature,
            coordinate=coordinate,
            previous_coordinate=previous,
            max_order=max_order,
        )
        previous = coordinate
        official_factors = cache["cache"][-1][0]["attn"]
        assert len(local_factors) == len(official_factors) == min(index + 1, max_order + 1)
        for order, local in enumerate(local_factors):
            expected = official_factors[order]
            difference = (local - expected).abs()
            max_abs = max(max_abs, float(difference.max()))
            denominator = expected.abs().clamp_min(torch.finfo(dtype).eps)
            max_rel = max(max_rel, float((difference / denominator).max()))
            assert local.dtype == expected.dtype == dtype
            assert local.shape == expected.shape == feature.shape
            torch.testing.assert_close(local, expected, rtol=0, atol=0)
    forecast_coordinate = coordinates[-1] + 2
    current["step"] = forecast_coordinate
    expected_forecast = official.taylor_formula(cache, current)
    local_forecast = taylor_forecast(
        local_factors,
        coordinate=forecast_coordinate,
        anchor_coordinate=coordinates[-1],
    )
    difference = (local_forecast - expected_forecast).abs()
    max_abs = max(max_abs, float(difference.max()))
    denominator = expected_forecast.abs().clamp_min(torch.finfo(dtype).eps)
    max_rel = max(max_rel, float((difference / denominator).max()))
    torch.testing.assert_close(local_forecast, expected_forecast, rtol=0, atol=0)
    print(f"official_formula_parity order={max_order} dtype={dtype} max_abs={max_abs} max_rel={max_rel}")

