from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from speca_style.error_metrics import (
    BatchGlobalMetricAccumulator,
    error_tensor,
    error_value,
)


ROOT = Path(__file__).resolve().parents[4]
MODELS = ROOT / "baselines" / "Cache4Diffusion" / "dit" / "speca-dit" / "models.py"
FUNCTIONS = {
    "calculate_l1_error",
    "calculate_l2_error",
    "calculate_relative_l1_error",
    "calculate_relative_l2_error",
    "calculate_cosine_similarity_error",
}


def _official_functions():
    tree = ast.parse(MODELS.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODELS), "exec"), namespace)
    return namespace


OFFICIAL = _official_functions()


@pytest.mark.parametrize(
    "metric,official_name",
    [
        ("l1", "calculate_l1_error"),
        ("l2", "calculate_l2_error"),
        ("relative_l1", "calculate_relative_l1_error"),
        ("relative_l2", "calculate_relative_l2_error"),
        ("cosine_similarity", "calculate_cosine_similarity_error"),
    ],
)
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4, 5)])
def test_metric_matches_released_function(metric, official_name, shape):
    generator = torch.Generator().manual_seed(123)
    pred = torch.randn(shape, generator=generator)
    exact = torch.randn(shape, generator=generator).add_(0.25)
    expected = OFFICIAL[official_name](pred, exact)
    assert error_value(pred, exact, metric=metric) == pytest.approx(expected, rel=0, abs=1e-7)


def test_published_toy_values_and_elementwise_relative_semantics():
    pred = torch.tensor([[3.0, 1.0]])
    exact = torch.tensor([[1.0, 2.0]])
    assert error_value(pred, exact, metric="l1") == pytest.approx(1.5)
    assert error_value(pred, exact, metric="l2") == pytest.approx(1.5811388492584229)
    assert error_value(pred, exact, metric="relative_l1") == pytest.approx(1.25)
    assert error_value(pred, exact, metric="relative_l2") == pytest.approx(1.457737922668457)
    assert error_value(pred, exact, metric="cosine_similarity") == pytest.approx(0.2928932905)
    global_norm_ratio = torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact)
    assert float(global_norm_ratio) == pytest.approx(1.0)
    assert error_value(pred, exact, metric="relative_l2") != pytest.approx(float(global_norm_ratio))


@pytest.mark.parametrize("metric", ["l1", "l2", "relative_l1", "relative_l2", "cosine_similarity"])
def test_streaming_cfg_aggregation_matches_real_concatenation(metric):
    generator = torch.Generator().manual_seed(9)
    pred_a, exact_a = torch.randn(2, 3, 4, generator=generator), torch.randn(2, 3, 4, generator=generator)
    pred_b, exact_b = torch.randn(2, 3, 4, generator=generator), torch.randn(2, 3, 4, generator=generator)
    accumulator = BatchGlobalMetricAccumulator(metric=metric)
    accumulator.update(pred_a, exact_a)
    accumulator.update(pred_b, exact_b)
    expected = error_tensor(
        torch.cat([pred_a, pred_b]), torch.cat([exact_a, exact_b]), metric=metric
    )
    assert accumulator.finalize_tensor() == pytest.approx(float(expected), rel=1e-6, abs=1e-7)


def test_bfloat16_follows_released_input_dtype_protocol():
    pred = torch.tensor([[3.0, 1.0]], dtype=torch.bfloat16)
    exact = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    expected = OFFICIAL["calculate_relative_l2_error"](pred, exact)
    assert error_value(pred, exact, metric="relative_l2") == expected
