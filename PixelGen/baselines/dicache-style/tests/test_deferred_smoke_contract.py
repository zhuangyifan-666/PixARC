import importlib.util
from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_deferred_smoke_tests.sh"
)
GENERATE = SCRIPT.parent / "generate_shard.py"
SPEC = importlib.util.spec_from_file_location("pixelgen_generate_shard", GENERATE)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_lightning_cli_config_strips_all_port_only_top_level_keys():
    source = {
        "schema_version": "pixarc-dicache-config-v1",
        "checkpoint": "/checkpoint",
        "dicache": {"mode": "dicache"},
        "runtime": {"batch_size": 1},
        "selection_provenance": {"status": "selected"},
        "trainer": {"devices": 1},
        "model": {"denoiser": {}},
        "data": {"pred_batch_size": 1},
    }
    resolved = GENERATOR._lightning_cli_base_config(source)
    assert set(resolved) == {"trainer", "model", "data"}
    assert "selection_provenance" in source


def test_deferred_smoke_has_three_configs_and_hard_gates():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--upstream-config" in source
    assert "--full-config" in source
    assert "--candidate-config" in source
    assert "run_gpu_model_parity.py" in source
    assert source.index('python "$SCRIPT_ROOT/run_gpu_model_parity.py"') < source.index(
        'DICACHE_INVOCATION_ID="smoke-upstream-'
    )
    assert "compare_image_trees.py" in source
    assert "--require-exact" in source
    assert "expected_nfe = int(parity.get" in source
    assert "expected_forwards = int(parity.get" in source
    for field in (
        "direct_full_count",
        "resumed_full_count",
        "reuse_count",
        "probe_count",
        "dcta_count",
        "trajectory_call_count_valid",
        "trajectory_total_nfe",
        "trajectory_network_forward_count",
        "candidate_sample_is_finite_before_uint8",
        "candidate_cache_is_zero",
    ):
        assert field in source
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
