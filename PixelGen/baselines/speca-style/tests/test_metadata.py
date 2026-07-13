import pytest

from speca_style.metadata import (
    PAIRING_FIELDS,
    SPECA_CONFIG_FIELDS,
    canonical_hash,
    validate_full_speca_roles,
    validate_paired_runs,
)


def test_pairing_identity_and_roles():
    shared = {field: f"value-{field}" for field in PAIRING_FIELDS}
    shared.update(
        {
            "checkpoint_size": 1,
            "steps": 50,
            "cfg_scale": 3.0,
            "resolution": 256,
            "world_size": 4,
            "batch_size": 1,
        }
    )
    reference = {**shared, "method": "instrumented_full"}
    candidate = {
        **shared,
        "method": "speca",
        "scheduler_mode": "released_code_faithful",
        "interval": None,
        "max_order": 4,
        "base_threshold": 0.3,
        "decay_rate": 0.05,
        "min_taylor_steps": 3,
        "max_taylor_steps": 8,
        "first_enhance": 3,
        "threshold_floor": 0.01,
        "error_metric": "relative_l1",
        "error_eps": 1.0e-10,
        "verify_layer": -1,
        "verification_token_scope": "all_tokens",
        "gate_mode": "batch_global",
        "coordinate_mode": "official_nfe_index",
        "force_last_full": False,
        "cache_dtype": "inherit",
        "trace_mode": "summary",
    }
    assert set(SPECA_CONFIG_FIELDS).issubset(candidate)
    validate_paired_runs(reference, candidate)
    validate_full_speca_roles(reference, candidate)
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    candidate["seed"] = "different"
    candidate[PAIRING_FIELDS[0]] = "different"
    with pytest.raises(ValueError, match=PAIRING_FIELDS[0]):
        validate_paired_runs(reference, candidate)
