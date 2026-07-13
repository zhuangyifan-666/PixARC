import pytest

from taylorseer_style.metadata import PAIRING_FIELDS, canonical_hash, validate_full_taylorseer_roles, validate_paired_runs


def test_pairing_identity_and_roles():
    shared = {field: f"value-{field}" for field in PAIRING_FIELDS}
    shared.update({"checkpoint_size": 1, "steps": 50, "cfg_scale": 3.0, "resolution": 256, "world_size": 4, "batch_size": 32})
    reference = {**shared, "method": "instrumented_full", "interval": 4, "max_order": 3}
    candidate = {**shared, "method": "taylorseer", "interval": 4, "max_order": 3}
    validate_paired_runs(reference, candidate)
    validate_full_taylorseer_roles(reference, candidate)
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    candidate["seed"] = "different"
    candidate[PAIRING_FIELDS[0]] = "different"
    with pytest.raises(ValueError, match=PAIRING_FIELDS[0]):
        validate_paired_runs(reference, candidate)

