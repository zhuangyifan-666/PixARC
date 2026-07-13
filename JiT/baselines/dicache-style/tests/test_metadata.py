import pytest

from dicache_style.metadata import DICACHE_CONFIG_FIELDS, validate_dicache_config, validate_full_dicache_roles


def main_config(mode="dicache", threshold=.2):
    return {
        "mode": mode, "profile": "flux_image_released", "probe_depth": 1,
        "error_choice": "delta_y", "rel_l1_thresh": threshold, "ret_ratio": .2,
        "gamma_min": 1.0, "gamma_max": 1.5, "warmup_semantics": "flux_inclusive",
        "gate_mode": "batch_global", "threshold_compare": "strict_less_for_reuse",
        "probe_token_scope": "image_tokens", "dcta_order": 1,
        "residual_anchor_count": 2, "numeric_mode": "official_no_epsilon",
        "epsilon": 1e-8, "nonfinite_policy": "force_full_reset_and_log",
        "gamma_nonfinite_policy": "force_full", "force_last_full": True,
        "cache_dtype": "inherit", "trace_mode": "summary",
    }


def test_main_profile_is_strict_and_complete():
    value = validate_dicache_config(main_config())
    assert set(DICACHE_CONFIG_FIELDS).issubset(value)
    value["gamma_max"] = 2.0
    with pytest.raises(ValueError):
        validate_dicache_config(value)


def test_threshold_unresolved_fails_closed_for_dicache():
    with pytest.raises(ValueError):
        validate_dicache_config(main_config(threshold=None))


def test_role_validation():
    candidate = main_config()
    candidate.pop("mode")
    candidate["method"] = "dicache"
    validate_full_dicache_roles({"method": "instrumented_full"}, candidate)


def test_shadow_probe_depth_ablation_does_not_relax_main_depth():
    shadow = main_config(mode="probe_shadow_full")
    shadow.update(probe_depth=3, trace_mode="shadow")
    assert validate_dicache_config(shadow)["probe_depth"] == 3

    main = main_config()
    main["probe_depth"] = 3
    with pytest.raises(ValueError, match="main flux_image_released"):
        validate_dicache_config(main)
