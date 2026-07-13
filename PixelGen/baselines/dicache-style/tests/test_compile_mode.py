import pytest

from dicache_style.pixelgen_model import configure_pixelgen_compile_mode
from model_helpers import tiny_model


def test_matched_eager_configuration_is_idempotent_without_wrappers():
    model = tiny_model()
    assert configure_pixelgen_compile_mode(model, "matched_eager") == 0
    with pytest.raises(ValueError):
        configure_pixelgen_compile_mode(model, "mismatched")
