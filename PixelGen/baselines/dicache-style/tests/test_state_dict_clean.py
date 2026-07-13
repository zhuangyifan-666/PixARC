from model_helpers import tiny_model


def test_runtime_never_enters_state_dict():
    model = tiny_model()
    keys = tuple(model.state_dict())
    assert keys
    assert not any("dicache_runtime" in key or "anchors" in key for key in keys)
