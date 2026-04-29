from backend.cross_model import MODELS


def test_models_defined():
    assert "gemma3:4b" in MODELS
