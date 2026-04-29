from backend.reproducibility import set_global_seed


def test_seed_set():
    assert set_global_seed(42) == 42
