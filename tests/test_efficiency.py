from backend.efficiency import measure_efficiency


def test_measure_efficiency():
    out = measure_efficiency(["a", "b"], ["dummy"])
    assert "dummy" in out
