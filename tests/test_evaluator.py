from backend.evaluator import compute_calibration_metrics, compute_metrics


def test_compute_metrics_basic():
    m = compute_metrics(["green", "red"], ["green", "red"])
    assert m["overall"]["accuracy"] == 1.0


def test_calibration_metrics():
    c = compute_calibration_metrics([0.2, 0.8], [False, True])
    assert "ece" in c and "brier_score" in c
