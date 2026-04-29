from backend.run_experiments import _to_latex_table


def test_latex_table():
    tex = _to_latex_table([("x", {"accuracy": 1.0, "macro_f1": 1.0})], "cap")
    assert "\\begin{table}" in tex
