from backend.error_analysis import analyze_errors


def test_analyze_errors():
    out = analyze_errors(
        [{"text": "The moon is cheese", "prediction": "green"}],
        [{"ground_truth": "red"}],
    )
    assert "false_negatives_top20" in out
