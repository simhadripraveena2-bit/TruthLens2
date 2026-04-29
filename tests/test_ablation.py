from backend.ablation import run_ablation
from pathlib import Path


def test_run_ablation(tmp_path: Path):
    samples = [{"text": "The sky is blue.", "ground_truth": "green"}] * 2
    out = run_ablation(samples, tmp_path / "a.json")
    assert "full_truthlens" in out
