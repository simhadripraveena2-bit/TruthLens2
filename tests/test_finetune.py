from backend.finetune import run_finetune


def test_finetune_config(tmp_path):
    out = run_finetune(tmp_path)
    assert out["method"] == "LoRA"
