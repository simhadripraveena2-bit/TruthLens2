from backend.prompt_sensitivity import PROMPTS


def test_prompts_present():
    assert len(PROMPTS) == 3
