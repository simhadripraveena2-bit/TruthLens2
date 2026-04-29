from backend.human_eval import get_next_sentence


def test_next_sentence_shape():
    out = get_next_sentence()
    assert "done" in out
