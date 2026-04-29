from backend.baselines import keyword_detector


def test_keyword_detector():
    assert keyword_detector("It may be true") == "yellow"
    assert keyword_detector("Earth orbits the sun") == "green"
