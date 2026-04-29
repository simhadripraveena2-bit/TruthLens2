from backend.analyzer import analyze_text, detect_domain


def test_detect_domain_medical():
    domain, _ = detect_domain("The patient received diagnosis and treatment.")
    assert domain == "medical"


def test_analyze_text_schema():
    result = analyze_text("The sky is blue. It might rain tomorrow.")
    assert "trust_score" in result
    assert "domain" in result
    assert isinstance(result["sentences"], list)
