"""Core TruthLens analyzer with confidence and domain support."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from statistics import mode
from typing import Any

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
RISK_LEVELS = ("green", "yellow", "red")

DOMAIN_RULES: dict[str, dict[str, Any]] = {
    "medical": {
        "keywords": [
            "patient", "diagnosis", "treatment", "drug", "disease",
            "symptom", "clinical", "medical", "doctor", "hospital"
        ],
        "warning": "Medical content — always consult a professional",
    },
    "legal": {
        "keywords": [
            "court", "law", "statute", "defendant", "plaintiff",
            "verdict", "legal", "attorney", "jurisdiction", "legislation"
        ],
        "warning": "Legal content — consult a qualified lawyer",
    },
    "scientific": {
        "keywords": [
            "study", "research", "experiment", "hypothesis", "findings",
            "data", "analysis", "published", "journal", "peer-reviewed"
        ],
        "warning": "Scientific content — verify with primary sources",
    },
    "news": {
        "keywords": [
            "reported", "according to", "said", "announced", "spokesperson",
            "sources", "claimed", "alleged", "breaking"
        ],
        "warning": "News content — verify with multiple sources",
    },
}


def _split_sentences(text: str) -> list[str]:
    """Split input text into non-empty sentences."""
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]


def detect_domain(text: str) -> tuple[str, str]:
    """Detect domain using keyword overlap."""
    lower = text.lower()
    scores = {
        domain: sum(1 for kw in rule["keywords"] if kw in lower)
        for domain, rule in DOMAIN_RULES.items()
    }
    if max(scores.values(), default=0) == 0:
        return "general", "General content — cross-check important claims"
    best = max(scores, key=scores.get)
    return best, DOMAIN_RULES[best]["warning"]


def _parse_model_json(raw_text: str) -> dict[str, Any]:
    """Parse JSON object from model text output."""
    cleaned = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model response")
    return json.loads(match.group(0))


def _heuristic_classify(sentence: str) -> dict[str, Any]:
    """Fallback heuristic classifier when model is unavailable."""
    lower = sentence.lower()
    hallucination_flags = [
        "always", "never", "everyone", "nobody",
        "100%", "proven fact", "scientifically proven",
        "definitely", "absolutely certain"
    ]
    uncertain_terms = [
        "may", "might", "could", "allegedly", "reportedly",
        "unclear", "supposedly", "believed to", "some say",
        "it is thought", "debated", "controversial"
    ]
    if any(t in lower for t in hallucination_flags):
        return {
            "risk_level": "red",
            "explanation": "Heuristic: high-risk absolute language detected."
        }
    if any(t in lower for t in uncertain_terms):
        return {
            "risk_level": "yellow",
            "explanation": "Heuristic: uncertainty language detected."
        }
    return {
        "risk_level": "green",
        "explanation": "Heuristic: no obvious risk markers found."
    }


def _classify_sentence(
    sentence: str,
    domain: str,
    temperature: float
) -> dict[str, Any]:
    """Classify one sentence with model or heuristic fallback."""
    prompt = (
        "You are a strict fact-checker. Analyze ONLY the factual accuracy "
        "of the given sentence. Do NOT assume anything is true just because "
        "it sounds plausible. Verify specific claims like names, dates, "
        "institutions, and inventions carefully.\n\n"
        f"Sentence: {sentence}\n\n"
        "Rules:\n"
        "- red: The sentence contains a clearly false or fabricated claim\n"
        "- yellow: The sentence contains uncertain or unverifiable claims\n"
        "- green: ALL claims in the sentence are accurate and verifiable\n\n"
        "Examples:\n"
        "- 'Einstein invented the telephone' → red (Bell invented it)\n"
        "- 'Einstein attended Harvard' → red (he attended ETH Zurich)\n"
        "- 'Einstein was born in Germany' → green (verifiable fact)\n"
        "- 'The drug may cause side effects' → yellow (uncertain)\n\n"
        "Return ONLY this JSON, nothing else:\n"
        "{\"risk_level\": \"red\", \"explanation\": \"reason\"}"
    )
    try:
        logger.info("Calling Ollama for: %s", sentence[:60])
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                    "keep_alive": "10m",
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        logger.info("Ollama raw response: %s", raw[:300])
        payload = _parse_model_json(raw)
        risk = payload.get("risk_level", "yellow").lower().strip()
        if risk not in RISK_LEVELS:
            logger.warning("Invalid risk '%s', defaulting to yellow", risk)
            risk = "yellow"
        return {
            "risk_level": risk,
            "explanation": payload.get(
                "explanation", "Model-generated assessment."
            ),
        }
    except requests.exceptions.Timeout:
        logger.warning("Ollama timed out for: %s", sentence[:50])
        return _heuristic_classify(sentence)
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama at %s", OLLAMA_URL)
        return _heuristic_classify(sentence)
    except Exception as exc:
        logger.warning("Model failed, using heuristic: %s", exc)
        return _heuristic_classify(sentence)


def _confidence_from_votes(
    votes: list[str]
) -> tuple[float, list[dict[str, float]], str]:
    """Convert risk votes into confidence score and alternatives."""
    counts = Counter(votes)
    top_label = mode(votes)
    agreement = counts[top_label]
    confidence_map = {3: 0.95, 2: 0.65, 1: 0.33}
    confidence = confidence_map.get(agreement, 0.33)
    distribution = [
        {
            "label": label,
            "probability": round(counts.get(label, 0) / len(votes), 2)
        }
        for label in RISK_LEVELS
        if label != top_label
    ]
    distribution.sort(key=lambda x: x["probability"], reverse=True)
    return confidence, distribution, top_label


def analyze_text(text: str, fast_mode: bool = True) -> dict[str, Any]:
    """Analyze text for hallucination risk and return structured output.

    Args:
        text: The AI-generated text to analyze
        fast_mode: If True, single-pass (faster, demo mode)
                   If False, 3-sample voting (slower, research mode)

    Returns:
        dict containing trust_score, domain, sentences with analysis
    """
    logger.info(
        "Starting analysis: %d chars, fast_mode=%s",
        len(text), fast_mode
    )

    sentences = _split_sentences(text)
    if not sentences:
        return {
            "trust_score": 0,
            "domain": "general",
            "domain_warning": "No text provided.",
            "overall_verdict": "No text to analyze.",
            "sentences": []
        }

    domain, domain_warning = detect_domain(text)
    logger.info("Detected domain: %s", domain)

    analyzed: list[dict[str, Any]] = []
    num_samples = 1 if fast_mode else 3

    for i, sentence in enumerate(sentences):
        logger.info(
            "Sentence %d/%d: %s",
            i + 1, len(sentences), sentence[:60]
        )

        sampled = []
        for j in range(num_samples):
            result = _classify_sentence(sentence, domain, temperature=0.1)
            sampled.append(result)
            if j < num_samples - 1:
                time.sleep(1)

        if fast_mode:
            chosen = sampled[0]
            analyzed.append({
                "text": sentence,
                "risk_level": chosen["risk_level"],
                "confidence": 0.7,
                "explanation": chosen["explanation"],
                "alternative_classifications": [],
            })
        else:
            votes = [s["risk_level"] for s in sampled]
            confidence, alternatives, label = _confidence_from_votes(votes)
            chosen = next(
                (s for s in sampled if s["risk_level"] == label),
                sampled[0]
            )
            analyzed.append({
                "text": sentence,
                "risk_level": label,
                "confidence": confidence,
                "explanation": chosen["explanation"],
                "alternative_classifications": alternatives,
            })

    score_map = {"green": 100, "yellow": 60, "red": 20}
    trust_score = int(
        sum(score_map[s["risk_level"]] for s in analyzed)
        / max(len(analyzed), 1)
    )

    if trust_score >= 75:
        overall = "Mostly reliable content."
    elif trust_score >= 45:
        overall = "Mixed reliability — verify key claims before using."
    else:
        overall = (
            "High hallucination risk detected — "
            "do not use without verification."
        )

    result = {
        "trust_score": trust_score,
        "domain": domain,
        "domain_warning": domain_warning,
        "overall_verdict": overall,
        "sentences": analyzed,
    }

    logger.info("Analysis complete. Trust score: %d", trust_score)
    return result