"""Baseline models for TruthLens benchmark comparisons (tests H1)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import requests
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

from evaluator import LABELS, _normalize_label

logger = logging.getLogger(__name__)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
UNCERTAINTY_KEYWORDS = [
    "allegedly", "reportedly", "claimed", "rumored", "unverified",
    "supposedly", "believed to", "some say", "it is thought",
    "may", "might", "could", "unclear", "debated",
]


def _metrics(pred: list[str], gt: list[str]) -> dict[str, Any]:
    """Compute classification metrics."""
    # Find active labels in this evaluation
    active = [l for l in LABELS if l in set(gt + pred)]
    if not active:
        active = LABELS

    p, r, f1, _ = precision_recall_fscore_support(
        gt, pred, labels=active, zero_division=0
    )
    try:
        kappa = float(cohen_kappa_score(gt, pred, labels=active))
    except Exception:
        kappa = 0.0

    return {
        "accuracy": float(accuracy_score(gt, pred)),
        "cohen_kappa": kappa,
        "macro_f1": float(sum(f1) / len(f1)) if len(f1) > 0 else 0.0,
        "per_class": {
            l: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i])
            }
            for i, l in enumerate(active)
        },
    }


def random_classifier(text: str) -> str:
    """Assign random risk label — baseline 1."""
    return random.choice(LABELS)


def keyword_detector(text: str) -> str:
    """Keyword uncertainty baseline — baseline 2.

    Returns yellow if uncertainty keywords found,
    red if hallucination flags found, else green.
    """
    lower = text.lower()

    # Strong hallucination indicators → red
    hallucination_flags = [
        "invented the", "discovered the internet",
        "made of cheese", "breathe underwater",
        "never happened", "fictional",
    ]
    if any(k in lower for k in hallucination_flags):
        return "red"

    # Uncertainty indicators → yellow
    if any(k in lower for k in UNCERTAINTY_KEYWORDS):
        return "yellow"

    return "green"


def selfcheck_sampling(text: str) -> tuple[str, float]:
    """SelfCheckGPT-style consistency sampling — baseline 3.

    Runs model 3 times with high temperature and checks consistency.
    If all 3 agree → high confidence prediction
    If 2 agree → medium confidence
    If all disagree → yellow (uncertain)
    """
    prompt = (
        "Classify this sentence as exactly one of: green, yellow, or red.\n"
        "green = accurate and verifiable\n"
        "yellow = uncertain or unverifiable\n"
        "red = false or hallucination\n"
        f"Sentence: {text}\n"
        "Reply with only one word: green, yellow, or red."
    )
    labels = []
    for _ in range(3):
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 10,  # only need one word
                    }
                },
                timeout=60,
            )
            res.raise_for_status()
            raw = res.json().get("response", "yellow").strip().lower()
            # Extract first word and normalize
            first_word = raw.split()[0] if raw.split() else "yellow"
            label = _normalize_label(first_word)
            if label not in LABELS:
                label = "yellow"
        except Exception as exc:
            logger.warning("selfcheck sample failed: %s", exc)
            label = "yellow"
        labels.append(label)

    best = max(set(labels), key=labels.count)
    consistency = labels.count(best) / 3

    logger.debug(
        "selfcheck votes=%s best=%s consistency=%.2f",
        labels, best, consistency
    )

    # Only confident if 2+ agree
    return (best if consistency >= 0.66 else "yellow"), consistency


def llm_as_judge(text: str) -> str:
    """Strict LLM-as-Judge classification — baseline 4.

    Uses a direct one-word classification prompt.
    This is the strongest baseline — TruthLens should match or beat it.
    """
    prompt = (
        "You are a strict fact-checker.\n"
        "Classify this sentence as EXACTLY one word:\n"
        "- Accurate (if the sentence is factually correct)\n"
        "- Uncertain (if the sentence cannot be verified)\n"
        "- Hallucinated (if the sentence is false or fabricated)\n\n"
        f"Sentence: {text}\n\n"
        "Reply with ONLY one word: Accurate, Uncertain, or Hallucinated."
    )
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # deterministic
                    "num_predict": 10,   # only need one word
                }
            },
            timeout=60,
        )
        res.raise_for_status()
        raw = res.json().get("response", "Uncertain").strip()
        # Extract first word
        first_word = raw.split()[0].rstrip(".,!?") if raw.split() else "Uncertain"
        result = _normalize_label(first_word)
        if result not in LABELS:
            result = "yellow"
        logger.debug("llm_as_judge raw=%s normalized=%s", raw, result)
        return result
    except Exception as exc:
        logger.warning("llm_as_judge failed: %s", exc)
        return "yellow"


def run_baselines(
    samples: list[dict[str, str]],
    out_path: Path
) -> dict[str, Any]:
    """Run all 4 baselines on samples and save metrics.

    Args:
        samples: List of {text, ground_truth} dicts
        out_path: Path to save results JSON

    Returns:
        Dict of baseline name → metrics
    """
    logger.info("Running baselines on %d samples", len(samples))
    gt = [_normalize_label(s["ground_truth"]) for s in samples]

    # Baseline 1 — Random
    logger.info("Running random classifier...")
    preds_random = [random_classifier(s["text"]) for s in samples]

    # Baseline 2 — Keyword
    logger.info("Running keyword detector...")
    preds_keyword = [keyword_detector(s["text"]) for s in samples]

    # Baseline 3 — SelfCheck sampling
    logger.info("Running selfcheck sampling (3x per sample = slow)...")
    preds_selfcheck = []
    for i, s in enumerate(samples):
        pred, conf = selfcheck_sampling(s["text"])
        preds_selfcheck.append(pred)
        if (i + 1) % 10 == 0:
            logger.info(
                "Selfcheck progress: %d/%d", i + 1, len(samples)
            )

    # Baseline 4 — LLM as Judge
    logger.info("Running LLM-as-Judge...")
    preds_judge = []
    for i, s in enumerate(samples):
        pred = llm_as_judge(s["text"])
        preds_judge.append(pred)
        if (i + 1) % 10 == 0:
            logger.info(
                "LLM judge progress: %d/%d", i + 1, len(samples)
            )

    # Log prediction distributions for debugging
    logger.info("Random predictions: %s", {
        l: preds_random.count(l) for l in LABELS
    })
    logger.info("Keyword predictions: %s", {
        l: preds_keyword.count(l) for l in LABELS
    })
    logger.info("Selfcheck predictions: %s", {
        l: preds_selfcheck.count(l) for l in LABELS
    })
    logger.info("Judge predictions: %s", {
        l: preds_judge.count(l) for l in LABELS
    })
    logger.info("Ground truth distribution: %s", {
        l: gt.count(l) for l in LABELS
    })

    result = {
        "random": _metrics(preds_random, gt),
        "keyword": _metrics(preds_keyword, gt),
        "selfcheck_sampling": _metrics(preds_selfcheck, gt),
        "llm_as_judge": _metrics(preds_judge, gt),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Baselines saved to %s", out_path)
    return result