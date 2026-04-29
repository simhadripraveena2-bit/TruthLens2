"""Prompt sensitivity experiments."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from evaluator import LABELS, _normalize_label

logger = logging.getLogger(__name__)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

PROMPTS = {
    "direct": (
        "Classify this sentence as exactly one of: green, yellow, or red.\n"
        "green = accurate and verifiable.\n"
        "yellow = uncertain or unverifiable.\n"
        "red = false or hallucination.\n"
        "Return ONLY this JSON: "
        "{\"risk_level\": \"green\", \"explanation\": \"reason\"}\n"
        "Sentence: "
    ),
    "cot": (
        "Analyze this sentence step by step:\n"
        "1. What specific facts does it claim?\n"
        "2. Are these facts verifiable?\n"
        "3. Classify as green (accurate), yellow (uncertain), or red (false).\n"
        "Return ONLY this JSON: "
        "{\"risk_level\": \"green\", \"explanation\": \"reason\"}\n"
        "Sentence: "
    ),
    "strict": (
        "You are a strict fact-checker.\n"
        "Rules:\n"
        "- If ANY claim is false or fabricated → red\n"
        "- If ANY claim is unverifiable → yellow\n"
        "- Only if ALL claims verified accurate → green\n"
        "Return ONLY this JSON: "
        "{\"risk_level\": \"green\", \"explanation\": \"reason\"}\n"
        "Sentence: "
    ),
}


def _run(prompt_prefix: str, text: str) -> str:
    """Run a single prompt strategy on one text.

    Args:
        prompt_prefix: The prompt template prefix
        text: The sentence to classify

    Returns:
        Predicted risk level: green, yellow, or red
    """
    full_prompt = f"{prompt_prefix}{text}"
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 100,    # ← short response needed
                    "keep_alive": "10m",   # ← keep model loaded
                }
            },
            timeout=120,                   # ← increased from 45
        )
        res.raise_for_status()
        raw = res.json().get("response", "yellow").lower()

        # Try JSON parsing first
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                import json as json_lib
                data = json_lib.loads(json_match.group())
                risk = data.get("risk_level", "yellow").lower()
                if risk in LABELS:
                    return risk
            except Exception:
                pass

        # Fallback: find first label word in response
        for label in LABELS:
            if label in raw:
                return label

        return "yellow"

    except requests.exceptions.Timeout:
        logger.warning(
            "Timeout for prompt '%s' on text: %s",
            prompt_prefix[:30], text[:50]
        )
        return "yellow"
    except Exception as exc:
        logger.warning("Prompt strategy failure: %s", exc)
        return "yellow"


def run_prompt_sensitivity(
    samples: list[dict[str, str]],
    out_path: Path
) -> dict[str, Any]:
    """Evaluate 3 prompt strategies on samples.

    Args:
        samples: List of {text, ground_truth} dicts
        out_path: Path to save results JSON

    Returns:
        Dict of strategy name → metrics
    """
    # Use smaller subset for speed — 20 samples is enough
    eval_samples = samples[:20]
    logger.info(
        "Running prompt sensitivity on %d samples",
        len(eval_samples)
    )

    gt = [_normalize_label(s["ground_truth"]) for s in eval_samples]
    results: dict[str, Any] = {}

    for name, prompt_prefix in PROMPTS.items():
        logger.info("Testing prompt strategy: %s", name)
        start = time.perf_counter()
        preds = []

        for i, s in enumerate(eval_samples):
            pred = _run(prompt_prefix, s["text"])
            preds.append(pred)
            if (i + 1) % 5 == 0:
                logger.info(
                    "Strategy '%s': %d/%d done",
                    name, i + 1, len(eval_samples)
                )

        elapsed = time.perf_counter() - start

        # Compute metrics
        _, _, f1, _ = precision_recall_fscore_support(
            gt, preds,
            labels=LABELS,
            zero_division=0
        )
        acc = float(accuracy_score(gt, preds))
        macro_f1 = float(sum(f1) / len(f1))

        # Log prediction distribution
        pred_dist = {l: preds.count(l) for l in LABELS}
        logger.info(
            "Strategy '%s' — acc=%.3f, f1=%.3f, preds=%s",
            name, acc, macro_f1, pred_dist
        )

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "avg_response_time_sec": float(
                elapsed / max(len(eval_samples), 1)
            ),
            "total_time_sec": float(elapsed),
            "prediction_distribution": pred_dist,
        }

    # Find best strategy
    best = max(results, key=lambda k: results[k]["macro_f1"])
    results["best_strategy"] = best
    logger.info("Best prompt strategy: %s", best)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )
    logger.info("Prompt sensitivity saved to %s", out_path)
    return results