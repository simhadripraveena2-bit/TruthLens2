"""Cross-model comparison for available Ollama models."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from evaluator import LABELS, _normalize_label

logger = logging.getLogger(__name__)
OLLAMA_BASE = "http://localhost:11434"

# Models to compare — gemma3:4b is primary, others optional
MODELS = {
    "gemma3:4b": "Gemma 3 4B (primary)",
    "mistral": "Mistral 7B",
    "llama3": "LLaMA 3 8B",
}


def _available_models() -> set[str]:
    """Get list of installed Ollama models.

    Returns:
        Set of available model names
    """
    try:
        res = requests.get(
            f"{OLLAMA_BASE}/api/tags",
            timeout=10
        )
        res.raise_for_status()
        models = {m["name"] for m in res.json().get("models", [])}
        logger.info("Available Ollama models: %s", models)
        return models
    except Exception as exc:
        logger.warning("Could not fetch model tags: %s", exc)
        return set()


def _classify_with_model(model_id: str, text: str) -> str:
    """Classify one sentence using specified model.

    Args:
        model_id: Ollama model identifier
        text: Sentence to classify

    Returns:
        Predicted risk level: green, yellow, or red
    """
    prompt = (
        "Classify this sentence as exactly one of: green, yellow, or red.\n"
        "green = accurate and verifiable.\n"
        "yellow = uncertain or unverifiable.\n"
        "red = false or hallucination.\n"
        "Return ONLY this JSON: "
        "{\"risk_level\": \"green\", \"explanation\": \"reason\"}\n"
        f"Sentence: {text}"
    )
    try:
        res = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 100,
                    "keep_alive": "10m",
                }
            },
            timeout=120,  # ← increased from 45
        )
        res.raise_for_status()
        raw = res.json().get("response", "yellow").lower()

        # Try JSON parsing first
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                risk = data.get("risk_level", "yellow").lower()
                if risk in LABELS:
                    return risk
            except Exception:
                pass

        # Fallback: find first label word
        for label in LABELS:
            if label in raw:
                return label

        return "yellow"

    except requests.exceptions.Timeout:
        logger.warning(
            "Timeout for model %s on: %s",
            model_id, text[:50]
        )
        return "yellow"
    except Exception as exc:
        logger.warning(
            "Model %s failed: %s",
            model_id, exc
        )
        return "yellow"


def run_cross_model(
    samples: list[dict[str, str]],
    out_path: Path
) -> dict[str, Any]:
    """Run benchmark across installed Ollama models.

    Args:
        samples: List of {text, ground_truth} dicts
        out_path: Path to save results JSON

    Returns:
        Dict of model name → metrics
    """
    available = _available_models()

    # Use smaller subset for speed
    eval_samples = samples[:20]
    logger.info(
        "Cross-model evaluation on %d samples",
        len(eval_samples)
    )

    gt = [_normalize_label(s["ground_truth"]) for s in eval_samples]
    results: dict[str, Any] = {}

    for model_id, display in MODELS.items():
        # Check if model is available
        if model_id not in available:
            logger.info(
                "Skipping %s — not installed", model_id
            )
            results[display] = {
                "status": "model not available",
                "install_cmd": f"ollama pull {model_id}"
            }
            continue

        logger.info("Evaluating model: %s", display)
        start = time.perf_counter()
        preds = []

        for i, s in enumerate(eval_samples):
            pred = _classify_with_model(model_id, s["text"])
            preds.append(pred)
            if (i + 1) % 5 == 0:
                logger.info(
                    "Model '%s': %d/%d done",
                    display, i + 1, len(eval_samples)
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
        gt_dist = {l: gt.count(l) for l in LABELS}
        logger.info(
            "Model '%s' — acc=%.3f, f1=%.3f",
            display, acc, macro_f1
        )
        logger.info(
            "Predictions: %s | GT: %s",
            pred_dist, gt_dist
        )

        results[display] = {
            "status": "evaluated",
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class": {
                label: {"f1": float(f1[i])}
                for i, label in enumerate(LABELS)
            },
            "prediction_distribution": pred_dist,
            "speed": {
                "total_time_sec": float(elapsed),
                "avg_time_per_sample": float(
                    elapsed / max(len(eval_samples), 1)
                ),
                "samples_per_sec": float(
                    len(eval_samples) / max(elapsed, 1e-6)
                ),
            },
            "calibration_proxy": float(
                sum(1 for p, g in zip(preds, gt) if p == g)
                / max(len(gt), 1)
            ),
        }

        logger.info("%s complete!", display)

    # Summary
    evaluated = {
        k: v for k, v in results.items()
        if isinstance(v, dict) and v.get("status") == "evaluated"
    }
    if evaluated:
        best = max(
            evaluated,
            key=lambda k: evaluated[k]["macro_f1"]
        )
        results["summary"] = {
            "best_model": best,
            "best_macro_f1": evaluated[best]["macro_f1"],
            "models_evaluated": list(evaluated.keys()),
            "models_skipped": [
                k for k, v in results.items()
                if isinstance(v, dict)
                and v.get("status") == "model not available"
            ]
        }
        logger.info("Best model: %s (F1=%.3f)",
                    best, evaluated[best]["macro_f1"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )
    logger.info("Cross-model results saved to %s", out_path)
    return results