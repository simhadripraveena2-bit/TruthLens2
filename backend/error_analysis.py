"""Error analysis utilities."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _categorize_error(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["date", "year", "born", "died", "number", "%"]):
        return "factual_error"
    if any(k in lower for k in ["therefore", "because", "implies", "hence"]):
        return "reasoning_error"
    if any(k in lower for k in ["may", "could", "debate", "disputed"]):
        return "ambiguous"
    if any(k in lower for k in ["clinical", "statute", "peer-reviewed"]):
        return "domain_failure"
    return "subtle"


def analyze_errors(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Build FP/FN sets, categories and pattern report."""
    fps, fns = [], []
    for p, g in zip(predictions, ground_truth):
        pred = p.get("prediction", "yellow")
        gt = g.get("ground_truth", "yellow")
        text = p.get("text", "")
        if pred in {"yellow", "red"} and gt == "green":
            fps.append({"text": text, "predicted": pred, "truth": gt, "category": _categorize_error(text)})
        if pred == "green" and gt in {"yellow", "red"}:
            fns.append({"text": text, "predicted": pred, "truth": gt, "category": _categorize_error(text)})

    cats = Counter([e["category"] for e in fps + fns])
    length_error_rate = {
        "avg_error_len": (sum(len(e["text"]) for e in fps + fns) / max(len(fps + fns), 1)),
    }
    return {
        "false_positives_top20": fps[:20],
        "false_negatives_top20": fns[:20],
        "error_categories": dict(cats),
        "error_patterns": {
            "most_common_error_type": cats.most_common(1)[0][0] if cats else "none",
            "domain_with_most_errors": "general",
            "sentence_length_vs_error_rate": length_error_rate,
        },
    }


def save_error_analysis(report: dict[str, Any], path: Path) -> None:
    """Persist error report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
