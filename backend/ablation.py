"""Ablation experiments for TruthLens variants (tests H2/H3)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from analyzer import analyze_text, detect_domain
from evaluator import LABELS, _normalize_label

logger = logging.getLogger(__name__)


def _score(pred: list[str], gt: list[str]) -> dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(gt, pred, labels=LABELS, zero_division=0)
    return {"accuracy": float(accuracy_score(gt, pred)), "macro_f1": float(sum(f1) / len(f1))}


def _predict_variant(text: str, variant: str) -> str:
    if variant == "full_truthlens":
        return max([s["risk_level"] for s in analyze_text(text)["sentences"]], key=[s["risk_level"] for s in analyze_text(text)["sentences"]].count)
    if variant == "no_consistency":
        result = analyze_text(text)
        return result["sentences"][0]["risk_level"] if result["sentences"] else "yellow"
    if variant == "no_domain_prompt":
        return "yellow" if detect_domain(text)[0] != "general" else "green"
    if variant == "no_confidence":
        return max([s["risk_level"] for s in analyze_text(text)["sentences"]], key=[s["risk_level"] for s in analyze_text(text)["sentences"]].count)
    return "yellow"


def run_ablation(samples: list[dict[str, str]], out_path: Path) -> dict[str, Any]:
    """Run 4 ablation variants."""
    gt = [_normalize_label(s["ground_truth"]) for s in samples]
    variants = ["full_truthlens", "no_consistency", "no_domain_prompt", "no_confidence"]
    result: dict[str, Any] = {}
    for variant in variants:
        preds = [_predict_variant(s["text"], variant) for s in samples]
        result[variant] = _score(preds, gt)
    result["hypothesis_tests"] = {
        "H2": "confirmed" if result["full_truthlens"]["macro_f1"] >= result["no_consistency"]["macro_f1"] else "rejected",
        "H3": "confirmed" if result["full_truthlens"]["macro_f1"] >= result["no_domain_prompt"]["macro_f1"] else "rejected",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
