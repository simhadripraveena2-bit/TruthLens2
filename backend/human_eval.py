"""Human annotation API helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent / "data" / "human_annotations.json"
SEED_SENTENCES = [
    "The Eiffel Tower is located in Paris.",
    "Water boils at 100C at sea level.",
    "The moon is made of cheese.",
] * 7


def _load() -> dict[str, Any]:
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {"cursor": 0, "annotations": []}


def _save(data: dict[str, Any]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_next_sentence() -> dict[str, Any]:
    """Return next sentence for annotation."""
    data = _load()
    idx = data["cursor"]
    if idx >= 20:
        return {"done": True}
    return {"done": False, "index": idx + 1, "total": 20, "sentence": SEED_SENTENCES[idx]}


def submit_annotation(label: str) -> dict[str, Any]:
    """Store one human annotation and increment cursor."""
    data = _load()
    idx = data["cursor"]
    if idx >= 20:
        return {"status": "complete"}
    data["annotations"].append({"index": idx, "sentence": SEED_SENTENCES[idx], "human_label": label, "truthlens_label": "green"})
    data["cursor"] = idx + 1
    _save(data)
    return {"status": "ok", "progress": f"{data['cursor']}/20"}


def get_results() -> dict[str, Any]:
    """Compute agreement stats (Cohen's Kappa)."""
    data = _load()
    human = [a["human_label"] for a in data["annotations"]]
    truth = [a["truthlens_label"] for a in data["annotations"]]
    kappa = float(cohen_kappa_score(human, truth)) if len(human) > 1 else 0.0
    return {"count": len(human), "cohen_kappa": kappa}
