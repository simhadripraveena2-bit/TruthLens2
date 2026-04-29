"""Reproducibility utilities for TruthLens experiments."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)

SEED = 42


def set_global_seed(seed: int = SEED) -> int:
    """Set random seeds for deterministic experiment behavior."""
    logger.info("Setting global seed to %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _safe_version(pkg: str) -> str:
    """Return installed package version or fallback value."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_reproducibility_config(
    model_name: str,
    dataset_versions: dict[str, str],
    prompts_used: dict[str, str],
    dataset_splits: dict[str, Any],
) -> dict[str, Any]:
    """Build reproducibility configuration payload."""
    logger.info("Collecting reproducibility configuration")
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "seed": SEED,
        "model": model_name,
        "model_version": "local-ollama",
        "dataset_versions": dataset_versions,
        "prompts_used": prompts_used,
        "dataset_splits": dataset_splits,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "ram_gb": ram_gb,
            "gpu": gpu_name,
        },
        "library_versions": {
            "transformers": _safe_version("transformers"),
            "torch": _safe_version("torch"),
            "datasets": _safe_version("datasets"),
        },
    }


def save_reproducibility_log(config: dict[str, Any], path: Path) -> None:
    """Persist reproducibility metadata to disk."""
    logger.info("Saving reproducibility log to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def freeze_requirements(path: Path) -> None:
    """Write exact dependency versions using import metadata when available."""
    logger.info("Writing exact requirements to %s", path)
    candidates = [
        "fastapi",
        "uvicorn",
        "requests",
        "python-dotenv",
        "pydantic",
        "datasets",
        "evaluate",
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas",
        "psutil",
        "torch",
        "transformers",
        "unsloth",
        "trl",
        "peft",
        "matplotlib",
        "seaborn",
    ]
    lines = [f"{name}=={_safe_version(name)}" for name in candidates]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
