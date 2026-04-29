"""Fine-tuning pipeline for Gemma 3 using Unsloth + LoRA.

NOTE: This file is a configuration scaffold for local reference.
Actual fine-tuning must be run on Google Colab (T4 GPU) or
any machine with a CUDA-compatible GPU.

See: https://colab.research.google.com for free GPU access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


def prepare_halueval_instruction_data(limit: int = 500) -> list[dict[str, str]]:
    """Convert HaluEval QA samples into instruction tuning format.

    Args:
        limit: Number of samples to prepare

    Returns:
        List of instruction-formatted examples
    """
    logger.info("Loading HaluEval dataset...")
    ds = load_dataset("pminervini/HaluEval", "qa", split=f"data[:{limit}]")
    rows = []
    for item in ds:
        question = item.get("question", "")
        answer = item.get("answer", "")
        hallucination = item.get("hallucination", "no")

        if str(hallucination).lower() in ["yes", "1", "true"]:
            risk_level = "red"
            explanation = "This answer contains hallucinated or false information."
        else:
            risk_level = "green"
            explanation = "This answer appears accurate and verifiable."

        # Format as Gemma chat template
        text = f"""<start_of_turn>user
You are a strict fact-checker. Analyze this text for hallucinations.

Question: {question}
Answer: {answer}

Return ONLY this JSON:
{{"risk_level": "green/yellow/red", "explanation": "reason"}}
<end_of_turn>
<start_of_turn>model
{{"risk_level": "{risk_level}", "explanation": "{explanation}"}}
<end_of_turn>"""

        rows.append({"text": text})

    logger.info("Prepared %d training examples", len(rows))
    return rows


def run_finetune(output_dir: Path) -> dict[str, Any]:
    """LoRA fine-tuning configuration.

    NOTE: Actual training runs on Google Colab.
    This function saves the config for reproducibility.

    Args:
        output_dir: Directory to save config and model

    Returns:
        Configuration dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "unsloth/gemma-3-4b-it",
        "library": "unsloth",
        "method": "LoRA",
        "lora_config": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        },
        "training_config": {
            "epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "max_seq_length": 512,
            "warmup_steps": 10,
            "seed": 42,
            "optimizer": "adamw_8bit"
        },
        "dataset": {
            "name": "pminervini/HaluEval",
            "config": "qa",
            "split": "data[:500]",
            "samples": 500
        },
        "hardware_note": "Requires GPU. Run on Google Colab T4 GPU (free tier).",
        "colab_notebook": "See README.md for Colab setup instructions",
        "status": "configured — run on Colab for actual training",
    }

    config_path = output_dir / "finetune_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("Fine-tuning config saved to %s", config_path)

    # Save training data preview
    try:
        samples = prepare_halueval_instruction_data(limit=5)
        preview_path = output_dir / "training_data_preview.json"
        preview_path.write_text(json.dumps(samples[:3], indent=2), encoding="utf-8")
        logger.info("Training data preview saved to %s", preview_path)
    except Exception as e:
        logger.warning("Could not generate data preview: %s", e)

    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = Path("models/truthlens-gemma-finetuned")
    result = run_finetune(output)
    print("Fine-tuning config:")
    print(json.dumps(result, indent=2))
    print("\nTo actually fine-tune, run the Colab notebook.")
    print("Colab link: https://colab.research.google.com")