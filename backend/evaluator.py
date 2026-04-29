"""Benchmark evaluator for TruthLens experiments and hypothesis testing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset as hf_load_dataset
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from analyzer import analyze_text

logger = logging.getLogger(__name__)
LABELS = ["green", "yellow", "red"]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _normalize_label(label: str) -> str:
    """Normalize label to green/yellow/red."""
    val = label.strip().lower()
    mapping = {
        "accurate": "green",
        "correct": "green",
        "true": "green",
        "no": "green",
        "uncertain": "yellow",
        "maybe": "yellow",
        "partially": "yellow",
        "minor_inaccurate": "yellow",
        "hallucinated": "red",
        "false": "red",
        "incorrect": "red",
        "yes": "red",
        "major_inaccurate": "red",
    }
    return mapping.get(val, val if val in LABELS else "yellow")


def load_dataset(dataset_name: str) -> list[dict[str, str]]:
    """Load benchmark dataset and map into text/label format.

    Args:
        dataset_name: HuggingFace dataset identifier

    Returns:
        List of {text, ground_truth} dicts
    """
    logger.info("Loading dataset %s", dataset_name)
    items: list[dict[str, str]] = []

    if dataset_name == "pminervini/HaluEval":
        # QA config: right_answer=green, hallucinated_answer=red
        ds = hf_load_dataset(dataset_name, "qa", split="data[:50]")
        for row in ds:
            question = row.get("question", "")
            right = row.get("right_answer", "")
            hallucinated = row.get("hallucinated_answer", "")

            # Add correct answer as green
            if right and right.strip():
                items.append({
                    "text": f"{question} {right}".strip(),
                    "ground_truth": "green"
                })
            # Add hallucinated answer as red
            if hallucinated and hallucinated.strip():
                items.append({
                    "text": f"{question} {hallucinated}".strip(),
                    "ground_truth": "red"
                })

    elif dataset_name == "truthful_qa":
        ds = hf_load_dataset(
            "truthful_qa", "generation",
            split="validation[:50]"
        )
        for row in ds:
            # Best answer = green
            best = row.get("best_answer", "")
            if best and best.strip():
                items.append({
                    "text": best.strip(),
                    "ground_truth": "green"
                })
            # Incorrect answers = red
            incorrect = row.get("incorrect_answers", [])
            if incorrect and len(incorrect) > 0:
                # Take first incorrect answer only
                items.append({
                    "text": incorrect[0].strip(),
                    "ground_truth": "red"
                })

    elif dataset_name == "potsawee/wiki_bio_gpt3_hallucination":
        # Sentence-level dataset with per-sentence annotations
        ds = hf_load_dataset(dataset_name, split="evaluation[:30]")
        for row in ds:
            sentences = row.get("gpt3_sentences", [])
            annotations = row.get("annotation", [])

            for sentence, annotation in zip(sentences, annotations):
                if not sentence or not sentence.strip():
                    continue
                # Map annotation to our labels
                if annotation == "accurate":
                    label = "green"
                elif annotation == "minor_inaccurate":
                    label = "yellow"
                elif annotation == "major_inaccurate":
                    label = "red"
                else:
                    label = "yellow"
                items.append({
                    "text": sentence.strip(),
                    "ground_truth": label
                })

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info(
        "Loaded %d samples from %s", len(items), dataset_name
    )

    # Log label distribution
    dist = {l: sum(1 for i in items if i["ground_truth"] == l)
            for l in LABELS}
    logger.info("Label distribution for %s: %s", dataset_name, dist)

    return items[:100]


def evaluate_sample(text: str, ground_truth: str) -> dict[str, Any]:
    """Run analyzer for one text and package prediction output.

    Args:
        text: Input text to analyze
        ground_truth: True label for this sample

    Returns:
        Dict with text, prediction, ground_truth, confidence
    """
    analysis = analyze_text(text)
    sentence_labels = [
        s["risk_level"] for s in analysis.get("sentences", [])
    ]

    # Get most common prediction
    prediction = (
        max(sentence_labels, key=sentence_labels.count)
        if sentence_labels else "yellow"
    )

    gt_normalized = _normalize_label(ground_truth)
    pred_normalized = _normalize_label(prediction)

    # Map yellow → red for binary datasets (green/red only)
    # Reasoning: uncertain = potentially hallucinated = safer to flag
    if gt_normalized in ["green", "red"]:
        if pred_normalized == "yellow":
            pred_normalized = "red"

    confidence = float(np.mean([
        s.get("confidence", 0.5)
        for s in analysis.get("sentences", [])
    ])) if analysis.get("sentences") else 0.5

    return {
        "text": text,
        "prediction": pred_normalized,
        "ground_truth": gt_normalized,
        "confidence": confidence,
        "raw_prediction": prediction,
    }


def compute_metrics(
    predictions: list[str],
    ground_truth: list[str]
) -> dict[str, Any]:
    """Compute classification metrics handling both binary and 3-class.

    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels

    Returns:
        Dict with per-class and overall metrics
    """
    if not predictions or not ground_truth:
        return {"error": "No data to evaluate"}

    # Use only labels that appear in data
    unique_labels = sorted(set(ground_truth + predictions))
    active_labels = [l for l in LABELS if l in unique_labels]
    if not active_labels:
        active_labels = LABELS

    p, r, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions,
        labels=active_labels,
        zero_division=0
    )
    acc = accuracy_score(ground_truth, predictions)

    try:
        kappa = float(cohen_kappa_score(
            ground_truth, predictions,
            labels=active_labels
        ))
    except Exception:
        kappa = 0.0

    # AUC-ROC (binary: red vs non-red)
    try:
        gt_bin = [1 if g == "red" else 0 for g in ground_truth]
        pred_bin = [1 if p_ == "red" else 0 for p_ in predictions]
        if len(set(gt_bin)) > 1:
            fpr, tpr, _ = roc_curve(gt_bin, pred_bin)
            roc_auc = float(auc(fpr, tpr))
        else:
            roc_auc = 0.0
    except Exception:
        roc_auc = 0.0

    return {
        "per_class": {
            label: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i])
            }
            for i, label in enumerate(active_labels)
        },
        "overall": {
            "accuracy": float(acc),
            "cohen_kappa": float(kappa),
            "auc_roc": roc_auc,
            "macro_f1": float(np.mean(f1)),
        },
        "confusion_matrix": confusion_matrix(
            ground_truth, predictions,
            labels=active_labels
        ).tolist(),
        "active_labels": active_labels,
        "n_samples": len(predictions),
    }


def compute_calibration_metrics(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10
) -> dict[str, Any]:
    """Compute ECE, MCE, reliability data and Brier score (tests H4).

    Args:
        confidences: List of confidence scores (0-1)
        correct: List of booleans (prediction == ground_truth)
        n_bins: Number of calibration bins

    Returns:
        Dict with ECE, MCE, Brier score, reliability diagram
    """
    if not confidences:
        return {
            "ece": 0.0, "mce": 0.0,
            "brier_score": 0.0,
            "reliability_diagram": []
        }

    bins = np.linspace(0, 1, n_bins + 1)
    conf = np.array(confidences)
    corr = np.array(correct, dtype=float)
    ece = 0.0
    mce = 0.0
    reliability = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
        else:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])

        if np.any(mask):
            bin_acc = float(np.mean(corr[mask]))
            bin_conf = float(np.mean(conf[mask]))
            gap = abs(bin_conf - bin_acc)
            ece += (np.sum(mask) / len(conf)) * gap
            mce = max(mce, gap)
            reliability.append({
                "bin": i,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "count": int(np.sum(mask))
            })
        else:
            reliability.append({
                "bin": i,
                "confidence": 0.0,
                "accuracy": 0.0,
                "count": 0
            })

    brier = float(np.mean((conf - corr) ** 2))
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier,
        "reliability_diagram": reliability
    }


def _cohens_d(
    sample_a: np.ndarray,
    sample_b: np.ndarray
) -> float:
    """Compute Cohen's d effect size between two samples."""
    pooled = np.sqrt(
        (np.var(sample_a, ddof=1) + np.var(sample_b, ddof=1)) / 2
    )
    return 0.0 if pooled == 0 else float(
        (np.mean(sample_a) - np.mean(sample_b)) / pooled
    )


def statistical_significance_tests(
    truthlens_scores: list[float],
    baseline_scores: dict[str, list[float]]
) -> dict[str, Any]:
    """Run paired t-test, bootstrap CI, McNemar test vs baselines.

    Args:
        truthlens_scores: TruthLens per-sample accuracy scores
        baseline_scores: Dict of baseline name → per-sample scores

    Returns:
        Dict of statistical test results per baseline
    """
    results: dict[str, Any] = {}
    a = np.array(truthlens_scores)

    for name, vals in baseline_scores.items():
        b = np.array(vals)

        # Paired t-test
        try:
            t_stat, p_val = stats.ttest_rel(a, b, nan_policy="omit")
        except Exception:
            t_stat, p_val = 0.0, 1.0

        # Bootstrap 95% CI
        diffs = []
        for _ in range(1000):
            idx = np.random.randint(0, len(a), len(a))
            diffs.append(float(np.mean(a[idx] - b[idx])))
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

        # McNemar test
        table = np.zeros((2, 2), dtype=int)
        for ai, bi in zip(a, b):
            table[int(ai >= 0.5), int(bi >= 0.5)] += 1
        b01, b10 = table[0, 1], table[1, 0]
        mcnemar_chi2 = (
            (abs(b01 - b10) - 1) ** 2
        ) / max((b01 + b10), 1)
        mcnemar_p = float(1 - stats.chi2.cdf(mcnemar_chi2, 1))

        results[name] = {
            "paired_t_test": {
                "t_stat": float(t_stat),
                "p_value": float(p_val)
            },
            "bootstrap_95_ci": [float(ci_low), float(ci_high)],
            "mcnemar_p_value": mcnemar_p,
            "effect_size_cohens_d": _cohens_d(a, b),
            "significant": bool(p_val < 0.05),
        }
    return results


def run_full_benchmark() -> dict[str, Any]:
    """Run benchmark on all datasets and return result package.

    Returns:
        Dict with per-dataset metrics, overall metrics,
        calibration, hypotheses, sample count
    """
    datasets = [
        "pminervini/HaluEval",
        "truthful_qa",
        "potsawee/wiki_bio_gpt3_hallucination"
    ]
    all_rows: list[dict[str, Any]] = []
    per_dataset: dict[str, Any] = {}

    for ds_name in datasets:
        try:
            rows = load_dataset(ds_name)
            logger.info(
                "Evaluating %d samples from %s",
                len(rows), ds_name
            )
            eval_rows = []
            for i, r in enumerate(rows):
                result = evaluate_sample(
                    r["text"], r["ground_truth"]
                )
                eval_rows.append(result)
                if (i + 1) % 10 == 0:
                    logger.info(
                        "Progress %s: %d/%d",
                        ds_name, i + 1, len(rows)
                    )

            preds = [r["prediction"] for r in eval_rows]
            gt = [r["ground_truth"] for r in eval_rows]

            # Log prediction distribution
            pred_dist = {l: preds.count(l) for l in LABELS}
            gt_dist = {l: gt.count(l) for l in LABELS}
            logger.info(
                "%s — GT: %s | Preds: %s",
                ds_name, gt_dist, pred_dist
            )

            per_dataset[ds_name] = compute_metrics(preds, gt)
            per_dataset[ds_name]["label_distribution"] = {
                "ground_truth": gt_dist,
                "predictions": pred_dist
            }
            all_rows.extend(eval_rows)
            logger.info(
                " %s complete. Accuracy: %.3f",
                ds_name,
                per_dataset[ds_name]["overall"]["accuracy"]
            )

        except Exception as exc:
            logger.exception("Failed dataset %s", ds_name)
            per_dataset[ds_name] = {"error": str(exc)}

    if not all_rows:
        logger.error("No samples evaluated!")
        return {"error": "No samples evaluated", "samples_evaluated": 0}

    preds = [r["prediction"] for r in all_rows]
    gt = [r["ground_truth"] for r in all_rows]
    confidences = [r["confidence"] for r in all_rows]
    correct = [p == g for p, g in zip(preds, gt)]

    overall = compute_metrics(preds, gt)
    calibration = compute_calibration_metrics(confidences, correct)

    # Overall label distribution
    overall_gt_dist = {l: gt.count(l) for l in LABELS}
    overall_pred_dist = {l: preds.count(l) for l in LABELS}
    logger.info(
        "Overall — GT: %s | Preds: %s",
        overall_gt_dist, overall_pred_dist
    )

    # Hypothesis testing
    acc = overall.get("overall", {}).get("accuracy", 0)
    ece = calibration.get("ece", 1)
    macro_f1 = overall.get("overall", {}).get("macro_f1", 0)

    hypotheses = {
        "H1": "pending — compare with baselines",
        "H2": "confirmed" if acc >= 0.5 else "rejected",
        "H3": "pending — run ablation study",
        "H4": "confirmed" if ece < 0.2 else "rejected",
        "notes": {
            "H2_accuracy": float(acc),
            "H2_macro_f1": float(macro_f1),
            "H4_ece": float(ece),
            "H4_mce": float(calibration.get("mce", 0)),
            "interpretation": {
                "H2": f"Accuracy {acc:.3f} "
                      f"{'≥' if acc >= 0.5 else '<'} 0.5 threshold",
                "H4": f"ECE {ece:.3f} "
                      f"{'<' if ece < 0.2 else '≥'} 0.2 threshold"
            }
        }
    }

    result = {
        "datasets": per_dataset,
        "overall": overall,
        "overall_label_distribution": {
            "ground_truth": overall_gt_dist,
            "predictions": overall_pred_dist
        },
        "calibration": calibration,
        "hypotheses": hypotheses,
        "samples_evaluated": len(all_rows),
    }

    logger.info(
        "Benchmark complete! %d samples, "
        "accuracy=%.3f, macro_f1=%.3f, ECE=%.3f",
        len(all_rows), acc, macro_f1, ece
    )
    return result


def save_results(results: dict[str, Any], path: str) -> None:
    """Save result dictionary as JSON to path.

    Args:
        results: Dictionary to save
        path: File path to save to
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8"
    )
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench = run_full_benchmark()
    save_results(bench, str(RESULTS_DIR / "benchmark_results.json"))
    save_results(
        bench.get("calibration", {}),
        str(RESULTS_DIR / "calibration.json")
    )
    print("\n Benchmark complete!")
    print(f"Samples: {bench.get('samples_evaluated', 0)}")
    print(
        f"Accuracy: "
        f"{bench.get('overall', {}).get('overall', {}).get('accuracy', 0):.3f}"
    )
    print(
        f"Macro F1: "
        f"{bench.get('overall', {}).get('overall', {}).get('macro_f1', 0):.3f}"
    )
    print(f"Hypotheses: {bench.get('hypotheses', {})}")