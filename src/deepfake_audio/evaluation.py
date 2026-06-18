"""Evaluation metrics and report plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Select the validation threshold minimizing balanced detection error."""
    _, threshold = compute_eer(y_true, probabilities)
    return float(np.clip(threshold, 0.05, 0.95))


def evaluate(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    eer, eer_threshold = compute_eer(y_true, probabilities)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "decision_threshold": float(threshold),
        "per_class_accuracy": {
            "genuine": float(per_class[0]),
            "deepfake": float(per_class[1]),
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Genuine", "Deepfake"],
            output_dict=True,
            zero_division=0,
        ),
        "samples": int(len(y_true)),
    }


def save_report(
    output_dir: Path,
    metrics: dict[str, Any],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import matplotlib.pyplot as plt
        import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    sns.set_theme(style="whitegrid")
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="mako",
        xticklabels=["Genuine", "Deepfake"],
        yticklabels=["Genuine", "Deepfake"],
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="#777777")
    axis.scatter(metrics["eer"], 1 - metrics["eer"], label=f"EER = {metrics['eer']:.3f}")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)
