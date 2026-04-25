from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(labels: Iterable[int], scores: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    # Project convention: 0=real, 1=fake.
    y_true = np.asarray(list(labels)).astype(int)
    y_score = np.asarray(list(scores), dtype=float)
    y_pred = (y_score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    return metrics


def aggregate_video_predictions(frame_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    # Video score = mean over sampled frame scores.
    grouped = (
        frame_df.groupby(["video_id", "label", "dataset", "condition"], dropna=False)["score"]
        .mean()
        .reset_index()
    )
    grouped["pred"] = (grouped["score"] > threshold).astype(int)
    grouped["threshold"] = threshold
    return grouped


def sweep_thresholds(video_df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(video_df["label"], video_df["score"], threshold=threshold)
        rows.append({"threshold": threshold, **metrics})
    return pd.DataFrame(rows)


def plot_confusion_matrix(labels: Iterable[int], scores: Iterable[float], output_path: str | Path, threshold: float = 0.5) -> None:
    y_true = np.asarray(list(labels)).astype(int)
    y_pred = (np.asarray(list(scores)) > threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curve(labels: Iterable[int], scores: Iterable[float], output_path: str | Path) -> None:
    y_true = np.asarray(list(labels)).astype(int)
    y_score = np.asarray(list(scores), dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curve(labels: Iterable[int], scores: Iterable[float], output_path: str | Path) -> None:
    y_true = np.asarray(list(labels)).astype(int)
    y_score = np.asarray(list(scores), dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_by_condition(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str,
    kind: str = "bar",
) -> None:
    plt.figure(figsize=(8, 4))
    if kind == "bar":
        sns.barplot(data=df, x="condition", y=metric, color="#3c78d8")
    else:
        sns.lineplot(data=df, x="condition", y=metric, marker="o", color="#cc4125")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
