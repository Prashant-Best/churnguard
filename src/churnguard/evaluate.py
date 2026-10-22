"""Metrics, threshold selection and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Ranking metrics (threshold-free) plus decision metrics at the chosen threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_true, proba, strategy: str = "f1", min_recall: float = 0.75) -> float:
    """Pick the probability cut-off that turns a score into a yes/no decision.

    0.5 is rarely right for imbalanced problems. Missing a churner usually costs more than
    an unnecessary retention call, so the default strategy keeps recall above a target.
    The threshold is always chosen on out-of-fold training predictions, never on the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds; drop the last point.
    precision, recall = precision[:-1], recall[:-1]

    if strategy == "f1":
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        return float(thresholds[int(np.argmax(f1))])
    if strategy == "recall":
        ok = np.where(recall >= min_recall)[0]
        if len(ok) == 0:
            return float(thresholds[0])
        return float(thresholds[ok[-1]])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def gain_at(y_true, proba, fraction: float = 0.2) -> float:
    """Share of all churners found in the top `fraction` of customers ranked by risk."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    k = max(1, int(round(len(y_true) * fraction)))
    total = y_true.sum()
    return float(y_true[order[:k]].sum() / total) if total else 0.0


# ---------------------------------------------------------------- plots


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_roc_pr(y_true, probas: dict[str, np.ndarray], out_path: Path) -> None:
    """ROC and precision-recall curves for every model on one figure."""
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax2.plot(rec, prec, label=f"{name} (AP {average_precision_score(y_true, proba):.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax1.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    base = float(np.mean(y_true))
    ax2.axhline(base, linestyle="--", color="grey", linewidth=1, label=f"baseline ({base:.2f})")
    ax2.set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(metrics: dict, out_path: Path, title: str) -> None:
    plt = _plt()
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        ax.text(j, i, f"{value:,}", ha="center", va="center", color=color, fontsize=13)
    ax.set_xticks([0, 1], ["Stays", "Churns"])
    ax.set_yticks([0, 1], ["Stays", "Churns"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gain(y_true, proba, out_path: Path, highlight: float = 0.2) -> None:
    """Cumulative gain: how fast we find churners if we contact customers riskiest-first."""
    plt = _plt()
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    found = np.cumsum(y_true[order]) / y_true.sum()
    contacted = np.arange(1, len(y_true) + 1) / len(y_true)
    g = gain_at(y_true, proba, highlight)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(contacted, found, label="Model ranking")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random calls")
    ax.scatter([highlight], [g], zorder=3)
    ax.annotate(
        f"Top {highlight:.0%} -> {g:.0%} of churners",
        (highlight, g),
        textcoords="offset points",
        xytext=(10, -15),
    )
    ax.set(
        title="Cumulative gain (test set)",
        xlabel="Share of customers contacted",
        ylabel="Share of churners reached",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(importance, out_path: Path, top: int = 15) -> None:
    """Horizontal bar chart of mean |SHAP| per original feature."""
    plt = _plt()
    data = importance.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(data.index, data.values, color="#3b5bdb")
    ax.set(title="What drives churn (mean |SHAP value|)", xlabel="Average impact on churn log-odds")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
