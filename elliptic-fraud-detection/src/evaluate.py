"""
evaluate.py
-----------
Metrics that make sense when positives are rare.

Accuracy is a trap here (predict "always licit" -> ~90% accurate, 0 fraud caught).
So we report:
  - PR-AUC (average precision) : the headline metric for imbalanced detection.
  - ROC-AUC                    : ranking quality, for completeness.
  - Precision / Recall / F1    : on the illicit class, at a 0.5 threshold.
  - Precision @ top-k%         : if analysts can only review the top k% of
                                 flagged transactions, how many are truly illicit?
                                 This is what drives real alert queues.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)


def precision_at_top_k(y_true, scores, k_percent: float = 1.0) -> float:
    y_true = np.asarray(y_true)
    n = max(int(len(scores) * k_percent / 100.0), 1)
    top = np.argsort(scores)[::-1][:n]
    return float(y_true[top].mean())


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (scores >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_top_5pct": precision_at_top_k(y_true, scores, 5.0),
    }


def print_table(metrics: dict):
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "precision_at_top_5pct"]
    header = f"{'model':<26}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<26}" + "".join(f"{m[c]:>14.4f}" for c in cols))


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_pr_curves(scores_by_model: dict, y_true, path: str):
    plt.figure(figsize=(7, 5))
    for name, scores in scores_by_model.items():
        p, r, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(r, p, label=f"{name} (AP={ap:.3f})")
    plt.axhline(np.mean(y_true), ls="--", color="grey",
                label=f"random ({np.mean(y_true):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall: illicit transaction detection")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_confusion(y_true, scores, path: str, title: str, threshold: float = 0.5):
    cm = confusion_matrix(y_true, (scores >= threshold).astype(int))
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["licit", "illicit"]); plt.yticks([0, 1], ["licit", "illicit"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_feature_importance(model, feature_names, path: str, top_n: int = 15):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    order = np.argsort(imp)[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(names, imp[order][::-1], color="#2b7bba")
    plt.xlabel("Importance"); plt.title(f"Top {top_n} features (XGBoost)")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
