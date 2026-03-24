"""
src/evaluation/evaluate.py

Compute all paper metrics and save results.

Metrics (paper Table 1):
  accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC

Also saves:
  per-fold CSV, mean CSV, ROC curve, Precision-Recall curve, confusion matrix
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
)


def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute all paper metrics from probability predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def summarise_folds(fold_metrics: list) -> dict:
    keys = [k for k in fold_metrics[0] if k not in ("tp","tn","fp","fn")]
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def print_summary(fold_metrics: list, condition: str = "RL-AUC"):
    print(f"\n{'='*65}")
    print(f"  {condition}")
    print(f"{'='*65}")
    print(f"{'Fold':>6} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'Spec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 65)
    for i, m in enumerate(fold_metrics):
        print(f"  {i+1:>4}  "
              f"{m['accuracy']:.4f}  {m['precision']:.4f}  "
              f"{m['recall']:.4f}  {m['specificity']:.4f}  "
              f"{m['f1']:.4f}  {m['roc_auc']:.4f}")
    print("-" * 65)
    means = summarise_folds(fold_metrics)
    print(f"  Mean  "
          f"{means['accuracy']:.4f}  {means['precision']:.4f}  "
          f"{means['recall']:.4f}  {means['specificity']:.4f}  "
          f"{means['f1']:.4f}  {means['roc_auc']:.4f}")
    print("=" * 65)
    return means


def save_results(fold_metrics: list, condition: str, reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    df   = pd.DataFrame(fold_metrics)
    df.index = [f"fold_{i+1}" for i in range(len(df))]
    mean_row = pd.DataFrame([summarise_folds(fold_metrics)], index=["mean"])
    df = pd.concat([df, mean_row])
    path = os.path.join(reports_dir, f"metrics_{condition}.csv")
    df.to_csv(path)
    print(f"  Saved: {path}")


def plot_roc(y_true, y_prob, condition: str, save_path: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"{condition} (AUC={auc:.4f})")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — DenseNet201"); plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr(y_true, y_prob, condition: str, save_path: str = None):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, lw=2, label=condition)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — DenseNet201")
    plt.legend(loc="upper right"); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion(y_true, y_prob, condition: str,
                   threshold: float = 0.5, save_path: str = None):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign","Malignant"],
                yticklabels=["Benign","Malignant"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix — {condition}")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()
