import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, accuracy_score

def select_threshold_percentile(errors, percentile=98.0):
    return float(np.percentile(errors, percentile))

def compute_metrics(y_true, scores, threshold):
    """
    Compute standard metrics given threshold on scores. scores: higher => more anomalous
    """
    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except:
        roc_auc = float("nan")
    pr_auc = average_precision_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "cm": cm
    }

def find_best_threshold(y_true, scores, percentiles=None):
    """
    Search thresholds over a grid (percentiles by default) and return the threshold that maximizes F1.
    Returns: best_threshold, best_metrics (dict)
    """
    if percentiles is None:
        percentiles = np.linspace(50, 99.9, 200)
    best_f1 = -1.0
    best_thr = None
    best_metrics = None
    for p in percentiles:
        thr = np.percentile(scores, p)
        metrics = compute_metrics(y_true, scores, thr)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thr = thr
            best_metrics = metrics
    if best_thr is None:
        best_thr = select_threshold_percentile(scores, 98.0)
        best_metrics = compute_metrics(y_true, scores, best_thr)
    return best_thr, best_metrics
