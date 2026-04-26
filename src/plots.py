import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {path}")

def plot_losses(history, save_path):
    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="Train")
    ax.set_title("Training Loss")
    ax.legend()
    _save(fig, save_path)

def plot_roc_curve(y_true, scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],'--')
    ax.legend()
    _save(fig, save_path)

def plot_pr_curve(y_true, scores, save_path):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    _save(fig, save_path)

def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    _save(fig, save_path)

def plot_feature_contribution(contrib, save_path):
    fig, ax = plt.subplots()
    ax.bar(range(len(contrib)), contrib)
    _save(fig, save_path)

def plot_model_comparison_bar(f1_scores, save_path):
    labels = ["CNN-LSTM", "Isolation", "Ensemble"]
    fig, ax = plt.subplots()
    ax.bar(labels, f1_scores)
    _save(fig, save_path)

def plot_sensor_timeseries_with_anomalies(signal, anomaly_idx, save_path, title):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(signal)
    ax.scatter(anomaly_idx, signal[anomaly_idx], color='red')
    ax.set_title(title)
    _save(fig, save_path)

def plot_error_timeseries(errors, threshold, save_path):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(errors)
    ax.axhline(threshold, color='red', linestyle='--')
    ax.set_title("Reconstruction Error Over Time")
    _save(fig, save_path)
