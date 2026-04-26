import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import RAW_DIR, FIGURES_DIR, METRICS_PATH, WINDOW_SIZE, DEVICE
from data_utils import load_csv_to_df, inject_anomalies, preprocess_data
from model import CNNAutoencoderLSTM
from trainer import Trainer
from evaluator import find_best_threshold
from plots import (
    plot_losses,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_feature_contribution,
    plot_model_comparison_bar
)
from utils import save_json


def main():
    print("🚀 Starting Explainable Hybrid Ensemble Anomaly Detection Pipeline...")

    # ---------------- LOAD DATA ----------------
    dataset_path = RAW_DIR / "industrial_sensor_streams.csv"
    df = load_csv_to_df(dataset_path)
    print(f"📂 Loaded dataset: {dataset_path.name} | Shape: {df.shape}")

    # ---------------- INJECT ANOMALIES ----------------
    df = inject_anomalies(df, ratio=0.05, mode="mix")

    # ---------------- PREPROCESS ----------------
    X, y, scaler, feature_names = preprocess_data(df, window_size=WINDOW_SIZE, normalize=True)


    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Train only on normal windows
    normal_idx = y == 0
    X_train = X[normal_idx]
    y_train = y[normal_idx]

    # Test on all
    X_test = X
    y_test = y

    print(f"✅ Train windows (normal only): {len(X_train)}")
    print(f"🧪 Test windows (all): {len(X_test)}")

    # ---------------- CNN-LSTM AUTOENCODER ----------------
    model = CNNAutoencoderLSTM(
        input_dim=X.shape[2],
        cnn_filters=64,
        lstm_hidden=128,
        latent_dim=16
    )
    print(f"🧠 Model initialized on device: {DEVICE}")

    trainer = Trainer(model, X_train, X_val=None)
    history = trainer.train(epochs=50)

    plot_losses(history, FIGURES_DIR / "loss_curve.png")

    # ---------------- AE EVALUATION ----------------
    errors_ae, recons, latents, y_true = trainer.evaluate(X_test, y_test)
    best_thr_ae, metrics_ae = find_best_threshold(y_true, errors_ae)

    print("\nCNN-LSTM AE Metrics:")
    for k, v in metrics_ae.items():
        if k != "cm":
            print(f"  {k}: {v:.4f}")

    # ---------------- ISOLATION FOREST ----------------
    print("\n🌲 Training Isolation Forest...")
    flat_X = X.reshape(X.shape[0], -1)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(flat_X[normal_idx])

    iso_scores = -iso.decision_function(flat_X)
    best_thr_iso, metrics_iso = find_best_threshold(y_true, iso_scores)

    print("\nIsolation Forest Metrics:")
    for k, v in metrics_iso.items():
        if k != "cm":
            print(f"  {k}: {v:.4f}")

    # ---------------- HYBRID ENSEMBLE ----------------
    ensemble_scores = 0.6 * errors_ae + 0.4 * iso_scores
    best_thr_ens, metrics_ens = find_best_threshold(y_true, ensemble_scores)

    print("\nHybrid Ensemble Metrics:")
    for k, v in metrics_ens.items():
        if k != "cm":
            print(f"  {k}: {v:.4f}")

    print("\n=========== MODEL COMPARISON ===========")
    print(f"CNN-LSTM F1: {metrics_ae['f1']:.4f}")
    print(f"Isolation F1: {metrics_iso['f1']:.4f}")
    print(f"Ensemble F1: {metrics_ens['f1']:.4f}")

    # ---------------- EXPLAINABILITY ----------------
    print("\n🔍 Feature Contribution:")
    # mean abs reconstruction error per feature
    feat_err = np.mean(np.abs(X_test - recons), axis=1)
    avg_feat = feat_err.mean(axis=0)
    contrib = avg_feat / avg_feat.sum()

    for name, c in zip(feature_names, contrib):
        print(f"{name}: {c*100:.2f}%")


    # ---------------- JOURNAL-LEVEL PLOTS ----------------
    plot_roc_curve(y_true, ensemble_scores, FIGURES_DIR / "roc_curve.png")
    plot_pr_curve(y_true, ensemble_scores, FIGURES_DIR / "pr_curve.png")
    plot_confusion_matrix(metrics_ens["cm"], FIGURES_DIR / "confusion_matrix.png")
    plot_feature_contribution(contrib, FIGURES_DIR / "feature_contribution.png")
    plot_model_comparison_bar(
        [metrics_ae["f1"], metrics_iso["f1"], metrics_ens["f1"]],
        FIGURES_DIR / "model_comparison.png"
    )

    # ---------------- SAVE METRICS ----------------
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    save_json(make_serializable({
        "cnn_lstm": metrics_ae,
        "isolation_forest": metrics_iso,
        "ensemble": metrics_ens
    }), METRICS_PATH)

    print(f"\n✅ Metrics saved at: {METRICS_PATH}")
    print("🎉 Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
