import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------
def load_csv_to_df(path):
    """Load CSV file into pandas DataFrame."""
    return pd.read_csv(path)


# --------------------------------------------------
# HELPER: CONTIGUOUS GROUPS
# --------------------------------------------------
def _contiguous_groups(idxs):
    """Turn sorted integer indices into contiguous groups."""
    if len(idxs) == 0:
        return []
    idxs = np.sort(np.array(idxs))
    groups = []
    current = [int(idxs[0])]

    for i in idxs[1:]:
        if int(i) == current[-1] + 1:
            current.append(int(i))
        else:
            groups.append(np.array(current))
            current = [int(i)]

    groups.append(np.array(current))
    return groups


# --------------------------------------------------
# APPLY ANOMALY MODE
# --------------------------------------------------
def _apply_mode(df, idxs, numeric_cols, mode, scale_factor):
    if len(idxs) == 0:
        return

    groups = _contiguous_groups(idxs) if mode in ("drift", "stuck") else [idxs]

    for col in numeric_cols:
        col_std = df[col].std(ddof=0)

        for group in groups:
            if len(group) == 0:
                continue

            if mode == "amplitude":
                df.loc[group, col] = df.loc[group, col] * (
                    1 + np.random.uniform(scale_factor * 0.5, scale_factor, size=len(group))
                )

            elif mode == "spike":
                spikes = np.random.normal(
                    0, scale_factor * (col_std if not np.isnan(col_std) else 1.0), size=len(group)
                )
                df.loc[group, col] += spikes

            elif mode == "drift":
                length = len(group)
                drift = np.linspace(0, scale_factor * (col_std if not np.isnan(col_std) else 1.0), length)
                df.loc[group, col] = df.loc[group, col].values + drift

            elif mode == "stuck":
                val = df.loc[group[0], col]
                df.loc[group, col] = val

            else:
                spikes = np.random.normal(
                    0, scale_factor * (col_std if not np.isnan(col_std) else 1.0), size=len(group)
                )
                df.loc[group, col] += spikes


# --------------------------------------------------
# INJECT ANOMALIES
# --------------------------------------------------
def inject_anomalies(df, ratio=0.05, scale_factor=3.0, segment_length=50, mode="mix", seed=42):
    """
    Inject synthetic anomalies into numeric columns.
    Adds/uses 'faulty' column as label.
    """
    np.random.seed(seed)
    df = df.copy()

    if "faulty" not in df.columns:
        df["faulty"] = 0.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "faulty" in numeric_cols:
        numeric_cols.remove("faulty")

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for anomaly injection.")

    n_total = len(df)
    n_to_affect = max(1, int(n_total * ratio))
    affected = np.zeros(n_total, dtype=bool)

    while affected.sum() < n_to_affect:
        start = np.random.randint(0, n_total)
        seg = int(np.random.normal(loc=segment_length, scale=max(1, segment_length * 0.2)))
        seg = max(1, min(seg, n_total))
        end = min(n_total, start + seg)
        affected[start:end] = True

    idxs = np.where(affected)[0]

    if mode == "mix":
        groups = _contiguous_groups(idxs)
        for g in groups:
            chosen = np.random.choice(["amplitude", "spike", "drift", "stuck"])
            _apply_mode(df, g, numeric_cols, chosen, scale_factor)
    else:
        _apply_mode(df, idxs, numeric_cols, mode, scale_factor)

    df.loc[idxs, "faulty"] = 1.0
    print(f"⚙️ Injected realistic anomalies: approx {affected.sum()} rows ({affected.sum()/n_total:.3%})")

    return df


# --------------------------------------------------
# PREPROCESS DATA
# --------------------------------------------------
def preprocess_data(df, window_size, normalize=True, step=1):
    """
    Convert DataFrame into sliding windows.

    Returns:
        X -> (n_windows, window_size, n_features)
        y -> (n_windows,)
        scaler -> fitted StandardScaler or None
    """

    df = df.ffill().bfill().reset_index(drop=True)

    if "faulty" not in df.columns:
        df["faulty"] = 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "faulty" in numeric_cols:
        numeric_cols.remove("faulty")

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for modeling.")

    features_df = df[numeric_cols].copy()

    scaler = None
    if normalize:
        scaler = StandardScaler()
        features_df.loc[:, :] = scaler.fit_transform(features_df.values)

    X, y = [], []
    total_len = len(df)

    for i in range(0, total_len - window_size, step):
        X.append(features_df.iloc[i:i + window_size].values)
        y.append(int(df["faulty"].iloc[i + window_size - 1]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, scaler, numeric_cols

