from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# create dirs (safe to call multiple times)
for p in [DATA_DIR, RAW_DIR, PROC_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# reproducibility & device
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# sliding window / model defaults (you can override in main.py)
WINDOW_SIZE = 100
STEP = 1

# model defaults
LATENT_DIM = 64
CNN_CHANNELS = [32, 64]
LSTM_HIDDEN = 128
LSTM_LAYERS = 2

# training / model hyperparams
BATCH_SIZE = 128
LR = 5e-4
EPOCHS = 50
WEIGHT_DECAY = 1e-6

# thresholding & splits
THRESHOLD_METHOD = "percentile"
THRESHOLD_PERCENTILE = 98.0
TEST_SPLIT = 0.2
ANOMALY_RATIO = 0.05
