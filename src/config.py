# config.py
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Dataset switching ---
# Default value
DATASET_NAME = "NSL-KDD"

# Override from CLI: --dataset CICIDS2017
for i, arg in enumerate(sys.argv):
    if arg == "--dataset" and i + 1 < len(sys.argv):
        DATASET_NAME = sys.argv[i + 1]
        break

# --- Auto-switch CSV path based on dataset ---
if "nsl" in DATASET_NAME.lower():
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'nsl_kdd.csv')
elif "cic" in DATASET_NAME.lower():
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'cicids2017_combined.csv')
elif "unsw" in DATASET_NAME.lower():
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'unsw_nb15.csv')
elif "sentry" in DATASET_NAME.lower():
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'sentry_combined.csv')
else:
    raise ValueError(f"❌ Unknown dataset name: {DATASET_NAME}")

# --- Model output paths ---
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', f"cnn_model_{DATASET_NAME.lower().replace('-', '_')}.pth")
SCALER_PATH = os.path.join(BASE_DIR, 'models', f"scaler_{DATASET_NAME.lower().replace('-', '_')}.pkl")
GRADCAM_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'gradcam_heatmaps', DATASET_NAME.lower())
FUSION_OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', f"fusion_metrics_{DATASET_NAME.lower().replace('-', '_')}.txt")

# --- GAF image settings ---
IMAGE_SIZE = 28

# --- Training ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
