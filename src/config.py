import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_PATH = os.path.join(BASE_DIR, 'data', 'nsl_kdd.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'cnn_model.pth')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
GRADCAM_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'gradcam_heatmaps')
FUSION_OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'fusion_metrics.txt') 

# Image size for GAF
IMAGE_SIZE = 28

# Training
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
