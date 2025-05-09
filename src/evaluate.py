import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .config import DATASET_NAME, MODEL_SAVE_PATH
from .utils import load_dataset, normalize_and_save, transform_to_gaf
from .models import AdvancedCNN as CNNModel

def evaluate_model():
    print(f"[INFO] Evaluating model for dataset: {DATASET_NAME}")
    
    # Load and preprocess dataset
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize_and_save(X_train, X_test)
    X_test_gaf = transform_to_gaf(X_test)

    # Build dynamic model path
    model_name = f"cnn_model_{DATASET_NAME.lower().replace('-', '_')}.pth"
    model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_name)

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    # Load model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_gaf, dtype=torch.float32)
        outputs = model(X_tensor)

        # Binary classification: apply sigmoid or softmax
        if outputs.shape[1] == 1:
            probs = torch.sigmoid(outputs).squeeze().numpy()
            y_prob = probs
            y_pred = (probs > 0.5).astype(int)
        else:
            probs = F.softmax(outputs, dim=1).numpy()
            y_prob = probs[:, 1]
            y_pred = np.argmax(probs, axis=1)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n📊 === Evaluation Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

if __name__ == '__main__':
    evaluate_model()
