
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .config import MODEL_SAVE_PATH
from .utils import load_nsl_kdd, transform_to_gaf, normalize_and_save
from .train import CNNModel

def evaluate_model():
    X, y = load_nsl_kdd()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize_and_save(X_train, X_test)
    X_test_gaf = transform_to_gaf(X_test)

    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test_gaf, dtype=torch.float32)
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1).numpy()
        y_prob = probs[:, 1]
        y_pred = np.argmax(probs, axis=1)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == '__main__':
    evaluate_model()