import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .config import MODEL_SAVE_PATH, FUSION_OUTPUT_PATH
from .utils import load_nsl_kdd, transform_to_gaf, normalize_and_save
from .train import CNNModel


def load_dummy_vae_scores(X):
    """Simulate VAE reconstruction loss as anomaly scores (normally you'd load a real VAE)."""
    np.random.seed(42)
    vae_loss = np.random.rand(len(X)) * 0.5  # simulated anomaly scores
    return (vae_loss - vae_loss.min()) / (vae_loss.max() - vae_loss.min())

def fusion_predict(y_true, cnn_prob, vae_score):
    # Simple average fusion
    combined = (cnn_prob + vae_score) / 2
    y_pred = (combined > 0.5).astype(int)
    return y_pred, combined

def evaluate_fusion():
    X, y = load_nsl_kdd()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize_and_save(X_train, X_test)

    # Load CNN model and get CNN probs
    X_test_gaf = transform_to_gaf(X_test)
    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_gaf, dtype=torch.float32)
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1).numpy()
        cnn_probs = probs[:, 1]

    # Simulated VAE anomaly scores
    vae_scores = load_dummy_vae_scores(X_test)

    # Fusion prediction
    y_pred, y_fused = fusion_predict(y_test, cnn_probs, vae_scores)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_fused)

    with open(FUSION_OUTPUT_PATH, 'w') as f:
        f.write("=== Fusion Evaluation ===\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc:.4f}\n")

    print("\n=== Fusion Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

if __name__ == '__main__':
    evaluate_fusion()
