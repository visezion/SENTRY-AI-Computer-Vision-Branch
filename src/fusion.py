# src/fusion.py

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.config import *
from src.models import AdvancedCNN, VAE
from src.utils import load_dataset, normalize_and_save, transform_to_gaf,fuse_predictions
from src.train import AdvancedCNN as CNNModel
from src.vae_train import train_vae
from torch.utils.data import DataLoader, TensorDataset


def evaluate_fusion():
    print(f"[INFO] Loading dataset: {DATASET_NAME} for fusion evaluation...")
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize_and_save(X_train, X_test)

    # === Load VAE ===
    vae_model = train_vae()  # already handles load/retrain logic
    vae_model.eval()
    vae_recon, _, _ = vae_model(torch.tensor(X_test, dtype=torch.float32))
    vae_recon = vae_recon.detach().numpy()
    vae_loss = np.mean((X_test - vae_recon) ** 2, axis=1)
    threshold = np.percentile(vae_loss, 95)
    y_prob_vae = (vae_loss - vae_loss.min()) / (vae_loss.max() - vae_loss.min())
    y_pred_vae = (vae_loss > threshold).astype(int)

    # === Load CNN ===
    cnn_model = CNNModel()
    model_name = f"cnn_model_{DATASET_NAME.lower().replace('-', '_')}.pth"
    model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_name)
    if os.path.exists(model_path):
        cnn_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print(f"✅ Loaded CNN model from {model_path}")

    cnn_model.eval()
    gaf_test = transform_to_gaf(X_test)

    if USE_BATCH_INFERENCE:
        batch_size = BATCH_SIZE
        cnn_outputs = []
        dataloader = DataLoader(TensorDataset(torch.tensor(gaf_test, dtype=torch.float32)), batch_size=batch_size)
        for batch in dataloader:
            with torch.no_grad():
                output = cnn_model(batch[0])
                cnn_outputs.append(output.squeeze().numpy())
        y_prob_cnn = np.concatenate(cnn_outputs, axis=0)
    else:
        inputs = torch.tensor(gaf_test, dtype=torch.float32)
        with torch.no_grad():
            cnn_outputs = cnn_model(inputs).squeeze().numpy()
        y_prob_cnn = cnn_outputs

    y_pred_cnn = (y_prob_cnn > 0.5).astype(int)

    # === Fusion (average) ===
    #y_prob_fusion = (y_prob_vae + y_prob_cnn) / 2
    #y_pred_fusion = (y_prob_fusion > 0.5).astype(int)

    y_prob_fusion, y_pred_fusion = fuse_predictions(y_prob_vae, y_prob_cnn, method='avg')

    def report(name, y_true, y_pred, y_prob):
        print(f"\n📊 {name} Metrics")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
        print(f"AUC-ROC:   {roc_auc_score(y_true, y_prob):.4f}")

    report("VAE", y_test, y_pred_vae, y_prob_vae)
    report("CNN", y_test, y_pred_cnn, y_prob_cnn)
    report("Fusion", y_test, y_pred_fusion, y_prob_fusion)

    # Save results (optional)
    result_path = os.path.join("outputs", f"fusion_metrics_{DATASET_NAME.lower()}.txt")
    with open(result_path, 'w') as f:
        f.write(f"Fusion Metrics for {DATASET_NAME}\n")
        f.write(f"Accuracy:  {accuracy_score(y_test, y_pred_fusion):.4f}\n")
        f.write(f"Precision: {precision_score(y_test, y_pred_fusion):.4f}\n")
        f.write(f"Recall:    {recall_score(y_test, y_pred_fusion):.4f}\n")
        f.write(f"F1-Score:  {f1_score(y_test, y_pred_fusion):.4f}\n")
        f.write(f"AUC-ROC:   {roc_auc_score(y_test, y_prob_fusion):.4f}\n")
    print(f"💾 Fusion report saved to {result_path}")
