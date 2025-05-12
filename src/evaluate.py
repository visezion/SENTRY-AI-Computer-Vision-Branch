import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    classification_report
)
from torch.utils.data import DataLoader, TensorDataset

from .config import DATASET_NAME, MODEL_SAVE_PATH
from .utils import load_dataset, normalize_and_save, transform_to_gaf
from .models import AdvancedCNN as CNNModel


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model():
    set_seed()
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

    # Create DataLoader for batch inference
    test_dataset = TensorDataset(torch.tensor(X_test_gaf, dtype=torch.float32), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    preds, probs, labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            if outputs.shape[1] == 1:
                prob = torch.sigmoid(outputs).squeeze()
                pred = (prob > 0.5).int()
            else:
                prob = F.softmax(outputs, dim=1)
                pred = torch.argmax(prob, dim=1)
                prob = prob[:, 1]  # Use only positive class prob for AUC

            preds.extend(pred.tolist())
            probs.extend(prob.tolist())
            labels.extend(batch_y.tolist())

    # Evaluation Metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs)
    balanced_acc = balanced_accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    class_report = classification_report(labels, preds, digits=4)

    print("\n📊 === Evaluation Metrics ===")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision:           {prec:.4f}")
    print(f"Recall:              {rec:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"AUC-ROC:             {auc:.4f}")
    print(f"Balanced Accuracy:   {balanced_acc:.4f}")
    print("\n📉 Confusion Matrix:")
    print(conf_matrix)
    print("\n📋 Classification Report:")
    print(class_report)

    # Optional: Save to file
    output_path = f"outputs/eval_metrics_{DATASET_NAME.lower()}.txt"
    with open(output_path, "w") as f:
        f.write("=== Evaluation Metrics ===\n")
        f.write(f"Accuracy:            {acc:.4f}\n")
        f.write(f"Precision:           {prec:.4f}\n")
        f.write(f"Recall:              {rec:.4f}\n")
        f.write(f"F1-Score:            {f1:.4f}\n")
        f.write(f"AUC-ROC:             {auc:.4f}\n")
        f.write(f"Balanced Accuracy:   {balanced_acc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix) + "\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)

    print(f"\n💾 Metrics saved to {output_path}")


if __name__ == '__main__':
    evaluate_model()
