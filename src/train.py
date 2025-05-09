# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import *
from src.utils import load_dataset, normalize_and_save, transform_to_gaf
from src.models import AdvancedCNN


from tqdm import tqdm  # ← Add this import

def train_model():
    print(f"[INFO] Loading dataset: {DATASET_NAME}...")
    X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val = normalize_and_save(X_train, X_val)

    X_train_gaf = transform_to_gaf(X_train)
    X_val_gaf = transform_to_gaf(X_val)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_gaf), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_gaf), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)

    model = AdvancedCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_name = f"cnn_model_{DATASET_NAME.lower().replace('-', '_')}.pth"
    model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_name)

    if os.path.exists(model_path):
        user_input = input(f"⚠️ Found saved model for {DATASET_NAME}. Load instead of retraining? (y/n) [default: y]: ").strip().lower()
        if user_input in ['', 'y', 'yes']:
            model.load_state_dict(torch.load(model_path))
            print(f"✅ Loaded existing model from {model_path}")
            return model

    print("🚀 Training new model...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        preds_all, labels_all = [], []

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch_x, batch_y in progress:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_all.extend((outputs.squeeze() > 0.5).int().tolist())
            labels_all.extend(batch_y.tolist())

            # Optional: show loss live in tqdm bar
            progress.set_postfix(loss=loss.item())

        acc = accuracy_score(labels_all, preds_all)
        prec = precision_score(labels_all, preds_all, zero_division=0)
        rec = recall_score(labels_all, preds_all, zero_division=0)
        f1 = f1_score(labels_all, preds_all, zero_division=0)

        print(f"📊 Epoch {epoch+1} Summary → Loss: {running_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved model to {model_path}")

    return model
