# src/vae_train.py (new file or integrated into existing VAE logic)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.config import *
from src.utils import load_dataset, normalize_and_save
from src.models import VAE


def train_vae():
    print(f"[INFO] Loading dataset: {DATASET_NAME} for VAE...")
    X, y = load_dataset()
    X_train, X_val, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val = normalize_and_save(X_train, X_val)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)

    model = VAE(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_name = f"vae_model_{DATASET_NAME.lower().replace('-', '_')}.pth"
    model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_name)

    if os.path.exists(model_path):
        user_input = input(f"⚠️ Found saved VAE model for {DATASET_NAME}. Load instead of retraining? (y/n) [default: y]: ").strip().lower()
        if user_input in ['', 'y', 'yes']:

            model.load_state_dict(torch.load(model_path))
            print(f"✅ Loaded VAE from {model_path}")
            return model

    print("🚀 Training new VAE model...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved VAE model to {model_path}")
    return model
