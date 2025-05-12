import torch
from torch.utils.data import Dataset
from src.utils import transform_to_gaf

class GAFDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_gaf = transform_to_gaf(self.X[idx].reshape(1, -1))  # shape: (1, 28, 28)
        x_tensor = torch.tensor(x_gaf[0], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor.unsqueeze(0), y_tensor
