import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class DiabetesDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.X = data.iloc[:, :-1].values.astype(np.float32)
        self.y = data.iloc[:, -1].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_loaders(batch_size, num_workers=0):
    train_ds = DiabetesDataset('dataset/train.csv')
    val_ds = DiabetesDataset('dataset/tune.csv')
    test_ds = DiabetesDataset('dataset/test.csv')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    input_dim = train_ds.X.shape[1]
    return train_loader, val_loader, test_loader, input_dim