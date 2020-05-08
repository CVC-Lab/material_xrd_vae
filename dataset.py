from torch.utils.data import Dataset
import torch

class ndarrayDataset(Dataset):
    """simple dataset"""

    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]