import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self):
        self.item_count = 0
        raise NotImplementedError
