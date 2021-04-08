import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self):
        self.num_items = 0
        raise NotImplementedError
