from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        self.num_items = num_items
        self.sequential =