from torch import nn
import torch
import math
import itertools

class LinearTower(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers):
        super().__init__()
        c = 2**math.floor(math.log2(input_size))
        if c == input_size:
            c //= 2

        l = [(nn.Linear(c//2**i, c//2**(i+1)), nn.ReLU(True)) for i in range(num_hidden_layers-1)]
        l.insert(0, (nn.Linear(input_size, c), nn.ReLU(True)))
        l = list(itertools.chain(*l))
        l.append(nn.Linear(c//2**(num_hidden_layers-1), output_size))
        self.sequential = nn.Sequential(*l)

    def forward(self, x):
        return self.sequential(x)


class Generator(nn.Module):
    def __init__(self, num_items, num_layers):
        super().__init__()
        self.num_items = num_items
        c = math.floor(math.log2(num_items))
        l = [nn.Linear()]
