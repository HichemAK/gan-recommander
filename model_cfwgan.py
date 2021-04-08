from torch import nn
import torch
import math
import itertools

class MLPTower(nn.Module):
    """Tower-shaped MLP.
    Example : MLPTower(16, 2, 3) would give you a network 16-8-4-2-2 with ReLu after each layer except for last one"""
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

class MLPRepeat(nn.Module):
    """Repeat-shaped MLP.
        Example : RepeatMLP(16, 8, 12, 3) would give you a network 16-12-12-12-8 with ReLu after each layer except for last one"""
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers):
        super().__init__()
        l = [(nn.Linear(hidden_size, hidden_size), nn.ReLU(True)) for _ in range(num_hidden_layers-1)]
        l.insert(0, (nn.Linear(input_size, hidden_size), nn.ReLU(True)))
        l = list(itertools.chain(*l))
        l.append(nn.Linear(hidden_size, output_size))
        self.sequential = nn.Sequential(*l)

    def forward(self, x):
        return self.sequential(x)


class Generator(nn.Module):
    def __init__(self, num_items, hidden_size, num_hidden_layers):
        super().__init__()
        self.mlp_repeat = MLPRepeat(2 * num_items, num_items, hidden_size, num_hidden_layers)

    def forward(self, item_neg_sample, item_full):
        x = torch.cat([item_neg_sample, item_full], dim=-1)
        return self.mlp_repeat(x)

class Discriminator(nn.Module):
    def __init__(self, num_items, num_hidden_layers):
        super().__init__()
        self.mlp_tower = MLPTower(2*num_items, 1, num_hidden_layers)

    def forward(self, generator_output, item_full):
        x = torch.cat([generator_output, item_full], dim=-1)
        return self.mlp_tower(x)
