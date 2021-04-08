import itertools
import math
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch import nn


class MLPTower(nn.Module):
    """Tower-shaped MLP.
    Example : MLPTower(16, 2, 3) would give you a network 16-8-4-2-2 with ReLu after each layer except for last one"""

    def __init__(self, input_size, output_size, num_hidden_layers):
        super().__init__()
        c = 2 ** math.floor(math.log2(input_size))
        if c == input_size:
            c //= 2

        l = [(nn.Linear(c // 2 ** i, c // 2 ** (i + 1)), nn.ReLU(True)) for i in range(num_hidden_layers - 1)]
        l.insert(0, (nn.Linear(input_size, c), nn.ReLU(True)))
        l = list(itertools.chain(*l))
        l.append(nn.Linear(c // 2 ** (num_hidden_layers - 1), output_size))
        self.sequential = nn.Sequential(*l)

    def forward(self, x):
        return self.sequential(x)


class MLPRepeat(nn.Module):
    """Repeat-shaped MLP.
        Example : RepeatMLP(16, 8, 12, 3) would give you a network 16-12-12-12-8 with ReLu after each layer except for last one"""

    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers):
        super().__init__()
        l = [(nn.Linear(hidden_size, hidden_size), nn.ReLU(True)) for _ in range(num_hidden_layers - 1)]
        l.insert(0, (nn.Linear(input_size, hidden_size), nn.ReLU(True)))
        l = list(itertools.chain(*l))
        l.append(nn.Linear(hidden_size, output_size))
        self.sequential = nn.Sequential(*l)

    def forward(self, x):
        return self.sequential(x)


class Generator(nn.Module):
    def __init__(self, num_items, hidden_size, num_hidden_layers):
        super().__init__()
        self.mlp_repeat = MLPRepeat(num_items, num_items, hidden_size, num_hidden_layers)

    def forward(self, items):
        return self.mlp_repeat(items)


class Discriminator(nn.Module):
    def __init__(self, num_items, num_hidden_layers):
        super().__init__()
        self.mlp_tower = nn.Sequential(nn.Linear(2*num_items, 1024), nn.ReLU(True), nn.Linear(1024, 512), nn.ReLU(True),
                                       nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, 1))

    def forward(self, generator_output, item_full):
        x = torch.cat([generator_output, item_full], dim=-1)
        return self.mlp_tower(x)


class CFWGAN(pl.LightningModule):
    def __init__(self, num_items, alpha=0.04, beta=0.04, g_steps=1, d_steps=1):
        super().__init__()
        self.generator = Generator(num_items, 128, 3)
        self.discriminator = Discriminator(num_items, 3)
        self.g_steps = g_steps
        self.d_steps = d_steps
        self._g_steps = 0
        self._d_steps = 0
        self.alpha = alpha
        self.beta = beta

    def forward(self, item_full):
        x = self.generator(item_full)
        return x

    def negative_sampling(self, items):
        # TODO : Implement negative sampling
        return items

    def training_step(self, batch, batch_idx, optimizer_idx):
        items = batch

        # clip_value = 0.01

        # train generator
        if optimizer_idx == 0:
            # adversarial loss is binary cross-entropy
            g_loss = -torch.mean(self.discriminator(self.generator(self.negative_sampling(items)), items))
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples

        # discriminator loss is the average of these
        elif optimizer_idx == 1:
            d_loss = -torch.mean(self.discriminator(items, items)) + \
                     torch.mean(self.discriminator(self.generator(items), items))

            # for p in self.discriminator.parameters():
            #     p.data.clamp_(-clip_value, clip_value)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        items = batch
        generator_output = self.generator(self.negative_sampling(items))
        items_rank = torch.argsort(generator_output, dim=-1)
        items_rank = items_rank[:, :5]
        items = [set([x for x,y in enumerate(items[i]) if y == 1]) for i in range(items.shape[0])]
        precision_at_5 = sum([len(items[i].intersection(items_rank[i])) / 5 for i in range(len(items))]) / len(items)
        tqdm_dict = {'precision_at_5': precision_at_5}
        output = OrderedDict({
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output


    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters())
        opt_d = torch.optim.Adam(self.discriminator.parameters())
        return [opt_g, opt_d], []

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp,
    #                   using_lbfgs):
    #     # update generator opt every 2 steps
    #     if optimizer_idx == 0:
    #         if batch_idx % 2 == 0 :
    #             optimizer.step(closure=optimizer_closure)
    #             optimizer.zero_grad()
    #     # update discriminator opt every 4 steps
    #     elif optimizer_idx == 1:
    #         if batch_idx % 4 == 0 :
    #             optimizer.step(closure=optimizer_closure)
    #             optimizer.zero_grad()
