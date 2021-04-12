import itertools
import math
import random
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
        return torch.sigmoid(self.mlp_repeat(items))


class Discriminator(nn.Module):
    def __init__(self, num_items, num_hidden_layers):
        super().__init__()
        self.mlp_tower = nn.Sequential(nn.Linear(2 * num_items, 256), nn.ReLU(True), nn.Linear(256, 128),
                                       nn.ReLU(True),
                                       nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, generator_output, item_full):
        x = torch.cat([generator_output, item_full], dim=-1)
        return torch.sigmoid(self.mlp_tower(x))


class CFWGAN(pl.LightningModule):
    def __init__(self, trainset, num_items, alpha=0.04, s_zr=0.6, s_pm=0.6, g_steps=1, d_steps=1, debug=False):
        super().__init__()
        self.generator = Generator(num_items, 256, 3)
        self.discriminator = Discriminator(num_items, 3)
        self.g_steps = g_steps
        self.d_steps = d_steps
        self.alpha = alpha
        self.s_zr = s_zr
        self.s_pm = s_pm
        self.trainset = trainset
        self.debug = debug
        self.step_gd = 0
        self.automatic_optimization = False

    def forward(self, item_full):
        x = self.generator(item_full)
        return x

    def negative_sampling(self, items):
        zr_all, pm_all = [], []
        for i in range(items.shape[0]):
            where_zeros = torch.where(items[i] == 0)[0].tolist()
            zr_pos = random.sample(where_zeros, round(len(where_zeros) * self.s_zr))
            zr = torch.zeros_like(items[i])
            zr[zr_pos] = 1
            zr_all.append(zr)

            pm_pos = random.sample(where_zeros, round(len(where_zeros) * self.s_pm))
            pm = torch.zeros_like(items[i])
            pm[pm_pos] = 1
            pm_all.append(pm)

        return torch.stack(zr_all, dim=0), torch.stack(pm_all, dim=0)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # access your optimizers with use_pl_optimizer=False. Default is True
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)

        items, idx = batch
        zr, k = self.negative_sampling(items)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        # discriminator loss is the average of these
        if self.step_gd % (self.g_steps + self.d_steps) < self.d_steps:
            d_loss = -torch.mean(torch.log(10e-5 + self.discriminator(items, items)) +
                                 torch.log(1 + 10e-5 - self.discriminator(self.generator(items) * (items + k), items)))

            self.log('d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.manual_backward(d_loss, opt_d, retain_graph=True)
            opt_d.step()
            opt_d.zero_grad()

        # train generator
        else:
            # adversarial loss is binary cross-entropy
            generator_output = self.generator(items)
            g_loss = torch.mean(torch.log(10e-5 + 1 - self.discriminator(generator_output * (items + k), items)))
            if self.alpha != 0:
                g_loss += self.alpha * torch.sum(((items - generator_output) ** 2) * zr) / zr.sum()
            self.log('g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.manual_backward(g_loss, opt_g, retain_graph=True)
            opt_g.step()
            opt_g.zero_grad()
        self.step_gd += 1

    def validation_step(self, batch, batch_idx):
        items, idx = batch
        train_items = self.trainset[idx][0].to(items.device)
        generator_output = self.generator(train_items)
        generator_output[torch.where(train_items == 1)] = -float('inf')

        precision_at_5 = CFWGAN.precision_at_n(generator_output, items, n=5)
        self.log('precision_at_5', precision_at_5, prog_bar=True, on_step=False, on_epoch=True)
        if self.debug:
            self._info_debug = CFWGAN.precision_at_n(generator_output, items, n=2)

    def test_step(self, batch, batch_idx):
        items, idx = batch
        train_items = self.trainset[idx][0].to(items.device)
        generator_output = self.generator(train_items)
        generator_output[torch.where(train_items == 1)] = -float('inf')

        precision_at_5 = CFWGAN.precision_at_n(generator_output, items, n=5)
        self.log('precision_at_5_test', precision_at_5, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
        return [opt_g, opt_d], []

    @staticmethod
    def precision_at_n(items_predicted, items, n=5):
        div = items.sum(-1)
        div, _ = torch.stack([div, torch.ones_like(div) * n]).min(0)
        where = torch.where(div > 0)
        items, items_predicted, div = items[where], items_predicted[where], div[where]
        items_rank = torch.argsort(items_predicted, dim=-1, descending=True)
        items_rank = items_rank[:, :n]
        precision = items[
            torch.repeat_interleave(torch.arange(items.shape[0]), n).to(items_rank.device).view(*items_rank.shape),
            items_rank].float()
        precision = (precision.sum(-1) / div).mean()
        return precision

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
