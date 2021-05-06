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
        self.mlp_repeat = nn.Sequential(
            nn.Linear(num_items, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_items),
            nn.Sigmoid()
        )

    def forward(self, items):
        return self.mlp_repeat(items)


class Discriminator(nn.Module):
    def __init__(self, num_items, num_hidden_layers):
        super().__init__()
        self.mlp_tower = nn.Sequential(
            nn.Linear(2 * num_items, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
        )

    def forward(self, generator_output, item_full):
        x = torch.cat([generator_output, item_full], dim=-1)
        return self.mlp_tower(x)


class CFWGAN(pl.LightningModule):
    def __init__(self, trainset, num_items, alpha=0.04, s_zr=0.6, s_pm=0.6, g_steps=1, d_steps=1, lambd=10,
                 debug=False):
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
        self.lambd = lambd
        self.automatic_optimization = False

    def forward(self, item_full):
        x = self.generator(item_full)
        return x

    def negative_sampling(self, items):
        zr_all, pm_all = [], []
        for i in range(items.shape[0]):
            where_zeros = torch.where(items[i] == 0)[0].tolist()
            n = round(len(where_zeros) * self.s_zr) if isinstance(self.s_zr, float) else self.s_zr
            zr_pos = random.sample(where_zeros, n)
            zr = torch.zeros_like(items[i])
            zr[zr_pos] = 1
            zr_all.append(zr)

            n = round(len(where_zeros) * self.s_pm) if isinstance(self.s_pm, float) else self.s_pm
            pm_pos = random.sample(where_zeros, n)
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
        if self.step_gd % (self.g_steps + self.d_steps) >= self.g_steps:
            fake_data = self.generator(items)
            epsilon = torch.rand(items.shape[0], 1, device=items.device)
            x_hat = epsilon * fake_data + (1 - epsilon) * items
            d_hat = self.discriminator(x_hat, items)
            gradients = torch.autograd.grad(outputs=d_hat, inputs=x_hat,
                                            grad_outputs=torch.ones_like(d_hat),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients_norm = gradients.norm(2, dim=-1)
            d_loss = torch.mean(self.discriminator(fake_data * items, items) - self.discriminator(items, items)
                                + self.lambd * (gradients_norm - 1) ** 2)

            self.log('d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=False)
            self.log('gradients_norm', gradients_norm.mean(), prog_bar=False, on_step=True, on_epoch=False)
            opt_d.zero_grad()
            self.manual_backward(d_loss, opt_d, retain_graph=True)
            opt_d.step()

        # train generator
        else:
            # adversarial loss is binary cross-entropy
            generator_output = self.generator(items)
            g_loss = torch.mean(-self.discriminator(generator_output * items, items))
            if self.alpha != 0:
                g_loss += self.alpha * torch.sum(((items - generator_output) ** 2) * zr) / items.shape[0]
            self.log('g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=False)
            self.log('output_mean', generator_output.mean(), prog_bar=False, on_step=True, on_epoch=False)
            opt_g.zero_grad()
            self.manual_backward(g_loss, opt_g, retain_graph=True)
            opt_g.step()
        self.step_gd += 1

    def validation_step(self, batch, batch_idx):
        items, idx = batch
        train_items = self.trainset[idx.cpu()][0].to(items.device)
        generator_output = self.generator(train_items)
        generator_output[torch.where(train_items == 1)] = -float('inf')
        precision_at_5 = CFWGAN.precision_at_n(generator_output, items, n=5)
        recall_at_5 = CFWGAN.recall_at_n(generator_output, items, n=5)
        ndcg_at_5 = CFWGAN.ndcg(generator_output, items, n=5)
        self.log('precision_at_5', precision_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_at_5', recall_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ndcg_at_5', ndcg_at_5, prog_bar=True, on_step=False, on_epoch=True)
        if self.debug:
            self._info_debug = CFWGAN.precision_at_n(generator_output, items, n=2)

    def test_step(self, batch, batch_idx):
        items, idx = batch
        train_items = self.trainset[idx.cpu()][0].to(items.device)
        generator_output = self.generator(train_items)
        generator_output[torch.where(train_items == 1)] = -float('inf')

        precision_at_5 = CFWGAN.precision_at_n(generator_output, items, n=5)
        recall_at_5 = CFWGAN.recall_at_n(generator_output, items, n=5)
        ndcg_at_5 = CFWGAN.ndcg(generator_output, items, n=5)
        self.log('precision_at_5', precision_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_at_5', recall_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ndcg_at_5', ndcg_at_5, prog_bar=True, on_step=False, on_epoch=True)

        precision_at_20 = CFWGAN.precision_at_n(generator_output, items, n=20)
        recall_at_20 = CFWGAN.recall_at_n(generator_output, items, n=20)
        ndcg_at_20 = CFWGAN.ndcg(generator_output, items, n=20)
        self.log('precision_at_20', precision_at_20, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_at_20', recall_at_20, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ndcg_at_20', ndcg_at_20, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
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

    @staticmethod
    def ndcg(items_predicted, items, n=5):
        w = items.sum(-1) > 0
        items, items_predicted = items[w], items_predicted[w]
        items_rank = torch.argsort(items_predicted, dim=-1, descending=True)
        items_rank = items_rank[:, :n]
        j = torch.log2(torch.arange(start=2, end=n + 2, dtype=torch.float, device=items.device))
        dcg = (torch.gather(items, 1, items_rank) / j).sum(-1)
        perfect_rank = torch.argsort(items, dim=-1, descending=True)[:, :n]
        idcg = (torch.gather(items, 1, perfect_rank) / j).sum(-1)
        ndcg = dcg / idcg
        return ndcg.mean()

    @staticmethod
    def recall_at_n(items_predicted, items, n=5):
        w = items.sum(-1) > 0
        items, items_predicted = items[w], items_predicted[w]
        items_rank = torch.argsort(items_predicted, dim=-1, descending=True)
        items_rank = items_rank[:, :n]
        recall = torch.gather(items, 1, items_rank).sum(-1) / items.sum(-1)
        return recall.mean()

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
