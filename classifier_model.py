import itertools
import math
import random
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, num_items, p=0.8, config='movielens-100k'):
        super().__init__()
        n = 256 if config == 'movielens-100k' else 512
        self.mlp_tower = nn.Sequential(
            nn.Linear(num_items, n),
            nn.Dropout(p),
            nn.ReLU(True),
            nn.Linear(n, num_items),
        )

    def forward(self, item_full):
        return self.mlp_tower(item_full)


class Model(pl.LightningModule):
    def __init__(self, trainset, num_items):
        super().__init__()
        self.classifier = Classifier(num_items)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.trainset = trainset
        self.automatic_optimization = False

    def forward(self, item_full):
        x = self.classifier(item_full)
        return x

    def training_step(self, batch, batch_idx):
        # access your optimizers with use_pl_optimizer=False. Default is True
        opt = self.optimizers(use_pl_optimizer=True)

        items, idx = batch

        output = self.classifier(items)
        loss = self.criterion(output, items)
        self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('output_mean', output.mean(), prog_bar=False, on_step=True, on_epoch=False)
        opt.zero_grad()
        self.manual_backward(loss, opt, retain_graph=True)
        opt.step()

    def validation_step(self, batch, batch_idx):
        items, idx = batch
        train_items = self.trainset[idx.cpu()][0].to(items.device)
        output = self.classifier(train_items)
        output[torch.where(train_items == 1)] = -float('inf')
        precision_at_5 = Model.precision_at_n(output, items, n=5)
        recall_at_5 = Model.recall_at_n(output, items, n=5)
        ndcg_at_5 = Model.ndcg(output, items, n=5)
        self.log('precision_at_5', precision_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_at_5', recall_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ndcg_at_5', ndcg_at_5, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        items, idx = batch
        train_items = self.trainset[idx.cpu()][0].to(items.device)
        output = self.classifier(train_items)
        output[torch.where(train_items == 1)] = -float('inf')

        precision_at_5 = Model.precision_at_n(output, items, n=5)
        recall_at_5 = Model.recall_at_n(output, items, n=5)
        ndcg_at_5 = Model.ndcg(output, items, n=5)
        self.log('precision_at_5', precision_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_at_5', recall_at_5, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ndcg_at_5', ndcg_at_5, prog_bar=True, on_step=False, on_epoch=True)

        precision_at_20 = Model.precision_at_n(output, items, n=20)
        recall_at_20 = Model.recall_at_n(output, items, n=20)
        ndcg_at_20 = Model.ndcg(output, items, n=20)
        self.log('precision_at_20', precision_at_20, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_at_20', recall_at_20, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ndcg_at_20', ndcg_at_20, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=0.00001)
        return opt

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
