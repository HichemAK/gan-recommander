import pytorch_lightning as pl
from torch import nn
import math
import torch

class Generator(nn.Module):
    def __init__(self, user_emb, item_emb, noise_size, history_summarizer):
        super().__init__()
        self.user_emb : nn.Embedding = user_emb
        self.item_emb : nn.Embedding = item_emb
        self.noise_size = noise_size
        size_l1 = self.user_emb.embedding_dim + self.item_emb.embedding_dim + self.noise_size
        size_l2 = int(math.pow(2, round(math.log2((size_l1 + self.item_emb.embedding_dim) / 2))))

        self.sequential = nn.Sequential(nn.Linear(size_l1, size_l2), nn.ReLU(), nn.Linear(size_l2, size_l2), nn.ReLU(),
                                        nn.Linear(size_l2, item_emb.embedding_dim))
        self.history_summarizer = history_summarizer

    def forward(self, user_id, history, noise):
        user_emb, item_emb = self.user_emb(user_id), self.item_emb(history)
        summary = self.history_summarizer(item_emb)
        x = torch.cat([user_emb, summary, noise], dim=-1)
        x = self.sequential(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, user_emb, item_emb, history_summarizer):
        super().__init__()
        self.history_summarizer = history_summarizer
        self.user_emb: nn.Embedding = user_emb
        self.item_emb: nn.Embedding = item_emb
        size_l1 = self.user_emb.embedding_dim + 2*self.item_emb.embedding_dim
        size_l2 = int(math.pow(2, round(math.log2((size_l1 + self.item_emb.embedding_dim) / 2))))
        self.sequential = nn.Sequential(nn.Linear(size_l1, size_l2), nn.ReLU(), nn.Linear(size_l2, size_l2//2), nn.ReLU(),
                                        nn.Linear(size_l2//2, 1))

    def forward(self, user_id, generator_output, history):
        user_emb, item_emb = self.user_emb(user_id), self.item_emb(history)
        summary = self.history_summarizer(item_emb)
        x = torch.cat([user_emb, summary, generator_output], dim=-1)
        x = self.sequential(x)
        return x

class HistorySummarizer(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, embedding_size)

    def forward(self, history):
        x = self.lstm(history)
        x = self.linear(x)
        return x


class Model(pl.LightningModule):
    def __init__(self, user_emb_shape, item_emb_shape, noise_size):
        super().__init__()
        self.user_emb = nn.Embedding(*user_emb_shape)
        self.item_emb = nn.Embedding(*item_emb_shape)
        self.history_summarizer = HistorySummarizer(item_emb_shape[-1], item_emb_shape[-1], 2)
        self.generator = Generator(self.user_emb, self.item_emb, noise_size, self.history_summarizer)
        self.discriminator = Discriminator(self.user_emb, self.item_emb, self.history_summarizer)

