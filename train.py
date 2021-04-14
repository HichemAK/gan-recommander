from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader

from model_cfwgan import CFWGAN
from dataset2 import MovieLensDataset
import torch
import pytorch_lightning as pl


pl.seed_everything(12323)

batch_size = 16

dataset = MovieLensDataset('movielens/ml-100k/ratings.csv')
train, test = dataset.split_train_test(test_size=0.2)

model = CFWGAN(train, dataset.item_count, alpha=0, s_zr=0, s_pm=0.2, d_steps=3, g_steps=5)

model_checkpoint = ModelCheckpoint(monitor='precision_at_5', save_top_k=5, save_weights_only=True, mode='max',
                                   filename='model-{epoch}-{precision_at_5:.4f}')

trainer = pl.Trainer(max_epochs=100, callbacks=[model_checkpoint])
trainer.fit(model, DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size*2))
model = CFWGAN.load_from_checkpoint(model_checkpoint.best_model_path, trainset=train, num_items=dataset.item_count)
trainer.test(model, DataLoader(test, batch_size*2))

