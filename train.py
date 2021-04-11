from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader

from model_cfwgan import CFWGAN
from dataset import MovieLensDataset
import torch
import pytorch_lightning as pl


pl.seed_everything(12323)

batch_size = 16

dataset = MovieLensDataset('movielens/ml-100k/ratings.csv', 'movielens/ml-100k/movies.csv')
train, test = dataset.split_train_test(test_size=0.2)

test_size = int(len(test)*0.5)

test, val = random_split(dataset, [test_size, len(test)-test_size], generator=torch.Generator().manual_seed(1234))

model = CFWGAN(train, dataset.item_count, alpha=0.1, s_zr=0.7, s_pm=0.7, d_steps=4)

model_checkpoint = ModelCheckpoint(monitor='precision_at_5', save_top_k=5, save_weights_only=True, mode='max',
                                   filename='model-{epoch}-{precision_at_5:.2f}')

trainer = pl.Trainer(max_epochs=100, callbacks=[model_checkpoint])
trainer.fit(model, DataLoader(train, batch_size, shuffle=True), DataLoader(val, batch_size*2))
model = CFWGAN.load_from_checkpoint(model_checkpoint.best_model_path, trainset=train, num_items=dataset.item_count)
trainer.test(model, DataLoader(test, batch_size*2))

