from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader

from model_cfwgan import CFWGAN
from dataset2 import MovieLensDataset
import torch
import pytorch_lightning as pl

pl.seed_everything(12323)

batch_size = 32
config = 'movielens-100k'

dataset = MovieLensDataset('movielens/ml-100k/ratings.csv', item_based=False, save_le=False)
print(dataset.matrix.shape)
train, test = dataset.split_train_test(test_size=0.2)
# train, val = train.split_train_test(test_size=0.2)

model = CFWGAN(train, test, None, dataset.item_count, alpha=0.1, s_zr=0.5, s_pm=0.5, d_steps=5, g_steps=1,
               config=config)

model_checkpoint = ModelCheckpoint(monitor='ndcg_at_5', save_top_k=5, save_weights_only=True, mode='max',
                                   filename='model-{step}-{ndcg_at_5:.4f}')

trainer = pl.Trainer(max_epochs=2, callbacks=[model_checkpoint], log_every_n_steps=5,
                     )
trainer.fit(model, DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size * 2))
model = CFWGAN.load_from_checkpoint(model_checkpoint.best_model_path, trainset=train, valset=test, testset=None,
                                    num_items=dataset.item_count)
trainer.test(model, DataLoader(test, batch_size * 2))
