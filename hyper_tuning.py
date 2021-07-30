import skopt.utils
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import random_split, DataLoader

from model_cfwgan import CFWGAN, MLPRepeat
from dataset2 import MovieLensDataset
import torch
import pytorch_lightning as pl
from skopt import gp_minimize
from skopt.space import Real, Integer
import pandas as pd

pl.seed_everything(12323)

batch_size = 32
config = 'movielens-100k'

dataset = MovieLensDataset('movielens/ml-100k/ratings.csv', item_based=False, save_le=False)
print(dataset.matrix.shape)
train, test = dataset.split_train_test(test_size=0.2)
train, val = train.split_train_test(test_size=0.2)

dimensions = [
    Real(name='alpha', low=0, high=5),
    Real(name='s_zr', low=0, high=1),
    Real(name='s_pm', low=0, high=1),
    Real(name='lambd', low=5, high=20),
    Real(name='lr_g', low=0.00001, high=0.1, prior='log-uniform'),
    Real(name='lr_d', low=0.00001, high=0.1, prior='log-uniform'),
    Real(name='weight_decay_g', low=10 ** -8, high=0.1, prior='log-uniform'),
    Real(name='weight_decay_d', low=10 ** -8, high=0.1, prior='log-uniform'),
    Integer(name='hidden_size_g', low=10, high=1024),
    Integer(name='hidden_size_d', low=10, high=1024),
    Integer(name='hidden_layers_g', low=1, high=5),
    Integer(name='hidden_layers_d', low=1, high=5),
]


path_save_results = 'tuning_results.csv'
list_results = []
columns = ['alpha', 's_zr', 's_pm', 'lambd', 'lr_g', 'lr_d', 'weight_decay_g', 'weight_decay_d',
          'hidden_size_g', 'hidden_layers_g', 'hidden_size_d', 'hidden_layers_d', 'score']
checkpoint_callback = skopt.callbacks.CheckpointSaver("checkpoint_tuning.pkl")
@skopt.utils.use_named_args(dimensions=dimensions)
def to_minimize(alpha, s_zr, s_pm, lambd, lr_g, lr_d, weight_decay_g, weight_decay_d,
                hidden_size_g, hidden_layers_g, hidden_size_d, hidden_layers_d):
    generator = torch.nn.Sequential(MLPRepeat(dataset.item_count, dataset.item_count, hidden_size_g, hidden_layers_g),
                                    torch.nn.Sigmoid())
    discriminator = MLPRepeat(2 * dataset.item_count, 1, hidden_size_d, hidden_layers_d)
    model = CFWGAN(train, val, None, dataset.item_count, alpha=alpha, s_zr=s_zr, s_pm=s_pm, d_steps=5, g_steps=1,
                   config=config, custom_mlps=(generator, discriminator), lambd=lambd, lrs=(lr_g, lr_d),
                   weight_decays=(weight_decay_g, weight_decay_d))

    model_checkpoint = ModelCheckpoint(monitor='ndcg_at_5', save_top_k=1, save_weights_only=True, mode='max',
                                       filename='model-{step}-{ndcg_at_5:.4f}')
    early_stop_callback = EarlyStopping(monitor="ndcg_at_5", min_delta=0.001, patience=50, verbose=False, mode="max")

    trainer = pl.Trainer(max_epochs=1000, callbacks=[model_checkpoint, early_stop_callback], log_every_n_steps=5,
                         )
    trainer.fit(model, DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size * 2))
    list_results.append((alpha, s_zr, s_pm, lambd, lr_g, lr_d, weight_decay_g, weight_decay_d,
          hidden_size_g, hidden_layers_g, hidden_size_d, hidden_layers_d, model_checkpoint.best_model_score.item()))
    df = pd.DataFrame(data=list_results, columns=columns)
    df.to_csv(path_save_results, index=False)
    return -model_checkpoint.best_model_score.item()


res = gp_minimize(to_minimize, dimensions, n_calls=30,
                  n_initial_points=5,
                  n_points=10000,
                  n_jobs=1,
                  # noise = 'gaussian',
                  noise=1e-5,
                  acq_func='gp_hedge',
                  acq_optimizer='auto',
                  random_state=None,
                  verbose=True,
                  n_restarts_optimizer=10,
                  xi=0.01,
                  kappa=1.96,
                  x0=None,
                  y0=None,
                  callback=[checkpoint_callback])

print(res)