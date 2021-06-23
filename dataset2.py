import copy
import random

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import json


class MovieLensDataset(Dataset):
    def __init__(self, path, item_based=False, save_le=False):
        with open(path, 'r') as file:
            s = file.readline()
        sep = ','
        names = None
        if '::' in s:
            sep = '::'
            names = ['userId', 'movieId', 'rating', 'timestamp']
        df = pd.read_csv(path, sep=sep, names=names)
        self.movie_le = LabelEncoder()
        self.user_le = LabelEncoder()
        df['userId'] = self.user_le.fit_transform(df['userId'])
        df['movieId'] = self.movie_le.fit_transform(df['movieId'])
        if save_le:
            user_d = {i : x.item() for i, x in enumerate(self.user_le.classes_)}
            movie_d = {i: x.item() for i, x in enumerate(self.movie_le.classes_)}
            with open('user_dict.json', 'w') as f:
                f.write(json.dumps(user_d))
            with open('movie_dict.json', 'w') as f:
                f.write(json.dumps(movie_d))

        self.dataframe = df

        row, column, data = df['userId'], df['movieId'], np.ones(len(df))
        self.matrix = scipy.sparse.csr_matrix((data, (row, column)))
        if item_based:
            self.matrix = self.matrix.transpose()

        self.item_count = self.matrix.shape[-1]

    def __getitem__(self, idx):
        data = self.matrix[idx]
        data = torch.tensor(data.toarray().squeeze()).float()
        return data, idx

    def __len__(self):
        return self.matrix.shape[0]

    def split_train_test(self, test_size=0.2):
        train_matrix = self.matrix.copy()
        nz = train_matrix.nonzero()
        nz = list(zip(*[x.tolist() for x in nz]))
        test = random.sample(nz, round(len(nz) * test_size))
        test = tuple(np.array(x) for x in zip(*test))
        test_matrix = sparse.csr_matrix(train_matrix.shape)
        test_matrix[test] = 1
        train_matrix[test] = 0

        train = copy.deepcopy(self)
        train.matrix = train_matrix

        test = copy.deepcopy(self)
        test.matrix = test_matrix

        return train, test