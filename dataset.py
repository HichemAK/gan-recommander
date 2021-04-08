import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import scipy.sparse as sparse

class MovieLensDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = MovieLensDataset.get_dataframe(csv_file)
        self.matrix = self.get_matrix()
        self.user_count = self.matrix.shape[0]
        self.item_count = self.matrix.shape[1]
        self.transform = transform

    def __len__(self):
        return self.user_count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.matrix[idx]
        sample = np.array(sample.toarray())

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def get_dataframe(csv_file):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError("csv file not found")

        df = pd.read_csv(csv_file)
        df.drop("rating", 1, inplace=True)
        df.drop("timestamp", 1, inplace=True)
        df["movieId"] = MovieLensDataset.remove_gaps(df["movieId"])

        def minusOne(x):
            return x - 1

        df["userId"] = df["userId"].apply(minusOne)
        df["movieId"] = df["movieId"].apply(minusOne)

        return df

    @staticmethod
    def remove_gaps(pd_series, flip=False):
        real_count = len(pd_series.unique())

        for i in range(1, real_count + 1):
            pd_series.replace(pd_series.max(), i*(-1), inplace=True)

        def positify(x):
            return x * (-1)

        pd_series = pd_series.apply(positify)

        if flip:
            pass

        return pd_series

    def get_matrix(self):
        item_count = self.dataframe["movieId"].max()
        user_count = self.dataframe["userId"].max()

        row = self.dataframe["userId"]
        col = self.dataframe["movieId"]
        data = np.ones((len(self.dataframe)))

        return sparse.csr_matrix((data, (row, col)), shape=(user_count + 1, item_count + 1))


if __name__ == "__main__":
    ds = MovieLensDataset("movielens/ml-latest-small/ratings.csv")
    print(ds[0])
