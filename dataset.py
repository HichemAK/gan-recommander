import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import scipy.sparse as sparse
import copy
import random

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file=None, movies_file=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.movie_dict = {}
        self.dataframe = MovieLensDataset.get_dataframe(ratings_file, movie_dict=self.movie_dict)
        self.movies_dataframe = MovieLensDataset.get_movies_dataframe(movies_file)
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
        sample = torch.tensor(sample.toarray()).float().squeeze()

        if self.transform:
            sample = self.transform(sample)

        return sample, idx

    @staticmethod
    def get_movies_dataframe(csv_file=None):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError("csv file not found")

        df = pd.read_csv(csv_file)
        return df

    @staticmethod
    def get_dataframe(csv_file, movie_dict=None):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError("csv file not found")

        df = pd.read_csv(csv_file)
        df.drop("rating", 1, inplace=True)
        df.drop("timestamp", 1, inplace=True)
        df["movieId"] = MovieLensDataset.remove_gaps(df["movieId"], movie_dict=movie_dict)

        def minusOne(x):
            return x - 1

        df["userId"] = df["userId"].apply(minusOne)
        df["movieId"] = df["movieId"].apply(minusOne)

        return df

    @staticmethod
    def remove_gaps(pd_series, movie_dict=None, flip=True):
        real_count = len(pd_series.unique())

        for i in range(1, real_count + 1):
            movie_dict[i-1] = pd_series.max()
            pd_series.replace(pd_series.max(), i*(-1), inplace=True)

        def positify(x):
            return x * (-1)

        pd_series = pd_series.apply(positify)

        if flip:
            pd_series = pd_series * (-1) + max(pd_series) + 1

            dict_max = max(list(movie_dict.values()))
            for key in movie_dict:
                val = movie_dict[key]
                new_val = (val * (-1)) + dict_max + 1
                movie_dict[key] = new_val

        return pd_series

    def get_matrix(self):
        item_count = self.dataframe["movieId"].max()
        user_count = self.dataframe["userId"].max()

        row = self.dataframe["userId"]
        col = self.dataframe["movieId"]
        data = np.ones((len(self.dataframe)))

        return sparse.csr_matrix((data, (row, col)), shape=(user_count + 1, item_count + 1))

    def get_movie(self, index):
        real_id = self.movie_dict[index]
        row = self.movies_dataframe[self.movies_dataframe["movieId"] == real_id]
        title = row["title"].values[0]
        genres = row["genres"].values[0]

        ret = f"#{real_id}: {title} ===> {genres}"

        return ret

    def get_movie_list_str(self, indexes=None):
        l = []

        for i in indexes:
            l.append(self.get_movie(i))

        return "\n".join(l)

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

        train.to_items_file("train.csv")
        test.to_items_file("test.csv")

        return train, test

    def to_items_file(self, filename="result.csv"):
        item_list = []
        array = self.matrix.toarray()
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if array[i][j] == 1:
                    item_list.append(f"{i+1}\t{j+1}\t{5}\t{1}")

        with open(filename, 'w') as file:
            file.write("\n".join(item_list))



if __name__ == "__main__":
    ds = MovieLensDataset(ratings_file="movielens/ml-100k/ratings.csv", movies_file="movielens/ml-100k/movies.csv")
