import torch

from dataset import MovieLensDataset
from model_cfwgan import CFWGAN

class Recommender():
    def __init__(self, path_to_model=None, ratings_file=None, movies_file=None):
        self.model = Recommender.load_model(path_to_model)
        self.dataset = MovieLensDataset(ratings_file=ratings_file, movies_file=movies_file)

    @staticmethod
    def load_model(path):
        if path is None or path == '':
            return None
        model = CFWGAN.load_from_checkpoint(path)
        return model

    def generate(self, vector):
        return self.model.forward(torch.tensor(vector))

    def vector_to_movies(self, vector):
        filtered = [i*vector[i] for i in range(len(vector)) if vector[i] != 0]
        filtered.reverse()
        return self.dataset.get_movie_list_str(filtered)
