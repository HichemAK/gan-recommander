import unittest
from dataset import MovieLensDataset
from recommender import Recommender
import scipy.sparse as sparse
import numpy as np
import torch

class MyTestCase(unittest.TestCase):
    def test_Recommender(self):
        recommender = Recommender(path_to_model="", ratings_file="test_ratings.csv", movies_file="test_movies.csv")
        expected_str_list = "#1: Toy Story (1995) ===> Adventure|Animation|Children|Comedy|Fantasy\n" + \
                            "#2: Jumanji (1995) ===> Adventure|Children|Fantasy"
        self.assertTrue(recommender.vector_to_movies([0,0,0,1,1]) == expected_str_list)

if __name__ == '__main__':
    unittest.main()
