import unittest
from dataset import MovieLensDataset
import scipy.sparse as sparse
import numpy as np
import torch

class MyTestCase(unittest.TestCase):
    def test_MovieLensDataset(self):
        dataset = MovieLensDataset(ratings_file='test_ratings.csv', movies_file='test_movies.csv')
        row  = [0,0,1,2,3,3]
        col  = [4,3,4,2,1,0]
        data = [1.,1.,1.,1.,1.,1.]

        # test for matrix equality
        expected_mat = sparse.csr_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1))
        self.assertTrue(np.array_equal(dataset.matrix.todense(), expected_mat.todense()))

        # test for first item retrieval
        expected_retrieval = torch.tensor([0., 0., 0., 1., 1.,])
        self.assertTrue(torch.equal(dataset[0], expected_retrieval))

        # test for last item retrieval
        expected_retrieval = torch.tensor([1., 1., 0., 0., 0.,])
        self.assertTrue(torch.equal(dataset[-1], expected_retrieval))

        # test for movie retrieval
        expected_string = "#1: Toy Story (1995) ===> Adventure|Animation|Children|Comedy|Fantasy"
        self.assertTrue(dataset.get_movie(4) == expected_string)

        # test for movie list retrieval
        expected_str_list = "#1: Toy Story (1995) ===> Adventure|Animation|Children|Comedy|Fantasy\n" + \
            "#2: Jumanji (1995) ===> Adventure|Children|Fantasy"
        self.assertTrue(dataset.get_movie_list_str([4,3]) == expected_str_list)

if __name__ == '__main__':
    unittest.main()
