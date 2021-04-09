import torch
import unittest

from recommender import Recommender


class MyTestCase(unittest.TestCase):
    def test_Recommender(self):
        recommender = Recommender(path_to_model="", ratings_file="test_ratings.csv", movies_file="test_movies.csv")

        # test vector_to_movies
        expected_str_list = "#1: Toy Story (1995) ===> Adventure|Animation|Children|Comedy|Fantasy\n" + \
                            "#2: Jumanji (1995) ===> Adventure|Children|Fantasy"
        self.assertTrue(recommender.vector_to_movies([0, 0, 0, 1, 1]) == expected_str_list)

        # test filter_vector
        input_array    = torch.tensor([1,0,1,1,0,0,0,1])
        output_array   = torch.tensor([1,0,0,1,1,1,0,0])
        expected_array = torch.tensor([1,0,0,1,0,0,0,0])
        self.assertTrue(expected_array.equal(recommender.filter_vector(input_array, output_array)))

        # test top_k
        input_vector = torch.tensor([0.9, 0.2, 0, 0.3, 0.8])
        top_k = 2
        expected_result = "#7: Father of the Bride Part II (1995) ===> Comedy\n" + \
            "#1: Toy Story (1995) ===> Adventure|Animation|Children|Comedy|Fantasy" + "\n\n" + \
            "probs: [0.9, 0.8]"
        self.assertTrue(recommender.top_k(input_vector, top_k) == expected_result)

if __name__ == '__main__':
    unittest.main()
