import unittest
from dataset import MovieLensDataset

class MyTestCase(unittest.TestCase):
    def test_MovieLensDataset(self):
        dataset = MovieLensDataset('test_ratings.csv')
        print(dataset.matrix)


if __name__ == '__main__':
    unittest.main()
