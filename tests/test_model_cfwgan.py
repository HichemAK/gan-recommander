import unittest

import numpy as np
import torch
from torch.utils.data import random_split

from dataset import MovieLensDataset
from model_cfwgan import MLPTower, MLPRepeat, Generator, Discriminator, CFWGAN
import pytorch_lightning as pl


class MyTestCase(unittest.TestCase):
    def test_mlpTower(self):
        model = MLPTower(16, 2, 3)
        out = model(torch.tensor(np.random.rand(5, 16), dtype=torch.float))
        self.assertEqual(out.shape, (5, 2))
        l = [(x.in_features, x.out_features) for x in model.sequential if isinstance(x, torch.nn.Linear)]
        self.assertEqual(l, [(16, 8), (8, 4), (4, 2), (2, 2)])

    def test_repeatMLP(self):
        model = MLPRepeat(16, 8, 12, 3)
        out = model(torch.tensor(np.random.rand(5, 16), dtype=torch.float))
        self.assertEqual(out.shape, (5, 8))
        l = [(x.in_features, x.out_features) for x in model.sequential if isinstance(x, torch.nn.Linear)]
        self.assertEqual(l, [(16, 12), (12, 12), (12, 12), (12, 8)])

    def test_generator(self):
        model = Generator(50, 60, 3)
        x = torch.tensor(np.random.rand(8, 50)).float()
        out = model(x)
        self.assertEqual(out.shape, (8, 50))

    def test_discriminator(self):
        model = Discriminator(50, 3)
        x = torch.tensor(np.random.rand(8, 50)).float()
        out = model(x, x)
        self.assertEqual(out.shape, (8, 1))

    def test_precision_at_n(self):
        items = torch.tensor([[0, 1, 0, 1, 0, 1],
                              [0, 0, 0, 1, 1, 1],
                              [1, 1, 0, 0, 1, 1]])
        items_predicted = torch.tensor([[1.2, -0.1, 2, 1.1, 1.8, 2],
                                        [0.26514016, 0.25176894, 0.41136022, 0.39306909, 0.13250113, 0.84741624],
                                        [0.14425929, 0.2018705, 0.15223548, 0.73594551, 0.76860745, 0.70887101]])
        precision = CFWGAN.precision_at_n(items_predicted, items, n=2)
        self.assertAlmostEqual(precision, (1 / 2 + 1 / 2 + 1 / 2) / 3)
        precision = CFWGAN.precision_at_n(items_predicted, items, n=3)
        self.assertAlmostEqual(precision, (1 / 3 + 2 / 3 + 2 / 3) / 3)

    def test_precision_at_n_few_pos(self):
        items = torch.tensor([[0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 1],
                              [0, 0, 1, 0, 1, 1]])
        items_predicted = torch.tensor([[1.2, 3, 2, 1.1, 1.8, 2.1],
                                        [0.26514016, 0.25176894, 0.41136022, 0.39306909, 0.13250113, 0.84741624],
                                        [0.14425929, 0.2018705, 0.15223548, 0.73594551, 0.76860745, 0.70887101]])
        precision = CFWGAN.precision_at_n(items_predicted, items, n=3)
        self.assertAlmostEqual(precision.item(), (2 / 2 + 2 / 2 + 2 / 3) / 3)

        items = torch.tensor([[0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 1, 1]])
        items_predicted = torch.tensor([[1.2, 3, 2, 1.1, 1.8, 2.1],
                                        [0.26514016, 0.25176894, 0.41136022, 0.39306909, 0.13250113, 0.84741624],
                                        [0.14425929, 0.2018705, 0.15223548, 0.73594551, 0.76860745, 0.70887101]])
        precision = CFWGAN.precision_at_n(items_predicted, items, n=3)
        self.assertAlmostEqual(precision.item(), (1 / 1 + 2 / 3) / 2)

    def test_validation_step(self):
        pl.seed_everything(123443)

        dataset = MovieLensDataset('test_ratings.csv', 'test_movies.csv')
        train, test = dataset.split_train_test(test_size=0.4)
        model = CFWGAN(train, dataset.item_count, alpha=0.1, s_zr=0.7, s_pm=0.7, debug=True)
        """train = [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0]]
        test = '[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0, 0.0, 0.0]]'
        """
        generator_output = [[0.05486110970377922, 0.01412893459200859, -0.016457993537187576, -0.07126838713884354,
                             -0.02538265474140644],
                            [0.039036281406879425, 0.010008297860622406, -0.02964741736650467, -0.02559179812669754,
                             -0.003673775587230921],
                            [0.03861333057284355, 0.04027758166193962, -0.046206556260585785, -0.014589466154575348,
                             -0.022400809451937675],
                            [0.01941085048019886, 0.027754511684179306, -0.03860647976398468, -0.046702973544597626,
                             -0.010901669971644878]]
        assert model.forward(train[:4][0]).tolist() == generator_output
        model.validation_step(test[:4], 0)
        self.assertAlmostEqual(model._info_debug, (0/1 + 1/1)/2)

    def test_negative_sampling(self):
        class Test:
            def __init__(self, s_zr=0.6, s_pm=0.6):
                self.s_zr = s_zr
                self.s_pm = s_pm
                self.negative_sampling = CFWGAN.negative_sampling

        test = Test(s_zr=0.6, s_pm=0.6)
        items = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        zr, pm = test.negative_sampling(test, items)
        for i in range(items.shape[0]):
            self.assertEqual(zr[i].sum(), 3)
            self.assertEqual(pm[i].sum(), 3)
        t = zr + items
        self.assertTrue(((t == 0) | (t == 1)).all())

        t = pm + items
        self.assertTrue(((t == 0) | (t == 1)).all())

        test.s_pm = 0
        test.s_zr = 0

        zr, pm = test.negative_sampling(test, items)
        for i in range(items.shape[0]):
            self.assertEqual(zr[i].sum(), 0)
            self.assertEqual(pm[i].sum(), 0)

        test.s_pm = 1
        test.s_zr = 1

        zr, pm = test.negative_sampling(test, items)
        for i in range(items.shape[0]):
            self.assertEqual(zr[i].sum(), 5)
            self.assertEqual(pm[i].sum(), 5)


if __name__ == '__main__':
    unittest.main()
