import unittest

import numpy as np
import torch

from model_cfwgan import MLPTower, MLPRepeat, Generator, Discriminator, CFWGAN


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
        self.assertEqual(precision, (1/2 + 1/2 + 1/2)/3)
        precision = CFWGAN.precision_at_n(items_predicted, items, n=3)
        self.assertEqual(precision, (1 / 3 + 2 / 3 + 2 / 3) / 3)

    def test_negative_sampling(self):
        class Test:
            def __init__(self, s_zr=0.6, s_pm=0.6):
                self.s_zr = s_zr
                self.s_pm = s_pm
                self.negative_sampling = CFWGAN.negative_sampling
        test = Test(s_zr=0.6, s_pm=0.6)
        items = torch.tensor([[1,0,1,0,1,0,1,0,1,0],
                              [0,0,0,0,0,1,1,1,1,1]])
        zr, pm = test.negative_sampling(test, items)
        for i in range(items.shape[0]):
            self.assertEqual(zr[i].sum(), 3)
            self.assertEqual(pm[i].sum(), 3)

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
