import unittest
from model_cfwgan import MLPTower, MLPRepeat, Generator, Discriminator
import torch
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_mlpTower(self):
        model = MLPTower(16, 2, 3)
        out = model(torch.tensor(np.random.rand(5,16), dtype=torch.float))
        self.assertEqual(out.shape, (5, 2))
        l = [(x.in_features, x.out_features) for x in model.sequential if isinstance(x, torch.nn.Linear)]
        self.assertEqual(l, [(16, 8), (8, 4), (4,2), (2, 2)])

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


if __name__ == '__main__':
    unittest.main()
