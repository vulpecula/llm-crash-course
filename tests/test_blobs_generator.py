import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from perceptron.blobs_generator import gaussian_mixture, gaussian_points_rotated


class TestGaussianPointsRotated(unittest.TestCase):
    def test_shape_and_center(self):
        pts = gaussian_points_rotated(
            center=(2.0, -1.0),
            sigma=(1.0, 0.5),
            angle=0.0,
            n=5000,
            seed=123,
        )
        self.assertEqual(pts.shape, (5000, 2))
        mean = pts.mean(axis=0)
        self.assertTrue(np.allclose(mean, [2.0, -1.0], atol=0.05))

    def test_rotation_swaps_variances(self):
        pts0 = gaussian_points_rotated(
            center=(0.0, 0.0),
            sigma=(2.0, 0.5),
            angle=0.0,
            n=4000,
            seed=42,
        )
        pts90 = gaussian_points_rotated(
            center=(0.0, 0.0),
            sigma=(2.0, 0.5),
            angle=np.pi / 2,
            n=4000,
            seed=42,
        )
        var0 = pts0.var(axis=0)
        var90 = pts90.var(axis=0)
        self.assertTrue(np.allclose(var0[::-1], var90, rtol=0.1, atol=0.1))


class TestGaussianMixture(unittest.TestCase):
    def test_shapes_and_labels(self):
        components = [
            {"center": (0.0, 0.0), "sigma": 0.2},
            {"center": (5.0, 5.0), "cov": [[0.1, 0.0], [0.0, 0.1]]},
        ]
        X, y = gaussian_mixture(components, n_per=1000, seed=7)
        self.assertEqual(X.shape, (2000, 2))
        self.assertEqual(y.shape, (2000,))
        self.assertEqual(np.bincount(y).tolist(), [1000, 1000])

        mean0 = X[y == 0].mean(axis=0)
        mean1 = X[y == 1].mean(axis=0)
        self.assertTrue(np.allclose(mean0, [0.0, 0.0], atol=0.05))
        self.assertTrue(np.allclose(mean1, [5.0, 5.0], atol=0.05))


if __name__ == "__main__":
    unittest.main()
