import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from perceptron.perceptron import Perceptron


class TestPerceptronCore(unittest.TestCase):
    def test_forward_activation_predict(self):
        model = Perceptron(n_features=2, learning_rate=0.1)
        model.weights = np.array([1.0, -2.0])
        model.bias = 0.5

        x = np.array([2.0, 1.0])
        z = model.forward(x)
        self.assertAlmostEqual(z, 0.5)
        self.assertEqual(model.activation(z), 1)

        X = np.array([[2.0, 1.0], [-1.0, 1.0]])
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, np.array([1, -1]))

    def test_loss(self):
        model = Perceptron(n_features=1)
        self.assertEqual(model.loss(1, -1), 2.0)
        self.assertEqual(model.loss(-1, -1), 0.0)

    def test_update(self):
        model = Perceptron(n_features=2, learning_rate=0.1)
        model.weights = np.array([0.0, 0.0])
        model.bias = 0.0

        x = np.array([1.0, -2.0])
        model.update(x, y_true=1, y_pred=-1)

        np.testing.assert_allclose(model.weights, np.array([0.2, -0.4]))
        self.assertAlmostEqual(model.bias, 0.2)


class TestPerceptronFit(unittest.TestCase):
    def test_fit_linearly_separable(self):
        X = np.array([
            [-2.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        y = np.array([-1, -1, 1, 1])

        model = Perceptron(n_features=2, learning_rate=0.1)
        model.weights = np.zeros(2)
        model.bias = 0.0

        model.fit(X, y, epochs=10)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)


if __name__ == "__main__":
    unittest.main()
