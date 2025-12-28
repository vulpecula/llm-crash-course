import numpy as np
from typing import Tuple


class Perceptron:
    def __init__(self, n_features: int, learning_rate: float = 0.1):
        """
        Initialize perceptron parameters.

        Args:
            n_features: Number of input features
            learning_rate: Step size for weight updates
        """
        self.n_features = n_features
        self.learning_rate = learning_rate

        # Initialize weights and bias at random
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

    def forward(self, x: np.ndarray) -> float:
        """
        Compute the raw score (before activation).

        Args:
            x: Input vector of shape (n_features,)

        Returns:
            Raw linear output (float)
        """
        return np.dot(x, self.weights) + self.bias

    def activation(self, z: float) -> int:
        """
        Apply step activation function.

        Args:
            z: Raw score

        Returns:
            Predicted class label (e.g. -1 or +1)
        """
        return 1 if z > 0 else -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for a batch of inputs.

        Args:
            X: Input matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)
        """
        z = X @ self.weights + self.bias
        return np.where(z > 0, 1, -1)

    def loss(self, y_true: int, y_pred: int) -> float:
        """
        Compute perceptron loss for a single example.

        Args:
            y_true: Ground truth label
            y_pred: Predicted label

        Returns:
            Loss value
        """
        return 0.5 * (y_true - y_pred) ** 2


    def update(self, x: np.ndarray, y_true: int, y_pred: int) -> None:
        """
        Update weights and bias using perceptron rule.

        Args:
            x: Input vector
            y_true: True label
            y_pred: Predicted label
        """
        self.weights += self.learning_rate * (y_true - y_pred) * x
        self.bias += self.learning_rate * (y_true - y_pred)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10
    ) -> None:
        """
        Train the perceptron.

        Args:
            X: Training data (n_samples, n_features)
            y: Labels (n_samples,)
            epochs: Number of passes over the dataset
        """
        for _ in range(epochs):
            for x, y_true in zip(X, y):
                y_pred = self.activation(self.forward(x))
                self.update(x, y_true, y_pred)

