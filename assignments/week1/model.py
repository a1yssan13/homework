import numpy as np


class LinearRegression:
    """
    Basic implementation of linear regression using np
    """

    w: np.ndarray
    b: float

    def __init__(self):
        """
        Initializes weight w and bias b as None.
        """
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear regression model on the given data.

        Arguments:
            X (np.ndarray) : shape (n_samples, n_features)
                The feature matrix of the training data.
            y (np.ndarray) : shape (n_samples, )
                The target variable of the training data.

        Returns:
            None
        """
        X = np.hstack((X, np.ones((X.shape[0], 1))))  # includes a bias layer
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        print(self.w)
        self.b = self.w[-1]
        self.w = self.w[:-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """

        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X @ np.hstack((self.w, self.b))


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fits the linear regression model on the given data using Gradient Descent.

        Arguments:
            X (np.ndarray) : shape (n_samples, n_features)
                The feature matrix of the training data.
            y (np.ndarray) : shape (n_samples, )
                The target variable of the training data.

        Returns:
            None: no output
        """
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features)
        for _ in range(epochs):
            y_pred = X @ self.w
            error = y_pred - y
            gradient = X.T @ error / n_samples
            self.w -= lr * gradient
        self.b = self.w[-1]
        self.w = self.w[:-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X @ np.hstack((self.w, self.b))
