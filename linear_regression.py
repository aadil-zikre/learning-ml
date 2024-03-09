import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, convergence_delta=1e-6, inspect=False, regularizer='none', alpha=0.1, scale_features=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.convergence_delta = convergence_delta
        self.weights = None
        self.bias = None
        self.inspect = inspect
        self.regularizer = regularizer
        self.alpha = alpha
        self.scale_features = scale_features

    def fit(self, X, y):
        if self.inspect:
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

        if self.scale_features:
            X = self._standardize(X)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        prev_cost = np.inf  # Initialize previous cost to infinity

        # Gradient descent
        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute cost (mean squared error)
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)

            # Add regularization term
            if self.regularizer == 'l1':
                cost += self.alpha * np.sum(np.abs(self.weights))
            elif self.regularizer == 'l2':
                cost += self.alpha * np.sum(self.weights ** 2)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Add regularization term to gradients
            if self.regularizer == 'l1':
                dw += self.alpha * np.sign(self.weights)
            elif self.regularizer == 'l2':
                dw += 2 * self.alpha * self.weights

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check for convergence
            if abs(prev_cost - cost) < self.convergence_delta:
                if self.inspect:
                    print(f"Converged after {i+1} iterations")
                break

            prev_cost = cost

    def predict(self, X):
        if self.inspect:
            print(f"X shape: {X.shape}")

        if self.scale_features:
            X = self._standardize(X)

        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    def fit_ols(self, X, y):
        if self.inspect:
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

        if self.scale_features:
            X = self._standardize(X)

        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add a column of ones for the bias term
        if self.inspect:
            print(f"X shape after adding bias column: {X.shape}")

        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y  # Ordinary Least Squares solution

    def predict_ols(self, X):
        if self.inspect:
            print(f"X shape: {X.shape}")

        if self.scale_features:
            X = self._standardize(X)

        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add a column of ones for the bias term
        if self.inspect:
            print(f"X shape after adding bias column: {X.shape}")

        y_predicted = X @ self.weights
        return y_predicted

    def _standardize(self, X):
        """Standardize features by removing the mean and scaling to unit variance"""
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        X_scaled = (X - feature_means) / feature_stds
        return X_scaled