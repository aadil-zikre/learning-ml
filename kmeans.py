import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)

            # Store the previous centroids
            prev_centroids = self.centroids.copy()

            # Update centroids
            for i in range(self.n_clusters):
                self.centroids[i] = np.mean(X[self.labels == i], axis=0)

            # Check if centroids have stopped moving
            if np.all(self.centroids == prev_centroids):
                break

    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)

    def _calc_distances(self, X):
        distances = np.zeros((len(X), self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances