# K-Means Clustering

This Python class implements the K-means clustering algorithm from scratch. K-means is an unsupervised learning algorithm that aims to partition n observations into k clusters, where each observation belongs to the cluster with the nearest mean.

## Class: `KMeans`

### Initialization

```python
KMeans(n_clusters, max_iter=100)
```

- `n_clusters`: The number of clusters to form (integer).
- `max_iter`: The maximum number of iterations to perform (integer, default=100).

### Methods

#### `fit(X)`

Fits the K-means model on the input data X.

- `X`: A 2D numpy array of shape (n_samples, n_features) representing the input data.

#### `predict(X)`

Predicts the cluster labels for the input data X.

- `X`: A 2D numpy array of shape (n_samples, n_features) representing the input data.
- Returns: A 1D numpy array of shape (n_samples,) containing the predicted cluster labels for each data point.

#### `_calc_distances(X)`

Calculates the Euclidean distances between each data point in X and all centroids.

- `X`: A 2D numpy array of shape (n_samples, n_features) representing the input data.
- Returns: A 2D numpy array of shape (n_samples, n_clusters) containing the distances between each data point and each centroid.

### Attributes

- `centroids`: A 2D numpy array of shape (n_clusters, n_features) representing the cluster centroids.
- `labels`: A 1D numpy array of shape (n_samples,) containing the assigned cluster labels for each data point.

## Usage Example

```python
from kmeans import KMeans
import numpy as np

# Create sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Create KMeans object
kmeans = KMeans(n_clusters=2)

# Fit the model
kmeans.fit(X)

# Print the cluster labels
print(kmeans.labels)

# Predict the cluster for a new data point
new_data = np.array([[1.1, 2.3], [7, 8.5]])
print(kmeans.predict(new_data))
```

## Algorithm Details

The K-means algorithm implemented in this class follows these steps:

1. Initialization:

   - Randomly select `n_clusters` data points from the input data `X` as the initial centroids.

2. Iteration:

   - Assign each data point to the nearest centroid based on the Euclidean distance.
   - Update the centroids by calculating the mean of all data points assigned to each centroid.
   - Repeat steps a and b until convergence (centroids stop moving) or the maximum number of iterations is reached.

3. Prediction:
   - For new data points, calculate the distances to all centroids and assign the data points to the nearest centroid.

Note: This implementation uses the Euclidean distance as the distance metric. You can modify the `_calc_distances` method to use a different distance metric if desired.
