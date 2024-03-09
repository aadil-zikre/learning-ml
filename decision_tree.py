import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return {'leaf_value': leaf_value}

        feature_idx, threshold = self._best_split(X, y)

        left_idxs, right_idxs = self._split(X[:, feature_idx], threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return {'feature_idx': feature_idx, 'threshold': threshold, 'left': left, 'right': right}

    def _best_split(self, X, y):
        best_score = -1
        split_idx, split_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                if self.criterion == 'entropy':
                    score = self._information_gain(y, X[:, feature_idx], threshold)
                elif self.criterion == 'gini':
                    score = self._gini_gain(y, X[:, feature_idx], threshold)
                if score > best_score:
                    best_score = score
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_feature, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_feature, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _gini_gain(self, y, X_feature, threshold):
        parent_gini = self._gini(y)
        left_idxs, right_idxs = self._split(X_feature, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * g_l + (n_r / n) * g_r

        gini_gain = parent_gini - child_gini
        return gini_gain

    def _split(self, X_feature, split_thresh):
        left_idxs = np.argwhere(X_feature <= split_thresh).flatten()
        right_idxs = np.argwhere(X_feature > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        gini = 1 - np.sum([p**2 for p in proportions])
        return gini

    def _most_common_label(self, y):
        most_common = np.argmax(np.bincount(y))
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if 'leaf_value' in node:
            return node['leaf_value']

        if x[node['feature_idx']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])