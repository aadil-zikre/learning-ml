# Decision Tree Classifier

This is an implementation of a Decision Tree Classifier from scratch using Python. The Decision Tree Classifier is a popular machine learning algorithm used for classification tasks. It builds a tree-like model of decisions and their possible consequences based on the features of the input data.

## Features

- Supports both entropy and Gini index as the split criteria for building the tree.
- Allows specifying the maximum depth of the tree and the minimum number of samples required to split a node.
- Handles categorical and numerical features.
- Provides methods for training the model and making predictions on new data.

## Installation

To use the Decision Tree Classifier, you need to have Python installed on your system. You can download Python from the official website: [https://www.python.org](https://www.python.org)

The implementation requires the NumPy library. You can install it using pip:

```
pip install numpy
```

## Usage

1. Import the necessary libraries and the `DecisionTree` class:

```python
import numpy as np
from decision_tree import DecisionTree
```

2. Create an instance of the `DecisionTree` class:

```python
dt = DecisionTree(max_depth=5, min_samples_split=10, criterion='entropy')
```

- `max_depth`: The maximum depth of the tree (default: None, which means no depth limit).
- `min_samples_split`: The minimum number of samples required to split a node (default: 2).
- `criterion`: The split criterion to use, either 'entropy' or 'gini' (default: 'entropy').

3. Train the Decision Tree Classifier using the `fit` method:

```python
X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], ...])
y_train = np.array([0, 1, 0, ...])
dt.fit(X_train, y_train)
```

- `X_train`: The training input samples, where each sample is an array of features.
- `y_train`: The corresponding target values (class labels) for the training samples.

4. Make predictions on new data using the `predict` method:

```python
X_test = np.array([[1, 2, 3], [4, 5, 6], ...])
predictions = dt.predict(X_test)
```

- `X_test`: The input samples for which you want to make predictions.
- `predictions`: The predicted class labels for the input samples.

## Example

Here's an example of using the Decision Tree Classifier:

```python
import numpy as np
from decision_tree import DecisionTree

# Create a sample dataset
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 1, 0, 1])

# Create an instance of the Decision Tree Classifier
dt = DecisionTree(max_depth=2, min_samples_split=2, criterion='gini')

# Train the Decision Tree
dt.fit(X_train, y_train)

# Make predictions on new data
X_test = np.array([[2, 3], [6, 7]])
predictions = dt.predict(X_test)

print("Predictions:", predictions)
```

Output:

```
Predictions: [0 1]
```
