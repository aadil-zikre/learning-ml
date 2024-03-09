# Logistic Regression from Scratch

This is a Python implementation of Logistic Regression from scratch using NumPy. The `LogisticRegression` class provides methods to train the model, make predictions, and evaluate the model's performance using various metrics and visualizations.

## Features

- Train a logistic regression model using gradient descent optimization
- Make predictions on new data using the trained model
- Evaluate the model's performance using accuracy, F1 score, precision-recall curve, ROC curve, and confusion matrix
- Regularization (L2) to prevent overfitting
- Feature scaling to improve convergence
- Optional inspection of matrix shapes during training and prediction
- Early stopping based on convergence tolerance

## Installation

To use this implementation, you need to have the following dependencies installed:

- NumPy
- scikit-learn
- Seaborn
- Matplotlib

You can install them using pip:

```
pip install numpy scikit-learn seaborn matplotlib
```

## Usage

### Importing the Class

```python
from logistic_regression import LogisticRegression
```

### Creating an Instance

```python
model = LogisticRegression(learning_rate=0.01, num_iterations=1000, regularization=0.01, inspect=False, tolerance=1e-4)
```

- `learning_rate`: The learning rate for gradient descent optimization (default: 0.01).
- `num_iterations`: The maximum number of iterations for training (default: 1000).
- `regularization`: The regularization parameter (lambda) for L2 regularization (default: 0.01).
- `inspect`: Whether to print the shapes of matrices during training and prediction (default: False).
- `tolerance`: The convergence tolerance for early stopping (default: 1e-4).

### Training the Model

```python
model.fit(X_train, y_train)
```

- `X_train`: The training features as a 2D NumPy array.
- `y_train`: The training labels as a 1D NumPy array.

### Making Predictions

```python
predictions = model.predict(X_test)
```

- `X_test`: The test features as a 2D NumPy array.
- Returns the predicted class labels for the test data.

### Evaluating the Model

```python
metrics = ['accuracy', 'f1', 'pr_curve', 'roc_curve', 'confusion_matrix']
evaluation_results = model.evaluate(X_test, y_test, metrics=metrics)
```

- `X_test`: The test features as a 2D NumPy array.
- `y_test`: The true labels for the test data as a 1D NumPy array.
- `metrics`: A list of evaluation metrics to calculate (default: all metrics).
  - Available metrics: 'accuracy', 'f1', 'pr_curve', 'roc_curve', 'confusion_matrix'.
- Returns a dictionary containing the calculated evaluation metrics.

### Accessing Evaluation Results

```python
accuracy = evaluation_results['accuracy']
f1 = evaluation_results['f1']
pr_auc = evaluation_results['pr_curve']
roc_auc = evaluation_results['roc_curve']
confusion_mat = evaluation_results['confusion_matrix']
```

- The evaluation results can be accessed individually from the dictionary returned by the `evaluate` method.

## Example

```python
from logistic_regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LogisticRegression class
model = LogisticRegression(learning_rate=0.01, num_iterations=1000, regularization=0.01, inspect=True, tolerance=1e-4)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
metrics = ['accuracy', 'f1', 'pr_curve', 'roc_curve', 'confusion_matrix']
evaluation_results = model.evaluate(X_test, y_test, metrics=metrics)

# Access the evaluation results
accuracy = evaluation_results['accuracy']
f1 = evaluation_results['f1']
pr_auc = evaluation_results['pr_curve']
roc_auc = evaluation_results['roc_curve']
confusion_mat = evaluation_results['confusion_matrix']

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision-Recall AUC: {pr_auc:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print("Confusion Matrix:\n", confusion_mat)
```

This example demonstrates how to use the `LogisticRegression` class to train a model on the breast cancer dataset, make predictions, and evaluate the model's performance using various metrics and visualizations.

## Conclusion

This implementation of Logistic Regression from scratch provides a basic framework for training and evaluating a binary classification model. It can be extended and customized further based on specific requirements. The use of Seaborn for visualizations enhances the readability and aesthetics of the evaluation plots.
