# Linear Regression from Scratch

linear_regression.py contains a Python implementation of a linear regression model from scratch using NumPy. The `LinearRegression` class provides methods for training the model using gradient descent or the ordinary least squares (OLS) solution, as well as options for L1 and L2 regularization and feature scaling.

## Features

- Train linear regression model using gradient descent or OLS solution
- L1 and L2 regularization for ridge regression and LASSO regression
- Feature scaling (standardization) option
- Convergence condition for gradient descent
- Inspection mode to print shapes of input data and intermediate matrices

## Installation

This implementation requires Python 3.x and NumPy. You can install NumPy using pip:

```
pip install numpy
```

## Usage

1. Import the `LinearRegression` class from the module:

```python
from linear_regression import LinearRegression
```

2. Create an instance of the `LinearRegression` class with desired parameters:

```python
model = LinearRegression(learning_rate=0.01, n_iters=1000, convergence_delta=1e-6,
                         inspect=False, regularizer='none', alpha=0.1, scale_features=False)
```

3. Prepare your training data `X` (features) and `y` (target variable).

4. Train the model using gradient descent:

```python
model.fit(X, y)
```

Or train the model using the OLS solution:

```python
model.fit_ols(X, y)
```

5. Make predictions on new data:

```python
y_pred = model.predict(X_new)
```

Or make predictions using the OLS solution:

```python
y_pred_ols = model.predict_ols(X_new)
```

## Parameters

- `learning_rate` (float, default=0.01): Learning rate for gradient descent.
- `n_iters` (int, default=1000): Number of iterations for gradient descent.
- `convergence_delta` (float, default=1e-6): Threshold for convergence in gradient descent.
- `inspect` (bool, default=False): Whether to print shapes of input data and intermediate matrices.
- `regularizer` (str, default='none'): Regularization method ('l1', 'l2', or 'none').
- `alpha` (float, default=0.1): Regularization strength.
- `scale_features` (bool, default=False): Whether to standardize features by removing mean and scaling to unit variance.

## Example

```python
import numpy as np
from linear_regression import LinearRegression

# Generate some sample data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([2, 6, 10, 14, 18])

# Create an instance of the LinearRegression class with L1 regularization and feature scaling
model = LinearRegression(inspect=True, regularizer='l1', alpha=0.5, scale_features=True)

# Train the model using gradient descent
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
print(y_pred)
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
