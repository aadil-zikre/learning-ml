import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=0.01, inspect=False, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.inspect = inspect
        self.tolerance = tolerance

    def scale_features(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if self.inspect:
            print(f"scale_features: X shape: {X.shape}")
        return (X - self.mean) / (self.std + 1e-8)

    def sigmoid(self, z):
        if self.inspect:
            print(f"sigmoid: z shape: {z.shape}")
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, weights, bias):
        num_samples = len(y)
        linear_model = np.dot(X, weights) + bias
        y_predicted = self.sigmoid(linear_model)
        cost = (-1 / num_samples) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        cost += (self.regularization / (2 * num_samples)) * np.sum(weights ** 2)
        return cost

    def fit(self, X, y):
        if self.inspect:
            print(f"fit: X shape: {X.shape}, y shape: {y.shape}")
        X = self.scale_features(X)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        prev_cost = float('inf')
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            if self.inspect:
                print(f"fit: linear_model shape: {linear_model.shape}")
            y_predicted = self.sigmoid(linear_model)
            if self.inspect:
                print(f"fit: y_predicted shape: {y_predicted.shape}")

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            if self.inspect:
                print(f"fit: dw shape: {dw.shape}, db shape: {db.shape}")

            # Apply regularization
            dw += (self.regularization / num_samples) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate the cost and check for convergence
            cost = self.cost_function(X, y, self.weights, self.bias)
            if abs(cost - prev_cost) < self.tolerance:
                break
            prev_cost = cost
        
        # Evaluate the model on the training data
        self.training_accuracy = self._accuracy(y, self.predict(X))
        self.training_f1 = self._f1_score(y, self.predict(X))

    def predict(self, X):
        if self.inspect:
            print(f"predict: X shape: {X.shape}")
        X = (X - self.mean) / (self.std + 1e-8)
        linear_model = np.dot(X, self.weights) + self.bias
        if self.inspect:
            print(f"predict: linear_model shape: {linear_model.shape}")
        y_predicted = self.sigmoid(linear_model)
        if self.inspect:
            print(f"predict: y_predicted shape: {y_predicted.shape}")
        return [1 if i > 0.5 else 0 for i in y_predicted]
    
    def evaluate(self, X, y, metrics=None):
        if metrics is None:
            metrics = ['accuracy', 'f1', 'pr_curve', 'roc_curve', 'confusion_matrix']

        y_pred = self.predict(X)

        evaluation_results = {}
        if 'accuracy' in metrics:
            evaluation_results['accuracy'] = self._accuracy(y, y_pred)
        if 'f1' in metrics:
            evaluation_results['f1'] = self._f1_score(y, y_pred)
        if 'pr_curve' in metrics:
            evaluation_results['pr_curve'] = self._pr_curve(X, y)
        if 'roc_curve' in metrics:
            evaluation_results['roc_curve'] = self._roc_curve(X, y)
        if 'confusion_matrix' in metrics:
            evaluation_results['confusion_matrix'] = self._confusion_matrix(y, y_pred)

        return evaluation_results

    def _accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def _f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)

    def _pr_curve(self, X, y_true):
        y_prob = self.predict_proba(X)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=recall, y=precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower right')
        plt.show()

        return pr_auc

    def _roc_curve(self, X, y_true):
        y_prob = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=fpr, y=tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        sns.lineplot(x=[0, 1], y=[0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        return roc_auc

    def _confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

        return cm

    def predict_proba(self, X):
        X = (X - self.mean) / (self.std + 1e-8)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def __repr__(self):
        attrs = {
            'learning_rate': self.learning_rate,
            'num_iterations': self.num_iterations,
            'regularization': self.regularization,
            'inspect': self.inspect,
            'tolerance': self.tolerance
        }
        attr_str = ', '.join(f'{key}={value}' for key, value in attrs.items())

        if self.weights is not None:
            attr_str += f', weights_shape={self.weights.shape}'
        if self.bias is not None:
            attr_str += f', bias={self.bias:.4f}'
        if self.mean is not None:
            attr_str += f', mean_shape={self.mean.shape}'
        if self.std is not None:
            attr_str += f', std_shape={self.std.shape}'
        if self.training_accuracy is not None:
            attr_str += f', training_accuracy={self.training_accuracy:.4f}'
        if self.training_f1 is not None:
            attr_str += f', training_f1={self.training_f1:.4f}'

        return f'LogisticRegression({attr_str})'