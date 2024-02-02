import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class LR_GradientDescent:
        def __init__(self, learning_rate=0.01, iterations=1000):
            self.learning_rate = learning_rate
            self.iterations = iterations
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            num_samples, num_features = X.shape

            # Standardize the features
            X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

            # Initialize weights and bias to zeros
            self.weights = np.zeros(num_features)
            self.bias = 0

            # Perform gradient descent
            for _ in range(self.iterations):
                y_predicted = np.dot(X_standardized, self.weights) + self.bias
                dw = (1 / num_samples) * np.dot(X_standardized.T, (y_predicted - y))
                db = (1 / num_samples) * np.sum(y_predicted - y)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        def predict(self, X):
            # Standardize the features
            X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

            return np.dot(X_standardized, self.weights) + self.bias

dataset = pd.read_csv('BostonHousingDataset.csv')
dataset = dataset.drop(['B', 'LSTAT'], axis=1)
dataset_altered = dataset.dropna()
dataset_altered = dataset_altered.astype(float)
dataset_altered.head(10)
from sklearn.model_selection import train_test_split
X = dataset_altered.drop('MEDV', axis=1)
y = dataset_altered['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)


linear_model = LR_GradientDescent()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print(y_pred)


# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(y_pred)