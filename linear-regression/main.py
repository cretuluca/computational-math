import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mse(self, X, y):
        y_predicted = self.predict(X)
        return np.mean((y - y_predicted) ** 2)
        
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(1000000, 1)
    y = 2 * X + 1 + np.random.randn(1000000, 1) * 0.1

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    X_test = np.array([[0.5], [1.5]])
    predictions = model.predict(X_test)

    print("Weights:", model.weights.flatten())
    print("Bias:", model.bias)
    print("Predictions:", predictions.flatten())
    print("MSE:", model.mse(X, y))