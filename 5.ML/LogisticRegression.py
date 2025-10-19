import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, n_iters) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def _sigmoid(self, z):
        output = 1 / (1 + np.exp(-z))
        return output
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred, y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_prob(self, X):
        linear = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(linear)
        return probs
    
    def predict(self, X, threshold = 0.5):
        output = (self.predict_prob(X) >= threshold).astype(int)
        return output