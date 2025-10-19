import numpy as np

class SoftmaxRegression:
    def __init__(self, n_classes, learning_rate=0.01, n_iters=1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.weights = None # [k, n]
        self.bias = None # [k, 1]
        
    def _softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        return probs
    
    def _one_hot(self, y):
        m = y.shape[0]
        y_one_hot = np.zeros((m, self.n_classes))
        y_one_hot[np.range(m), y] = 1
        return y_one_hot
        
    def fit(self, X, y):
        m, n = X.shape
        y_one_hot = self._one_hot(y) # [m, k]
        self.weights = np.zeros((self.n_classes, n)) # [k, n]
        self.bias = np.zeros(self.n_classes) # [k, 1]
        
        for _ in range(self.n_iters):
            Z = np.dot(X, self.weights) + self.bias # [m, k]
            probs = self._softmax(Z) # [m, k]
            
            dZ = probs - y_one_hot
            dw = (1 / m) * np.dot(dZ.T, X) # [k, n]
            db = (1 / m) * np.sum(dZ, axis=0)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        Z = np.dot(X, self.weights.T) + self.bias
        probs = self._softmax(Z)
        cls = np.argmax(probs, axis=1)
        