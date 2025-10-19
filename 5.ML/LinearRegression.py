import numpy as np

class LinearRegression:
    def __init__(self, learing_rate = 0.001, n_iters = 1000) -> None:
        self.lr = learing_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        # 如果X是2D数组，确保维度正确
        if X.ndim == 2:
            output = np.dot(X, self.weights) + self.bias
        else:
            # 如果X是1D数组，转换为2D
            X = X.reshape(-1, 1)
            output = np.dot(X, self.weights) + self.bias
        return output

def test_linear_regression():
    """测试线性回归模型"""
    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_weights = np.array([2.5, -1.8, 0.9])
    true_bias = 3.2
    y = np.dot(X, true_weights) + true_bias + np.random.randn(100) * 0.1
    
    # 训练模型
    model = LinearRegression(learing_rate=0.01, n_iters=1000)
    model.fit(X, y)
    
    # 预测和评估
    y_pred = model.predict(X)
    mse = np.mean((y_pred - y) ** 2)
    
    # 输出关键结果
    print(f"真实权重: {true_weights}")
    print(f"学习权重: {model.weights}")
    print(f"真实偏置: {true_bias}")
    print(f"学习偏置: {model.bias}")
    print(f"MSE: {mse:.6f}")
    
    # 测试新数据
    new_X = np.array([[1.0, 2.0, 3.0]])
    new_y_pred = model.predict(new_X)
    expected_y = np.dot(new_X, true_weights) + true_bias
    print(f"新数据预测: {new_y_pred[0]:.4f}, 期望: {expected_y[0]:.4f}")

if __name__ == "__main__":
    test_linear_regression()