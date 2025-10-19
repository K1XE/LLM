import numpy as np

def mse_loss(y_true, y_pred):
    squared_error = (y_pred - y_true) ** 2
    return np.mean(squared_error)

def mse_test():
    y_true = np.random.rand(10)
    y_pred = np.random.rand(10)
    loss = mse_loss(y_true, y_pred)
    print(loss)

if __name__ == "__main__":
    mse_test()