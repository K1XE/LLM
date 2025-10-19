import numpy as np

def bce_loss(y_pred, y_true):
    sigmoid_o = 1 / (1 + np.exp( - y_pred))
    epsilon = 1e-7
    clipped = np.clip(sigmoid_o, epsilon, 1 - epsilon)
    log_likelihood = - (y_pred * np.log(clipped) + (1 - y_pred) * np.log(1 - clipped))
    return np.mean(log_likelihood)

def bce_test():
    num_class = 5
    num_samples = 10
    y_pred = np.random.rand(10, 5)
    labels = np.random.randint(0, num_class, num_samples)
    y_true = np.eye(num_class)[labels]
    loss = bce_loss(y_pred, y_true)
    print(loss)

if __name__ == "__main__":
    bce_test()