import numpy as np

def ce_loss(y_pred, y_true):
    exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    safe_softmax_o = exps / np.sum(exps, axis=1, keepdims=True)
    epsilon = 1e-7
    clipped = np.clip(safe_softmax_o, epsilon, 1 - epsilon)
    n_samples = y_true.shape[0]
    log_likelihood = -np.log(clipped[range(n_samples), y_true.argmax(axis=1)])
    return np.mean(log_likelihood)

def ce_test():
    num_class = 5
    num_samples = 10
    labels = np.random.randint(0, num_class, size=num_samples)
    y_true = np.eye(num_class)[labels]
    y_pred = np.random.rand(num_samples, num_class) * 3
    loss = ce_loss(y_pred, y_true)
    print(loss)

if __name__ == "__main__":
    ce_test()