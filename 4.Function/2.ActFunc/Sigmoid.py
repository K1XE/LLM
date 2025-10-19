import numpy as np

def sigmoid(x):
    clipped = np.clip(x, -50, 50)
    ret = 1 / (1 + np.exp(-clipped))
    return ret

def sigmoid_test():
    x = np.random.uniform(-100, 100, size=(100,))
    output = sigmoid(x)
    return output

if __name__ == "__main__":
    sigmoid_test()