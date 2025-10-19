import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def relu_test():
    x = np.random.uniform(-100, 100, (20,))
    ret = ReLU(x)
    return ret

if __name__ == "__main__":
    output = relu_test()