import numpy as np

def swish(x, beta = 1.0):
    output = x / (1 + np.exp(-beta * x))
    return output

def swish_test():
    x = np.random.uniform(-5, +5, 20)
    output = swish(x, beta=2.0)
    return output

if __name__ == "__main__":
    o = swish_test()
    print(o)