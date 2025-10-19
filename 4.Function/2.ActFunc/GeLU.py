import numpy as np

def GeLU(x):
    output = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    return output

def gelu_test():
    x = np.random.uniform(-10, 10, 20)
    output = GeLU(x)
    return output

if __name__ == "__main__":
    output = gelu_test()
    print(output)