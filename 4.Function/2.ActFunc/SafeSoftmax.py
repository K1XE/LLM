import numpy as np

def SafeSoftmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    output = e / np.sum(e, axis=-1, keepdims=True)
    return output

def test():
    x = np.random.uniform(-5, 5, 20)
    output = SafeSoftmax(x)
    return output

if __name__ == "__main__":
    output = test()
    print(output)
    