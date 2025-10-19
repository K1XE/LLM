import numpy as np

def tanh1(x):
    ex = np.exp(x)
    enx = np.exp(-x)
    ret = (ex - enx) / (ex + enx)
    return ret

def tanh2(x):
    ret = np.tanh(x)
    return ret

def tanh_test():
    x = np.random.uniform(-100, 100, (20,))
    output1 = tanh1(x)
    output2 = tanh2(x)
    print(output1 == output2)
    print(1)
if __name__ == "__main__":
    tanh_test()