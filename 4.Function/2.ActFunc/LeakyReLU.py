import numpy as np

def LeakyReLU(x, alpha = 0.01):
    ret = np.where(x > 0, x, alpha * x)
    return ret
def LeakyReLU_test():
    x = np.random.uniform(-100, 100, (20,))
    output = LeakyReLU(x, alpha=0.01)
    return output
if __name__ == "__main__":
    o = LeakyReLU_test()
