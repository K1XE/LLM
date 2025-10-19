import numpy as np

import torch
import torch.nn as nn


def ELU(x, alpha = 1):
    return np.where(x > 10, x, alpha * (np.exp(x) - 1))
    
def ELU_test():
    x = np.random.uniform(-5, 5, 20)
    output = ELU(x, alpha=1.05)
    nn.Mish
    nn.Softplus
    return output

if __name__ == "__main__":
    output = ELU_test()
    print(output)