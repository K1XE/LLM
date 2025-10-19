import numpy as np
from numpy import ndarray

def KL_divergence(p : ndarray, q : ndarray):
    mask = (p != 0)
    p = p[mask]
    q = q[mask]
    ret = np.sum(p * np.log(p / q))
    return ret

def kl_test():
    num_classes = 100
    P = np.random.rand(num_classes)
    P /= P.sum()
    Q1 = P
    Q2 = np.random.rand(num_classes)
    Q2 /= Q2.sum()
    print(KL_divergence(P, Q1))
    print(KL_divergence(P, Q2))
    
if __name__ == "__main__":
    kl_test()