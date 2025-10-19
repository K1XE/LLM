# http://joschu.net/blog/kl-approx.html
import torch.distributions as dis

def KLEstimator():
    p = dis.Normal(loc=0, scale=1)
    q = dis.Normal(loc=0.1, scale=1)
    x = q.sample(sample_shape=(10_000_000, ))
    trueKL = dis.kl_divergence(q, p) # should be reversed KL
    print(f"true KL is {trueKL}")
    logr = p.log_prob(x) - q.log_prob(x)
    k1 = - logr
    k2 = logr ** 2 / 2
    k3 = (logr.exp() - 1) - logr
    for k in (k1, k2, k3):
        print((k.mean() - trueKL) / trueKL, k.std() / trueKL)

if __name__ == "__main__":
    KLEstimator()