import torch
from torch import nn, Tensor


def masked_sum(values: Tensor, mask: Tensor, dim=None):
    return (values * mask).sum(dim=dim)

def masked_mean(values: Tensor, mask: Tensor, dim=None):
    s = masked_sum(values, mask, dim)
    return s / mask.sum(dim=dim)

def masked_var(values: Tensor, mask: Tensor, unbias, dim=None):
    mean = masked_mean(values, mask, dim)
    var = torch.mean((values - mean)**2, dim=dim)
    if unbias:
        s = mask.sum()
        if s > 1: ratio = s / (s - 1); var = ratio * var
    return var

def masked_whiten(values, mask, dim=None):
    mean, var = masked_mean(values, mask, dim), masked_var(values, mask, dim)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    return whitened


def compute_gae_adv_return(rewards, value, mask, gamma, lam):
    
    reversed_gae = []
    lastgae = 0
    nextvalue = 0
    l = value.shape[-1]
    for i in reversed(range(l)):
        delta = rewards[:, i] + gamma * nextvalue - value[:, i]
        lastgae_ = delta + lam * gamma * lastgae
        nextvalue = value[:, i] * mask[:, i] + nextvalue * (1 - mask[:, i])
        lastgae = lastgae_ * mask[:, i] + lastgae * (1 - mask[:, i])
        reversed_gae.append(lastgae)
    adv = torch.stack(reversed_gae, dim=1)
    returns = adv + value
    adv = masked_whiten(adv, mask)
    return adv, returns