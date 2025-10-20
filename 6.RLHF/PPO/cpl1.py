import torch
import torch.nn.functional as F

def compute_policy_loss(logp, old_logp, adv, clip_ratio):
    neg_kl = logp - old_logp
    ratio = torch.exp(neg_kl)
    loss1 = - ratio * adv
    loss2 = - torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    clipped_ratio = torch.mean(torch.gt(loss2, loss1).float())
    loss = torch.max(loss1, loss2)
    
    return torch.mean(loss), clipped_ratio

def compute_critic_loss(value, vpred, clipped, returns):
    vpred_clipped = torch.clamp(vpred, value - clipped ,value + clipped)
    loss1 = 0.5 * (vpred - returns) ** 2
    loss2 = 0.5 * (vpred_clipped - returns) ** 2
    clipped_ratio = torch.mean(torch.gt(loss2, loss1).float())
    
    return max(loss1, loss2).mean(), clipped_ratio

def compute_entropy_loss(logits):
    p = F.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) + torch.sum(logits * p, dim=-1)
    return torch.mean(entropy)

def kl_penalty(logp, ref_logp, kl_mode):
    r = logp - ref_logp
    kl = 0
    if kl_mode == "k1":
        kl = r
    elif kl_mode in ("k2", "mse"):
        kl = 0.5 * kl ** 2
    elif kl_mode in ("k3", "low_var_kl"):
        kl = r - 1 + torch.exp(-r)
    else: raise ValueError
    return kl

def compute_gae_returns(values, rewards, gamma, lam):
    l = values.shape[0]
    rev_gae = []
    gae = 0
    for t in reversed(range(l)):
        nextvalue = values[:, t + 1] if t < l else 0.0
        delta = rewards[:, t] + gamma * nextvalue - values[:, t]
        gae = delta + lam * gamma * gae
        rev_gae.append(gae)
    res = torch.stack(rev_gae[::-1], dim=1)
    returns = res + values
    return res, returns