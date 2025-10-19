import torch
from torch import nn, Tensor


def compute_policy_loss(
    logp,
    old_logp,
    ref_logp,
    adv,
    clip_ratio,
    clip_dual,
    clip_hi,
    clip_lo,
    response_mask,
    agg_type,
):
    log_ratio = logp - old_logp
    w1 = torch.exp(log_ratio)
    w2 = torch.exp(old_logp - ref_logp)
    ppo_kl = masked_mean(-log_ratio, response_mask, dim=None)

    loss1 = -w1 * adv
    loss2 = -torch.clamp(w1, clip_hi if clip_hi else clip_ratio, clip_lo if clip_lo else clip_ratio)
    
    clipped_loss1 = torch.max(loss1, loss2)
    
    loss3 = -clip_dual * adv
    
    clipped_loss2 = torch.min(loss3, clipped_loss1)
    
    loss = torch.where(adv < 0, clipped_loss2, clipped_loss1)
    
    return agg_loss(loss, response_mask, agg_type)

def compute_gae(value, rewards, gamma, lam):
    nextvalue = 0
    lastgae = 0
    l_ = value.shape[-1]
    reversed_gae = []
    for t in reversed(range(l_)):
        delta = rewards[:, t] + gamma * nextvalue - value[:, t]
        lastgae = delta + gamma * lam * lastgae
        reversed_gae.append(lastgae)
    gae_list = torch.stack(reversed_gae, dim=1)
    returns = gae_list + value
    return gae_list, returns
    
