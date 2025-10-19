import torch
from torch import nn, Tensor


def mask_sum(value, mask, axis=-1):
    return (value * mask).sum(axis=axis)


def mask_mean(value, mask, axis=-1):
    s = mask_sum(value, mask, axis)
    return s / mask.sum(axis=axis)


def agg_loss(loss_mat, loss_mask, agg_type):
    if agg_type == "token_mean":
        return mask_mean(loss_mat, loss_mask)
    elif agg_type == "token_sum_seq_mean":
        s = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.mean(s)
    elif agg_type == "token_sum_seq_mean_norm":
        s = torch.sum(loss_mat * loss_mask)
        return torch.sum(s) / loss_mask.shape[-1]
    elif agg_type == "token_mean_seq_mean":
        s = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        return torch.mean(s)


def compute_policy_loss(
    old_log_prob,
    log_prob,
    mask,
    advantage,
    clip_ratio,
    agg_type,
    epsilon_hi=None,
    epsilon_lo=None,
    epsilon_dual=None,
):
    neg_kl = log_prob - old_log_prob
    neg_kl = torch.clamp(neg_kl, -20, 20)
    ratio = torch.exp(neg_kl)
    ppo_kl = mask_mean(-neg_kl, mask)
    loss1 = -ratio * advantage
    if epsilon_hi is None:
        epsilon_hi = clip_ratio
    if epsilon_lo is None:
        epsilon_lo = clip_ratio
    loss2 = -torch.clamp(ratio, 1 - epsilon_lo, 1 - epsilon_hi) * advantage
    clipped_frac1 = mask_mean(torch.gt(loss2, loss1).float(), mask)
    clipped_loss1 = torch.max(loss1, loss2)
    assert epsilon_dual > 1.0
    loss3 = -epsilon_dual * advantage
    clipped_frac2 = mask_mean(torch.gt(clipped_loss1, loss3) * (advantage < 0).float(), mask)
    clipped_loss2 = torch.min(loss3, clipped_loss1)
    pg_loss = torch.where(advantage < 0, clipped_loss2, clipped_loss1)
    loss = agg_loss(pg_loss, mask, agg_type)
    return loss, ppo_kl, clipped_frac1, clipped_frac2

def compute_value_loss(vpred, value_clip, value, returns, mask):
    vpred_clipped = torch.clamp(vpred, vpred - value_clip, vpred + value_clip)
    loss1 = (vpred_clipped - returns) ** 2
    loss2 = (value - returns) ** 2
    loss = torch.max(loss1, loss2)
    clipped_frac = mask_mean(torch.gt(loss2, loss1))
    return loss, clipped_frac
