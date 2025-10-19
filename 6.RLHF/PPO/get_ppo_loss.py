import torch
from torch import nn, Tensor


def masked_sum(values, mask, axis=None):
    valid = torch.where(mask.float(), values)
    return (valid * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def agg_loss(loss_mat, loss_mask, agg_type):
    if agg_type == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    elif agg_type == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.mean(seq_losses)
    elif agg_type == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(
            loss_mask, dim=-1
        )
        return torch.mean(seq_losses)
    elif agg_type == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.sum(seq_losses) / loss_mask.shape[-1]


def get_policy_loss(
    log_prob,
    old_log_prob,
    advantage,
    clip,
    clip_epsilon_hi,
    clip_epsilon_lo,
    clip_c,
    response_mask,
    agg_type,
):
    neg_kl = log_prob - old_log_prob
    neg_kl = torch.clamp(neg_kl, -20, 20)
    ratio = torch.exp(neg_kl)
    ppo_kl = torch.mean(-neg_kl, dim=-1)
    loss1 = -ratio * advantage
    if clip_epsilon_hi is None:
        clip_epsilon_hi = clip
    if clip_epsilon_lo is None:
        clip_epsilon_lo = clip
    loss2 = -torch.clamp(ratio, clip_epsilon_lo, clip_epsilon_hi) * advantage
    loss_ret1 = torch.maximum(loss1, loss2)
    clip_frac = masked_mean(torch.gt(loss2, loss1).float(), response_mask)
    loss3 = -clip_c * advantage
    loss_ret2 = torch.min(loss3, loss_ret1)
    clip_frac_lo = masked_mean(
        torch.gt(loss3, loss_ret1) * (advantage < 0).float(), response_mask
    )
    pg_loss = torch.where(advantage < 0, loss_ret2, loss_ret1)
    # agg pg_loss
    pg_loss = agg_loss(pg_loss, response_mask, agg_type)
    return pg_loss, ppo_kl, clip_frac, clip_frac_lo


def get_value_loss(vpred, values, returns, clip, response_mask, agg_type):
    vpredclipped = torch.clamp(vpred, values - clip, values + clip)
    vf_loss1 = (vpredclipped - returns) ** 2
    vf_loss2 = (values - returns) ** 2
    clipped_vf_loss = torch.max(vf_loss1, vf_loss2)
    clip_frac = masked_mean(torch.gt(vf_loss1, vf_loss2).float())
    # agg vf_loss
    clipped_vf_loss = agg_loss(clipped_vf_loss, response_mask, agg_type)
    return clipped_vf_loss, clip_frac
