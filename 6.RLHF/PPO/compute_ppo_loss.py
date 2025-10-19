import torch
from torch import nn, Tensor
import torch.nn.functional as F
from numpy import ndarray


def masked_sum(values: Tensor, mask: Tensor, axis=None):
    return (values * mask).sum(dim=axis)


def masked_mean(values: Tensor, mask: Tensor, axis=None):
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(dim=axis) + 1e-8)


def agg_loss(loss_mat, loss_mask, agg_type):
    if agg_type == "token_mean":
        return masked_mean(loss_mat, loss_mask)
    elif agg_type == "seq_mean_token_sum":
        s = (loss_mat * loss_mask).sum(dim=-1)
        return torch.mean(s)
    elif agg_type == "seq_mean_token_sum_norm":
        s = (loss_mat * loss_mask).sum(dim=-1)
        return torch.sum(s) / loss_mask.shape[-1]
    elif agg_type == "seq_mean_token_mean":
        mid = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        return torch.mean(mid)
    else:
        raise ValueError


def compute_policy_loss(
    old_logp,
    roll_out_logp,
    logp,
    adv,
    response_mask,
    clip_tis,
    clip_ratio,
    clip_hi,
    clip_lo,
    clip_dual,
    agg_type,
):
    neg_kl = logp - old_logp
    ratio = torch.exp(neg_kl)
    tis_ratio = torch.exp(roll_out_logp - old_logp)
    ppo_kl = masked_mean(-neg_kl, response_mask)
    loss1 = -ratio * adv
    if clip_hi is None:
        clip_hi = clip_ratio
    if clip_lo is None:
        clip_lo = clip_ratio
    loss2 = -torch.clamp(ratio, 1 - clip_lo, 1 + clip_hi) * adv
    clipped_loss1 = torch.maximum(loss1, loss2)
    clipped_rate1 = masked_mean(torch.gt(loss2, loss1).float(), response_mask)

    loss3 = -clip_dual * adv
    clipped_loss2 = torch.min(clipped_loss1, loss3)
    clipped_rate2 = masked_mean(
        torch.gt(clipped_loss2, loss3) * (adv < 0).float(), response_mask
    )

    clipped_loss = torch.where(adv < 0, clipped_loss2, clipped_loss1)

    clipped_loss = torch.clamp(tis_ratio, max=clip_tis) * clipped_loss

    loss = agg_loss(clipped_loss, response_mask, agg_type)

    return ppo_kl, loss, clipped_rate1, clipped_rate2


def kl_penalty(logp, ref_logp, kl_type):
    forward = kl_penalty_forward(logp, ref_logp, kl_type)

    if kl_type in ("k2", "mse"):
        return forward

    backward = kl_penalty_forward(logp, ref_logp, "k2")

    backward = backward - backward.detach() + forward.detach()

    return backward


def kl_penalty_forward(logp, ref_logp, kl_type):
    r = logp - ref_logp

    if kl_type == "k1":
        return r
    elif kl_type in ("k2", "mse"):
        return 0.5 * (logp - ref_logp) ** 2
    elif kl_type in ("k3", "low_var_kl"):
        ret = r - 1 + torch.exp(-r)
        return ret
    else:
        raise ValueError


def compute_entropy_loss(logits, mask, agg_type):
    probs = F.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(logits * probs, dim=-1)
    loss = agg_loss(entropy, mask, agg_type)
    return loss


def compute_value_loss(vpred, returns, value, clipped_hi, clipped_lo, mask, agg_type):
    clipped_vpred = torch.clamp(vpred, value - clipped_lo, value + clipped_hi)
    loss1 = 0.5 * (clipped_vpred - returns) ** 2
    loss2 = 0.5 * (vpred - returns) ** 2
    loss = torch.max(loss1, loss2)
    clipped_rate = masked_mean(torch.gt(loss1, loss2).float(), mask)

    return loss, clipped_rate


def compute_actor_loss(
    ref_logp,
    old_logp,
    roll_out_logp,
    logp,
    logits,
    adv,
    clip_tis,
    clip_ratio,
    clip_hi,
    clip_lo,
    clip_dual,
    agg_type,
    kl_type,
    use_kl_loss,
    mask,
):
    policy_loss = compute_policy_loss(
        old_logp=old_logp,
        roll_out_logp=roll_out_logp,
        logp=logp,
        adv=adv,
        response_mask=mask,
        clip_tis=clip_tis,
        clip_ratio=clip_ratio,
        clip_hi=clip_hi,
        clip_lo=clip_lo,
        clip_dual=clip_dual,
        agg_type=agg_type,
    )
    entropy_loss = compute_entropy_loss(logits, mask, agg_type)
    kl_loss = kl_penalty(logp, ref_logp, kl_type) if use_kl_loss else 0

    return policy_loss - entropy_loss + kl_loss


def compute_critic_loss(vpred, returns, old_value, clip_lo, clip_hi, mask, agg_type):
    loss = compute_value_loss(
        vpred, returns, old_value, clip_hi, clip_lo, mask, agg_type
    )

    return loss
