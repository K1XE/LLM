import torch
from torch import Tensor


def masked_sum(values, mask, axis=None):
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    s = masked_sum(values, mask, axis)
    return s / mask.sum(axis=axis)


def agg_loss(loss_mat, loss_mask, agg_type):
    if agg_type == "token_mean":
        return masked_mean(loss_mat, loss_mask)
    elif agg_type == "token_sum_seq_mean":
        s = (loss_mat * loss_mask).sum(axis=-1)
        return torch.mean(s)
    elif agg_type == "token_sum_seq_mean_norm":
        s = (loss_mat * loss_mask).sum(axis=-1)
        return torch.sum(s) / loss_mask.shape[-1]
    elif agg_type == "token_mean_seq_mean":
        m = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        return torch.mean(m)


def compute_policy_loss(
    log_probs,
    old_log_probs,
    advantage,
    clip_ratio,
    clip_dual,
    clip_hi,
    clip_lo,
    response_mask,
    agg_type,
):
    neg_kl = log_probs - old_log_probs
    ppo_kl = masked_mean(-neg_kl, response_mask)
    ratio = torch.exp(neg_kl)
    loss1 = -ratio * advantage
    if clip_hi is None:
        clip_hi = clip_ratio
    if clip_lo is None:
        clip_lo = clip_ratio
    loss2 = -torch.clamp(ratio, 1 + clip_hi, 1 - clip_lo) * advantage
    clip_loss1 = torch.maximum(loss1, loss2)
    clip_rate1 = masked_mean(torch.gt(loss2, loss1).float(), response_mask)
    loss3 = -clip_dual * advantage
    clip_loss2 = torch.minimum(loss3, clip_loss1)
    clip_rate2 = masked_mean(
        torch.gt(clip_rate1, loss3) * (advantage < 0).float(), response_mask
    )

    loss = agg_loss(clip_loss2, response_mask, agg_type)

    return loss, ppo_kl, clip_rate1, clip_rate2


def compute_value_loss(vpreds, old_value, clip_value, returns, response_mask, agg_type):
    clipped_value = torch.clamp(vpreds, old_value - clip_value, old_value + clip_value)
    loss1 = 0.5 * (clipped_value - returns) ** 0.5
    loss2 = 0.5 * (vpreds - returns) ** 0.5

    loss = torch.maximum(loss1, loss2)

    vloss = agg_loss(loss, response_mask, agg_type)
    clipped_rate = masked_mean(torch.gt(loss1, loss2).float(), response_mask)

    return vloss, clipped_rate


def compute_entropy_loss(logits, agg_type, response_mask):
    prob = torch.nn.functional.softmax(logits, dim=-1)
    mid = torch.logsumexp(logits, dim=-1) - torch.sum(logits * prob, dim=-1)
    loss = agg_loss(mid, response_mask, agg_type)
    return loss


def kl_penalty(log_prob, ref_log_prob, kl_type):
    forward: Tensor = kl_penalty_forward(log_prob, ref_log_prob, kl_type)
    if kl_type in ("mse", "k2"):
        return forward
    backward: Tensor = kl_penalty_forward(log_prob, ref_log_prob, "k2")
    return backward - backward.detach() + forward.detach()


def kl_penalty_forward(log_prob, ref_log_prob, kl_type):
    r = log_prob - ref_log_prob
    if kl_type == "k1":
        return r
    elif kl_type in ("k1", "mse"):
        return 0.5 * r**2
    elif kl_type in ("k3", "low_var_kl"):
        b = -r - 1
        return b + torch.exp(-r)


def compute_pg_loss(
    logp,
    old_logp,
    ref_logp,
    adv,
    clip_ratio,
    clip_hi,
    clip_lo,
    clip_dual,
    logits,
    response_mask,
    agg_type_policy,
    agg_type_entropy,
    kl_type,
):
    loss = (
        compute_policy_loss(
            logp,
            old_logp,
            adv,
            clip_ratio,
            clip_dual,
            clip_hi,
            clip_lo,
            response_mask,
            agg_type_policy,
        )
        - compute_entropy_loss(logits, agg_type_entropy, response_mask)
        + kl_penalty(logp, ref_logp, kl_type)
    )
    return loss
