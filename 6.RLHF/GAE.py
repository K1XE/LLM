import torch
from torch import Tensor

def GAE(values: Tensor, rewards: Tensor, gamma, lambd):
    # [bsz, seq_len]
    suffix_gae = 0
    adv_reversed = []
    response_length = values.shape[1]
    
    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t + 1 < response_length else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        suffix_gae = delta + gamma * lambd * suffix_gae
        adv_reversed.append(suffix_gae)
    advantages = torch.stack(adv_reversed[::-1], dim=1)
    return advantages.detach()