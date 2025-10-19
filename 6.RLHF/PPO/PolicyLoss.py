import torch
from torch import Tensor

def PolicyLoss(log_probs, old_log_probs, advantages):
    ratio: Tensor = log_probs - old_log_probs
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps, ratio) * advantages
    loss = - -torch.min(surr1, surr2)
    return loss.mean()