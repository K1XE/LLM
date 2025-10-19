import torch

def ValueLoss(values, old_values, returns):
    values_clipped = old_values + (values - old_values).clamp(-self.eps, self.eps)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)
    return 0.5 * loss.mean()