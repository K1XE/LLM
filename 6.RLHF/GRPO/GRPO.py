import math
from typing import Optional, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def compute_token_logprobs(logits: Tensor, target_ids: Tensor, ignore_index: int = -100) -> Tensor:
    """
    计算给定标签下，每个位置的对数概率 log pi(a_t | s_t)。

    参数:
        logits: 张量，形状 [batch_size, seq_len, vocab_size]，模型未归一化的输出。
        target_ids: 张量，形状 [batch_size, seq_len]，目标 token 的 id（通常是右移后的标签）。
        ignore_index: 整数，被标记为该值的位置会被忽略（例如 padding 或无需训练的部分）。

    返回:
        张量，形状 [batch_size, seq_len]，对应位置的对数概率。被忽略的位置填充为 0。
    """
    log_probs = F.log_softmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = log_probs.shape

    # 展平后再按目标 id 采样对应类别的 log prob
    flat_log_probs = log_probs.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1)

    # Mask-out ignored positions to avoid invalid gather
    valid_mask = flat_targets != ignore_index
    safe_targets = flat_targets.clone()
    safe_targets[~valid_mask] = 0

    # 在最后一维上根据目标 id 进行 gather，得到每个位置的 log prob
    gathered = flat_log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    gathered[~valid_mask] = 0.0

    return gathered.reshape(batch_size, seq_len)


def compute_sequence_logprobs(logits: Tensor, target_ids: Tensor, ignore_index: int = -100, reduction: str = "sum") -> Tensor:
    """
    将逐位置对数概率聚合为逐序列的对数概率（序列对数似然）。

    参数:
        logits: [batch_size, seq_len, vocab_size]
        target_ids: [batch_size, seq_len]
        ignore_index: 被忽略的标签 id。
        reduction: 聚合方式，"sum" 表示对有效位置求和，"mean" 表示按有效长度取平均。

    返回:
        [batch_size] 的张量，表示每个样本的序列对数概率。
    """
    token_logprobs = compute_token_logprobs(logits, target_ids, ignore_index)
    valid_mask = (target_ids != ignore_index).float()
    if reduction == "sum":
        seq_logprobs = (token_logprobs * valid_mask).sum(dim=-1)
    elif reduction == "mean":
        lengths = valid_mask.sum(dim=-1).clamp_min(1.0)
        seq_logprobs = (token_logprobs * valid_mask).sum(dim=-1) / lengths
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")
    return seq_logprobs


def group_normalize(values: Tensor, group_ids: Tensor, eps: float = 1e-8) -> Tensor:
    """
    按组进行标准化：对同一组内的值做 (x - 组均值) / (组标准差 + eps)。

    典型用法：将逐样本奖励转换为逐样本优势（组内零均值、单位方差），
    其中一组通常对应同一个 prompt 的多条采样。

    参数:
        values: [batch_size]，例如每个样本的奖励。
        group_ids: [batch_size]，整型，同组样本具有相同的 id。
    返回:
        [batch_size]，组内标准化后的数值。
    """
    unique_ids = torch.unique_consecutive(group_ids) if torch.all(group_ids[:-1] <= group_ids[1:]) else torch.unique(group_ids)
    normalized = torch.empty_like(values)
    for gid in unique_ids:
        mask = group_ids == gid
        group_vals = values[mask]
        mean = group_vals.mean()
        std = group_vals.std(unbiased=False)
        normalized[mask] = (group_vals - mean) / (std + eps)
    return normalized


def clipped_surrogate_objective(new_logprobs: Tensor, old_logprobs: Tensor, advantages: Tensor, clip_ratio: float = 0.2) -> Tensor:
    """
    PPO 的剪切代理目标（返回值为需最小化的损失）：
        L = - E[ min(r * A, clip(r, 1-ε, 1+ε) * A) ]

    其中 r = exp(new_logprobs - old_logprobs) 是概率比，A 是优势。

    参数:
        new_logprobs: [batch_size]，新策略下的对数概率（通常为序列级或动作级的和）。
        old_logprobs: [batch_size]，行为策略（旧策略）下的对数概率。
        advantages: [batch_size]，优势。
        clip_ratio: ε，剪切范围。
    返回:
        标量张量，损失值。
    """
    ratios = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    surrogate = torch.minimum(unclipped, clipped)
    return -surrogate.mean()


class GRPOTrainer:
    """
    最小可用的 GRPO 训练器实现。

    核心流程：
    - 计算每个样本的序列级对数概率（可看作整条回复的 log prob）。
    - 将逐样本奖励按组标准化得到优势（组内零均值、单位方差），使得每个 prompt 的相对排名更重要。
    - 使用 PPO 剪切目标进行策略更新，稳定训练。
    - 可选：加入对参考策略（ref model）的 KL 正则（逐动作近似），防止策略漂移过大。
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        lr: float = 1e-4,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.0,
        ignore_index: int = -100,
        max_grad_norm: Optional[float] = 1.0,
        optimizer_cls=torch.optim.AdamW,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef
        self.ignore_index = ignore_index
        self.max_grad_norm = max_grad_norm

        # Reference policy should not update
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def compute_old_logprobs(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        logits = self.model(input_ids)
        seq_logprobs = compute_sequence_logprobs(logits, labels, self.ignore_index, reduction="sum")
        return seq_logprobs.detach()

    def step(
        self,
        input_ids: Tensor,
        labels: Tensor,
        group_ids: Tensor,
        rewards: Tensor,
        old_logprobs: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """
        执行一步 GRPO 更新。

        参数:
            input_ids: [batch_size, seq_len]，模型输入 token。
            labels: [batch_size, seq_len]，监督标签（通常与 input 对齐，未训练位置置为 ignore_index）。
            group_ids: [batch_size]，同一 prompt 的样本使用相同的组 id。
            rewards: [batch_size]，逐样本标量奖励。
            old_logprobs: [batch_size]，可选，行为策略的对数概率；若为 None，则用当前模型前向计算并 detach 作为旧策略。

        返回:
            训练统计信息字典，例如损失、平均比率、优势均值等。
        """
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        group_ids = group_ids.to(device)
        rewards = rewards.to(device)

        if old_logprobs is None:
            with torch.no_grad():
                logits_old = self.model(input_ids)
                old_logprobs = compute_sequence_logprobs(logits_old, labels, self.ignore_index, reduction="sum")
        else:
            old_logprobs = old_logprobs.to(device)

        logits = self.model(input_ids)
        new_logprobs = compute_sequence_logprobs(logits, labels, self.ignore_index, reduction="sum")

        advantages = group_normalize(rewards, group_ids)
        loss_policy = clipped_surrogate_objective(new_logprobs, old_logprobs, advantages, self.clip_ratio)

        kl_loss = torch.tensor(0.0, device=device)
        if self.ref_model is not None and self.kl_coef > 0.0:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids)
                ref_logprobs = compute_sequence_logprobs(ref_logits, labels, self.ignore_index, reduction="sum")
            # 逐动作 KL 的近似：sum_t KL ≈ sum_t (log pi - log pi_ref) 在已采样动作上的估计
            kl_estimate = (new_logprobs - ref_logprobs).mean()
            kl_loss = self.kl_coef * kl_estimate

        loss = loss_policy + kl_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            ratio_mean = torch.exp(new_logprobs - old_logprobs).mean().item()
            adv_mean = advantages.mean().item()
            reward_mean = rewards.mean().item()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(loss_policy.item()),
            "kl_loss": float(kl_loss.item()),
            "ratio_mean": ratio_mean,
            "adv_mean": adv_mean,
            "reward_mean": reward_mean,
        }


class TinyToyLM(nn.Module):
    """
    极简的因果语言模型示例：仅包含 Embedding + 线性输出层。

    用于演示 GRPO 训练流程，不包含位置编码与掩码逻辑。
    """

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        # 简化的 teacher-forcing：直接用当前位置的 embedding 预测当前位置的 token（未做右移）
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        return logits


def grpo_demo_test() -> Dict[str, float]:
    """
    在合成数据上运行一小步 GRPO 训练，验证实现可用。
    """
    torch.manual_seed(0)

    batch_size = 12
    seq_len = 8
    vocab_size = 32
    hidden_dim = 16

    # 假设有 4 个 prompt，每个 prompt 采样 3 条回复（组大小 = 3）
    num_prompts = 4
    samples_per_prompt = 3
    assert batch_size == num_prompts * samples_per_prompt

    model = TinyToyLM(vocab_size, hidden_dim)
    ref_model = TinyToyLM(vocab_size, hidden_dim)
    ref_model.load_state_dict(model.state_dict())  # 让参考模型与当前模型参数一致，便于稳定起步

    trainer = GRPOTrainer(model, ref_model=ref_model, lr=5e-4, clip_ratio=0.2, kl_coef=0.01, ignore_index=-100)

    # 构造假输入与标签
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()  # 自监督式示例：预测自身（真实任务中通常为右移标签）

    # 组 id 示例: [0,0,0, 1,1,1, 2,2,2, 3,3,3]
    group_ids = torch.arange(num_prompts).repeat_interleave(samples_per_prompt)

    # 每个样本的合成奖励（均值约为 1.0）
    rewards = torch.randn(batch_size) * 0.5 + 1.0

    stats = trainer.step(input_ids, labels, group_ids, rewards)
    print({k: round(v, 6) for k, v in stats.items()})
    return stats


if __name__ == "__main__":
    grpo_demo_test()


