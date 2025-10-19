# https://www.cnblogs.com/mengrennwpu/p/17999027#:~:text=direct%2Dpreference%2Doptimization-,5.1%20DPO%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0,-1%20def%20preference_loss
import torch
import torch.nn.functional as F
from typing import Tuple


def concatenated_forward(model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
    """
    前向传播函数，计算策略模型或参考模型的log概率
    
    Args:
        model: 策略模型或参考模型
        chosen_ids: 选择的token ids
        c_mask: 选择的mask
        reject_ids: 拒绝的token ids  
        r_mask: 拒绝的mask
        prompt_id_lens: prompt的长度
    
    Returns:
        chosen_logps: 选择序列的log概率
        rejected_logps: 拒绝序列的log概率
    """
    # 这里需要根据具体的模型实现来计算log概率
    # 假设模型返回logits，然后计算log概率
    chosen_logits = model(chosen_ids, attention_mask=c_mask)
    rejected_logits = model(reject_ids, attention_mask=r_mask)
    
    # 计算log概率 (简化实现，实际需要根据具体模型调整)
    chosen_logps = F.log_softmax(chosen_logits, dim=-1)
    rejected_logps = F.log_softmax(rejected_logits, dim=-1)
    
    # 返回(batch_size,)的张量
    return chosen_logps.mean(dim=-1), rejected_logps.mean(dim=-1)


def DPOLoss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """
    计算DPO损失
    
    Args:
        policy_chosen_logps: 策略模型选择序列的log概率 (batch_size,)
        policy_rejected_logps: 策略模型拒绝序列的log概率 (batch_size,)
        reference_chosen_logps: 参考模型选择序列的log概率 (batch_size,)
        reference_rejected_logps: 参考模型拒绝序列的log概率 (batch_size,)
        beta: DPO温度参数，控制奖励和惩罚的强度
    
    Returns:
        loss: DPO损失值
    """
    # 计算策略模型和参考模型的 log 概率比
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    # 构造 logits 差值
    logits = pi_logratios - ref_logratios
    
    # 计算DPO损失: -log σ(beta * (策略比值 - 参考比值))
    losses = -F.logsigmoid(beta * logits)
    
    return losses.mean()


class DPOTrainer:
    """
    DPO训练器类
    """
    def __init__(self, policy_model, ref_model, beta=0.1):
        """
        初始化DPO训练器
        
        Args:
            policy_model: 策略模型
            ref_model: 参考模型
            beta: DPO温度参数
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta
    
    def compute_loss(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """
        计算DPO损失
        
        Args:
            chosen_ids: 选择的token ids
            c_mask: 选择的mask
            reject_ids: 拒绝的token ids
            r_mask: 拒绝的mask
            prompt_id_lens: prompt的长度
            
        Returns:
            loss: DPO损失值
        """
        # 策略模型的概率
        policy_chosen_logps, policy_rejected_logps = concatenated_forward(
            self.policy_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        
        # 参考模型的概率
        reference_chosen_logps, reference_rejected_logps = concatenated_forward(
            self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        
        # 计算DPO损失
        loss = DPOLoss(
            policy_chosen_logps, 
            policy_rejected_logps, 
            reference_chosen_logps, 
            reference_rejected_logps, 
            self.beta
        )
        
        return loss


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    batch_size = 4
    seq_len = 10
    vocab_size = 1000
    
    # 模拟模型输出
    policy_chosen_logps = torch.randn(batch_size)
    policy_rejected_logps = torch.randn(batch_size)
    reference_chosen_logps = torch.randn(batch_size)
    reference_rejected_logps = torch.randn(batch_size)
    
    # 计算DPO损失
    loss = DPOLoss(
        policy_chosen_logps,
        policy_rejected_logps, 
        reference_chosen_logps,
        reference_rejected_logps,
        beta=0.1
    )
    
    print(f"DPO Loss: {loss.item():.4f}")
