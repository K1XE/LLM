import numpy as np

def belu(refs, candidate, weights=[0.25]*4):
    
    p_n = []
    for n in range(1, len(weights) + 1):
        candidate_grams = [tuple(candidate[i:i+n]) for i in range(0, len(candidate) - n + 1)]
        refs_grams = [[tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)] for ref in refs]
        
        cnt_clipped = 0
        for gram in set(candidate_grams):
            max_ref_cnt = max(ref.count(gram) for ref in refs_grams)
            cnt_clipped += min(max_ref_cnt, candidate_grams.count(gram))
        
        p_n.append(cnt_clipped / len(candidate_grams) if candidate_grams else 0)
    
    c, r = len(candidate), min(len(ref) for ref in refs)
    BP = 1 if c > r else np.exp(1 - r / c)
    
    # 检查是否所有n-gram都不匹配
    if all(p == 0 for p in p_n):
        return 0.0  # 完全不匹配时返回0
    
    score = sum(w * np.log(p) if p > 0 else 0 for w, p in zip(weights, p_n))
    
    return BP * np.exp(score)

def belu_test():
    """
    测试BLEU评分函数的各种情况
    """
    print("开始测试BLEU评分函数...")
    
    # 测试用例1: 完全匹配的情况
    print("\n测试用例1: 完全匹配")
    refs = [['我', '爱', '你'], ['我', '喜欢', '你']]
    candidate = ['我', '爱', '你']
    score = belu(refs, candidate)
    print(f"参考: {refs}")
    print(f"候选: {candidate}")
    print(f"BLEU分数: {score:.4f}")
    print(f"期望: 分数应该接近1.0 (完全匹配)")
    
    # 测试用例2: 部分匹配的情况
    print("\n测试用例2: 部分匹配")
    refs = [['我', '爱', '你'], ['我', '喜欢', '你']]
    candidate = ['我', '爱', '他']
    score = belu(refs, candidate)
    print(f"参考: {refs}")
    print(f"候选: {candidate}")
    print(f"BLEU分数: {score:.4f}")
    print(f"期望: 分数应该在0.5-0.8之间 (部分匹配)")
    
    # 测试用例3: 完全不匹配的情况
    print("\n测试用例3: 部分匹配 (有1个词相同)")
    refs = [['我', '爱', '你'], ['我', '喜欢', '你']]
    candidate = ['他', '恨', '我']
    score = belu(refs, candidate)
    print(f"参考: {refs}")
    print(f"候选: {candidate}")
    print(f"BLEU分数: {score:.4f}")
    print(f"期望: 分数应该反映部分匹配 (约0.76)")
    
    # 测试用例3.5: 真正的完全不匹配 (没有任何词汇相同)
    print("\n测试用例3.5: 真正的完全不匹配")
    refs = [['我', '爱', '你'], ['我', '喜欢', '你']]
    candidate = ['他', '恨', '她']  # 没有任何词与参考句子相同
    score = belu(refs, candidate)
    print(f"参考: {refs}")
    print(f"候选: {candidate}")
    print(f"BLEU分数: {score:.4f}")
    print(f"期望: 分数应该为0 (完全不匹配)")
    
    # 测试用例4: 候选句子比参考句子短
    print("\n测试用例4: 候选句子较短")
    refs = [['我', '爱', '你', '很', '多'], ['我', '喜欢', '你', '很', '多']]
    candidate = ['我', '爱', '你']
    score = belu(refs, candidate)
    print(f"参考: {refs}")
    print(f"候选: {candidate}")
    print(f"BLEU分数: {score:.4f}")
    print(f"期望: 分数应该受到长度惩罚")
    
    # 测试用例5: 候选句子比参考句子长
    print("\n测试用例5: 候选句子较长")
    refs = [['我', '爱', '你'], ['我', '喜欢', '你']]
    candidate = ['我', '爱', '你', '很', '多', '很', '多']
    score = belu(refs, candidate)
    print(f"参考: {refs}")
    print(f"候选: {candidate}")
    print(f"BLEU分数: {score:.4f}")
    print(f"期望: 分数应该受到长度惩罚")

    # 测试用例6: 自定义权重
    print("\n测试用例7: 自定义权重")
    refs = [['我', '爱', '你'], ['我', '喜欢', '你']]
    candidate = ['我', '爱', '你']
    custom_weights = [0.4, 0.3, 0.2, 0.1]  # 更重视1-gram和2-gram
    score = belu(refs, candidate, custom_weights)
    print(f"参考: {refs}")
    print(f"候选: {candidate}")
    print(f"自定义权重: {custom_weights}")
    print(f"BLEU分数: {score:.4f}")
    
    print("\nBLEU测试完成!")

if __name__ == "__main__":
    belu_test()
    