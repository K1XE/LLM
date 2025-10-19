import numpy as np

def rouge_l(ref, candidate):
    n1, n2 = len(ref), len(candidate)
    # dp[i][j] 表示 ref[:i] 和 candidate[:j] 的最长公共子序列长度
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ref[i - 1] == candidate[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n1][n2] 
    
    # 处理边界情况：当没有公共子序列时
    if lcs == 0:
        return 0.0
    
    precision = lcs / n2
    recall = lcs / n1
    beta = 1.2
    
    # 处理分母为0的情况
    denominator = recall + beta ** 2 * precision
    if denominator == 0:
        return 0.0
    
    output = (1 + beta ** 2) * recall * precision / denominator
    return output

def rouge_l_test():
    # 测试用例1：完全匹配
    ref1 = "the cat is on the mat"
    candidate1 = "the cat is on the mat"
    print(f"测试1 - 完全匹配:")
    print(f"参考文本: {ref1}")
    print(f"候选文本: {candidate1}")
    score1 = rouge_l(ref1.split(), candidate1.split())
    print(f"ROUGE-L 分数: {score1:.4f}")
    print()
    
    # 测试用例2：部分匹配
    ref2 = "the cat is on the mat"
    candidate2 = "the cat is under the table"
    print(f"测试2 - 部分匹配:")
    print(f"参考文本: {ref2}")
    print(f"候选文本: {candidate2}")
    score2 = rouge_l(ref2.split(), candidate2.split())
    print(f"ROUGE-L 分数: {score2:.4f}")
    print()
    
    # 测试用例3：完全不匹配
    ref3 = "the cat is sleeping"
    candidate3 = "dogs are running fast"
    print(f"测试3 - 完全不匹配:")
    print(f"参考文本: {ref3}")
    print(f"候选文本: {candidate3}")
    score3 = rouge_l(ref3.split(), candidate3.split())
    print(f"ROUGE-L 分数: {score3:.4f}")
    print()
    
    # 测试用例4：长度不同
    ref4 = "this is a test"
    candidate4 = "this is a very long test sentence with many words"
    print(f"测试4 - 长度不同:")
    print(f"参考文本: {ref4}")
    print(f"候选文本: {candidate4}")
    score4 = rouge_l(ref4.split(), candidate4.split())
    print(f"ROUGE-L 分数: {score4:.4f}")
    print()
    
    # 测试用例5：顺序不同
    ref5 = "the quick brown fox"
    candidate5 = "fox brown quick the"
    print(f"测试5 - 顺序不同:")
    print(f"参考文本: {ref5}")
    print(f"候选文本: {candidate5}")
    score5 = rouge_l(ref5.split(), candidate5.split())
    print(f"ROUGE-L 分数: {score5:.4f}")

if __name__ == "__main__":
    rouge_l_test()