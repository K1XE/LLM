import numpy as np

def Focal_loss(y_pred, y_true, alpha = 0.20, gamma = 2.0):
    '''
    如果正样本非常少（稀有），你可以给正样本一个较高的 α（比如0.75），让模型更关注正样本。
    '''
    epsilon = 1e-7
    clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    p_t = y_true * clipped + (1 - y_true) * (1 - clipped)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    loss = -alpha_factor * (1 - p_t) ** gamma * np.log(p_t)
    ret = np.mean(loss)
    return ret

def focal_test():
    num_samples = 6
    y_true = np.random.randint(0, 2, (num_samples,))
    y_pred = np.random.rand(num_samples)
    loss = Focal_loss(y_pred, y_true, alpha=0.2, gamma=2.05)
    print(loss)
    
if __name__ == "__main__":
    focal_test()
