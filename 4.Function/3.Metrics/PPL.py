import numpy as np

def calculate_ppl(log_probs):
    output = np.exp(-np.mean(log_probs))
    return output

def perplexity_test():
    log_probs = -np.random.random(5)
    output = calculate_ppl(log_probs)
    return output

if __name__ == "__main__":
    perplexity_test()
    