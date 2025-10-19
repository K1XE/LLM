# https://zhuanlan.zhihu.com/p/642569664
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import LlamaForCausalLM, LlamaConfig
from copy import deepcopy

torch.manual_seed(0)

def main():
    beta = 0.1
    policy_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))
    ref_model = deepcopy(policy_model)
    
    prompt_id = [1, 2, 3, 4, 5, 6]
    good_response_ids = [7, 8, 9, 10]
    bad_response_ids_list = [[1, 2, 3, 0], [4, 5, 6, 0]]
    
    input_ids = torch.LongTensor(
        [prompt_id + good_response_ids, *[prompt_id + bad_response_ids for bad_response_ids in bad_response_ids_list]]
    )
    
    labels = torch.LongTensor(
        [[-100] * len(prompt_id) + good_response_ids,
         *[[-100] * len(prompt_id) + bad_response_ids for bad_response_ids in bad_response_ids_list]]
    )[:, 1:] # shift
    
    loss_mask = (labels != -100)
    labels[labels == -100] = 0
    
    logits = policy_model(input_ids)["logits"][:, :-1, :]
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    all_logps = (per_token_logps * loss_mask).sum(-1)
    policy_good_logps, policy_bad_logps = all_logps[:1], all_logps[1:]
    with torch.no_grad():
        logits: Tensor = ref_model(input_ids)["logits"][:, :-1, :]
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        all_logps = (per_token_logps * loss_mask).sum(-1)
        ref_good_logps, ref_bad_logps = all_logps[:1], all_logps[1:]
    
    logits = (policy_good_logps - ref_good_logps) - (policy_bad_logps - ref_bad_logps)
    loss = -F.log_softmax(beta * logits).mean()
    print(loss)
    
if __name__ == "__main__":
    main()