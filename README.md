<div align="center">
  <h1 style="margin-bottom: 0;">LLM & RL手撕记录</h1>
  <p>
    ♾️<a href="https://hwcoder.top/Manual-Coding-1" target="_blank">Large Language Model Algorithm</a>
  </p>
  <p>
    ♾️<a href="https://www.jackzhu.top/categories/%E9%9D%A2%E8%AF%95%E8%AE%B0%E5%BD%95/" target="_blank">Jack's Blog</a>
  </p>
  <p>
    ♾️<a href="https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py" target="_blank">veRL-core-algos</a>
  </p>
  <p>
    ♾️<a href="https://github.com/boyu-ai/Hands-on-RL" target="_blank">Hands-on-RL
</a>
  </p>
  <p>
    ♾️<a href="https://pcnik3wco47l.feishu.cn/wiki/RZsswSD1eiK7vykbgy8cHfSsnaf" target="_blank">手撕小记</a>
  </p>
</div>

<div align="center">

### ⚡️ LLM 手撕经典算法



| 序号 | 篇章                                                      | 算法                                                                                                 |
| :----: | :-------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------- |
| 1    | Attention            | <ul><li>✔️ Scaled Dot-Product Attention</li><li>✔️ Multi-Head Attention (MHA)</li><li>✔️ Multi-Query Attention (MQA)</li><li>✔️ Grouped Query Attention (GQA)</li><li>✔️ Multi-head Latent Attention (MLA)</li><li>⬜ Flash Attention (FA)</li></ul> |
| 2    | nn             | <ul><li>✔️ Layer Normalization (LN)</li><li>✔️ RMSNorm</li><li>✔️ Batch Normalization (BN)</li><li>✔️ Dropout</li><li>✔️ Backpropagation</li><li>✔️ Gradient Accumulation</li></ul> |
| 3    | Transformer          | <ul><li>✔️ Token Embedding</li><li>✔️ Positional Embedding</li><li>✔️ RoPE</li><li>✔️ MoE</li><li>✔️ Encoder Layer (MHA + FFN)</li><li>✔️ Decoder Layer</li><li>✔️ Stacked Encoder/Decoder</li><li>✔️ Transformer (MHA\SinCos)</li><li>✔️ Transformer (GQA\RoPE)</li></ul> |
| 4    | Function             | <ul><li>✔️ MSE</li><li>✔️ CE, BCE, KL, Focal</li><li>✔️ Sigmoid, Tanh</li><li>✔️ ReLU, Leaky ReLU, ELU</li><li>✔️ Swish, GeLU, SwiGLU</li><li>✔️ Softmax</li><li>✔️ PPL, ROUGE, BLEU</li></ul> |
| 5    | ML            | <ul><li>✔️ 线性回归</li><li>✔️ 逻辑回归</li><li>✔️ Softmax 回归</li></ul> |
| 6    | RLHF                 | <ul><li>✔️ 广义优势估计 (GAE)</li><li>✔️ PPO Loss & Value Loss</li><li>✔️ DPO Loss</li><li>⬜ GRPO Loss</li><li>✔️ 无偏 KL 散度</li><li>✔️ 三种 KL 散度估计器</li></ul> |
| 7    | veRL                 | <ul><li>✔️ agg_loss</li><li>✔️ compute_policy_loss_vanilla</li><li>✔️ kl_penalty</li><li>✔️ compute_gae_advantage_return</li><li>✔️ compute_value_loss</li><li>✔️ compute_entropy_loss</li></ul> |