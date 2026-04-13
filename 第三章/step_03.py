# ========== 3.4 带可训练权重的自注意力（缩放点积注意力） ==========
# 目标：引入可训练的权重矩阵 W_q, W_k, W_v，将输入投影到查询、键、值空间
import torch
# 1. 沿用之前的 inputs 张量（6个token，每个3维），并定义输入维度 d_in = 3，输出维度 d_out = 2
#    提示：d_out 可以任意设定，这里用 2 便于观察
d_in = 3
d_out = 2
inputs = torch.rand(6 , d_in)
# 2. 导入 torch.nn as nn
import torch.nn as nn
# 3. 定义三个线性层（无偏置）：
#    W_q = nn.Linear(d_in, d_out, bias=False)
#    W_k = nn.Linear(d_in, d_out, bias=False)
#    W_v = nn.Linear(d_in, d_out, bias=False)
W_q = nn.Linear(d_in, d_out, bias=False)
W_k = nn.Linear(d_in, d_out, bias=False)
W_v = nn.Linear(d_in, d_out, bias=False)
# 4. 计算查询、键、值矩阵：
#    queries = W_q(inputs)   # 形状 (6, d_out)
#    keys    = W_k(inputs)   # (6, d_out)
#    values  = W_v(inputs)   # (6, d_out)
queries =  W_q(inputs)
keys    = W_k(inputs)
values  = W_v(inputs)
# 5. 计算注意力分数：attn_scores = queries @ keys.T   # 形状 (6,6)
attn_scores = queries @ keys.T
# 6. 缩放：attn_scores = attn_scores / (d_out ** 0.5)
attn_scores = attn_scores / (d_out ** 0.5)
# 7. softmax 归一化得到注意力权重（dim=1，对每个查询的行）
#    attn_weights = torch.softmax(attn_scores, dim=1)
attn_weights =  torch.softmax(attn_scores, dim=1)
# 8. 计算上下文向量：context_vecs = attn_weights @ values   # 形状 (6, d_out)
context_vecs = attn_weights @ values
# 9. 打印 context_vecs 的形状和第一行（或全部）
print(context_vecs)