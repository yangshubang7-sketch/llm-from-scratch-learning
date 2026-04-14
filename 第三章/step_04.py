# ========== 3.5 因果注意力（带掩码） ==========
# 目标：在注意力分数矩阵上添加一个上三角掩码，使每个 token 只能关注它自己和之前的 token
import  torch
# 1. 沿用之前的 inputs 张量（6个token，每个3维）作为示例（也可以使用您自己的嵌入矩阵）
#    inputs = torch.tensor([...])  与之前相同
inputs = torch.rand(6 , 3)
# 2. 为了模拟实际场景，我们假设已经通过线性层得到了查询 Q 和键 K（形状都是 (6, d_out)）
#    为简单起见，直接用 inputs 作为 Q 和 K（无投影），d_out=3
#    所以 attn_scores = inputs @ inputs.T  形状 (6,6)
attn_scores = inputs @ inputs.T
# 3. 定义序列长度 seq_len = 6
#    创建一个上三角掩码矩阵 mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
#    diagonal=1 表示从第一条对角线上方开始置1（不包含对角线）
#    mask 中上三角部分（不含对角线）为 1，其余为 0
seq_len = 6
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
# 4. 将 mask 中的 1 替换为 -inf（负无穷大），0 替换为 0（或保持原值）
#    使用 masked_fill：attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
# 5. 对掩码后的注意力分数做 softmax（dim=1）
#    attn_weights = torch.softmax(attn_scores, dim=1)
attn_weights = torch.softmax(attn_scores, dim=1)
# 6. 打印掩码后的注意力权重矩阵，观察每一行中未来位置的概率是否为 0
print(attn_weights)
# 7. 计算上下文向量：context_vecs = attn_weights @ values（这里 values 可用 inputs 代替）
context_vecs = attn_weights @ inputs
# 8. 打印上下文向量的形状（应为 (6,3)）
print(context_vecs)