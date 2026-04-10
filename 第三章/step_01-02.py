# 第3章 任务1：简单自注意力（无训练权重）
# 目标：计算第二个 token (x2) 与其他所有 token 的注意力分数和上下文向量

# 1. 导入 torch
import torch
# 2. 定义输入张量 inputs，形状 (6, 3)（6个token，每个3维）
#    数值来自书本第56页：
#    inputs = torch.tensor([
#        [0.43, 0.15, 0.89],  # x1
#        [0.55, 0.87, 0.66],  # x2
#        [0.57, 0.85, 0.64],  # x3
#        [0.22, 0.58, 0.33],  # x4
#        [0.77, 0.25, 0.10],  # x5
#        [0.05, 0.80, 0.55]   # x6
#    ])
inputs = torch.tensor([[0.43, 0.15, 0.89] , [0.55, 0.87, 0.66] , [0.57, 0.85, 0.64] , [0.22, 0.58, 0.33] , [0.77, 0.25, 0.10] , [0.05, 0.80, 0.55]])
# 3. 取第二个 token (索引1) 作为查询向量 query ps:这里 我先实现了 用第二个词 然后 最终使用for循环 实现了 六个词
for j in range(len(inputs)):
    query = inputs[j]
    # 4. 初始化空张量 attn_scores = torch.empty(6)，用来存储6个注意力分数
    attn_scores = torch.empty(6)
    # 5. 用 for 循环 i 从0到5：
    #    计算 inputs[i] 与 query 的点积 (torch.dot)，存入 attn_scores[i]
    for i in range(0 , 6):
        attn_scores[i] = torch.dot(inputs[i] , query)
    # 6. 打印 attn_scores
    print(attn_scores)
    # 7. 用 torch.softmax 将 attn_scores 归一化，得到注意力权重 attn_weights，dim=0
    attn_weights = torch.softmax(attn_scores , dim = 0)
    # 8. 打印 attn_weights，并验证其总和为1 (用 sum() 方法)
    print(attn_weights , sum(attn_weights))
    # 9. 计算上下文向量 context_vec：
    #    将 attn_weights 变成形状 (6,1) 以便与 inputs 逐元素相乘 (attn_weights.unsqueeze(-1))
    #    然后按第0维求和 (dim=0)
    attn_weights = (attn_weights.unsqueeze(-1))
    context_vec = torch.sum(attn_weights * inputs , dim = 0)
    # 10. 打印 context_vec
    print(context_vec)



# ========== 任务2：矩阵化自注意力（无训练权重） ==========
# 目标：一次性计算所有 token 的上下文向量

# 1. 沿用之前的 inputs 张量（6个token，每个3维）

# 2. 计算注意力分数矩阵：attn_scores = inputs @ inputs.T
#    inputs 形状 (6,3)，转置后 (3,6)，相乘得到 (6,6)
#    其中 attn_scores[i][j] 表示第 i 个 token 作为查询与第 j 个 token 的点积
attn_scores = inputs @ inputs.T
# 3. 打印 attn_scores 的形状（应为 (6,6)）和第一行（可选）
print(attn_scores)
# 4. 用 softmax 对每一行归一化，得到注意力权重矩阵 attn_weights
#    注意：dim=1 表示沿着行方向（即对每个查询，对所有的键做 softmax）
#    attn_weights = torch.softmax(attn_scores, dim=1)
attn_weights = torch.softmax(attn_scores , dim =1)
# 5. 打印 attn_weights 的形状，并验证第一行的和为1
print(attn_weights)
# 6. 计算上下文向量矩阵：context_vecs = attn_weights @ inputs
#    形状：(6,6) @ (6,3) → (6,3)
#    每一行就是对应 token 的上下文向量
context_vecs = attn_weights @ inputs
# 7. 打印 context_vecs 的形状
#    可以打印第一个 token 的上下文向量，与之前手动计算的结果对比（注意之前只算了第二个）
print(context_vecs)