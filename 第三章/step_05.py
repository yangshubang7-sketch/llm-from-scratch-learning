# ========== 练习：将多头注意力封装成类 ==========
# 目标：定义一个 MultiHeadAttention 类，继承 nn.Module，包含 __init__ 和 forward 方法
import torch
# 1. 导入必要的库（torch, nn）
import torch.nn as nn
# 2. 定义类 MultiHeadAttention，继承 nn.Module
class MultiHeadAttention(nn.Module):
# 3. 在 __init__ 方法中：
#    3.1 调用 super().__init__() 初始化父类
#    3.2 接收参数：d_in（输入维度）、d_out（每个头的输出维度）、num_heads（头的数量）、bias（是否使用偏置，默认 False）
#    3.3 将参数保存到 self 中（例如 self.d_out = d_out, self.num_heads = num_heads）
#    3.4 创建三个 ModuleList：self.W_q_list, self.W_k_list, self.W_v_list
#        每个 ModuleList 中包含 num_heads 个 nn.Linear(d_in, d_out, bias=bias)
    def __init__(self , d_in , d_out , num_heads , bias = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.W_q_list = nn.ModuleList([nn.Linear(self.d_in , self.d_out , bias = bias) for _ in range(self.num_heads)])
        self.W_k_list = nn.ModuleList([nn.Linear(self.d_in , self.d_out , bias = bias) for _ in range(self.num_heads)])
        self.W_v_list = nn.ModuleList([nn.Linear(self.d_in , self.d_out , bias = bias) for _ in range(self.num_heads)])
# 4. 在 forward 方法中：
#    4.1 接收输入 x（形状可以是 (seq_len, d_in) 或 (batch, seq_len, d_in)）
#    4.2 获取输入的形状，确定是否有 batch 维度（如果有 batch，需要处理成适合矩阵乘法的形式）
#    4.3 初始化空列表 outputs = []
#    4.4 循环 for h in range(num_heads)：
#        4.4.1 取出当前头的线性层 W_q, W_k, W_v
#        4.4.2 计算 queries = W_q(x), keys = W_k(x), values = W_v(x)  # 形状 (seq_len, d_out) 或 (batch, seq_len, d_out)
#        4.4.3 计算注意力分数：attn_scores = queries @ keys.transpose(-2, -1)  # 转置最后两个维度
#        4.4.4 缩放：attn_scores = attn_scores / (d_out ** 0.5)
#        4.4.5 （可选）添加因果掩码：生成上三角掩码，将未来位置设为 -inf
#        4.4.6 计算注意力权重：attn_weights = torch.softmax(attn_scores, dim=-1)
#        4.4.7 计算当前头的输出：head_output = attn_weights @ values  # 形状与 queries 相同
#        4.4.8 将 head_output 添加到 outputs 列表
#    4.5 将所有头的输出在最后一维拼接：concat_output = torch.cat(outputs, dim=-1)
#    4.6 （可选）添加输出投影层：self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads) 并在 forward 中应用
#    4.7 返回 concat_output（或投影后的结果）
    def forward(self , x):
        seq_len = x.shape[0]
        outputs = []
        for h in range(self.num_heads):
            W_q = self.W_q_list[h]
            W_k = self.W_k_list[h]
            W_v = self.W_v_list[h]
            queries = W_q(x)
            keys = W_k(x)
            values = W_v(x)
            attn_scores = queries @ keys.transpose(-2 , -1)
            attn_scores = attn_scores / (self.d_out ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            head_output = attn_weights @ values
            outputs.append(head_output)
        concat_output = torch.cat(outputs, dim=-1)
        return concat_output
# 5. 测试你的类：创建实例，输入随机张量，打印输出形状

d_in = 3
d_out = 2
num_heads = 2
muti = MultiHeadAttention(d_in , d_out , num_heads)
inputs = torch.rand(6, d_in)   # 6个token，每个3维
output = muti(inputs)
print(output)


