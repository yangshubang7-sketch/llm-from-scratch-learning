# ========== GPT 模型 ==========
# 目标：将词嵌入、位置编码、多个 TransformerBlock、层归一化、输出层组合成完整的 GPT 模型
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self , d_in , d_out , num_heads , bias = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.W_q_list = nn.ModuleList([nn.Linear(self.d_in , self.d_out , bias = bias) for _ in range(self.num_heads)])
        self.W_k_list = nn.ModuleList([nn.Linear(self.d_in , self.d_out , bias = bias) for _ in range(self.num_heads)])
        self.W_v_list = nn.ModuleList([nn.Linear(self.d_in , self.d_out , bias = bias) for _ in range(self.num_heads)])

    def forward(self, x):
        # x 形状: (batch, seq_len, d_in)
        batch, seq_len, _ = x.shape
        outputs = []
        for h in range(self.num_heads):
            W_q = self.W_q_list[h]
            W_k = self.W_k_list[h]
            W_v = self.W_v_list[h]
            queries = W_q(x)  # (batch, seq_len, d_out)
            keys = W_k(x)
            values = W_v(x)
            attn_scores = queries @ keys.transpose(-2, -1)  # (batch, seq_len, seq_len)
            attn_scores = attn_scores / (self.d_out ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            head_output = attn_weights @ values  # (batch, seq_len, d_out)
            outputs.append(head_output)
        concat_output = torch.cat(outputs, dim=-1)  # (batch, seq_len, d_out * num_heads)
        return concat_output

class FeedForward(nn.Module):

    def __init__(self , emb_dim , hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * emb_dim
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self , x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self , emb_dim , num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_in=emb_dim, d_out=emb_dim // num_heads, num_heads=num_heads)
        self.ffn = FeedForward(emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self , x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + shortcut
        return x

# 1. 定义 GPTModel 类，继承 nn.Module
#    __init__ 参数：vocab_size, emb_dim, num_heads, num_layers, max_seq_len, dropout_rate (可选)
#    - 保存参数到 self
#    - 创建 token_embedding = nn.Embedding(vocab_size, emb_dim)
#    - 创建 position_embedding = nn.Embedding(max_seq_len, emb_dim)
#    - 创建 dropout 层：self.dropout = nn.Dropout(dropout_rate)  （如果 dropout_rate>0）
#    - 创建多个 TransformerBlock 的列表：self.blocks = nn.ModuleList([TransformerBlock(emb_dim, num_heads) for _ in range(num_layers)])
#    - 创建最终的层归一化 self.norm = nn.LayerNorm(emb_dim)
#    - 创建输出线性层 self.out = nn.Linear(emb_dim, vocab_size)
class GPTModule(nn.Module):

    def __init__(self , vocab_size , emb_dim , num_heads , num_layers , max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size , emb_dim)
        self.position_embedding = nn.Embedding(max_seq_len , emb_dim)
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, vocab_size)
# 2. forward 方法：
#    输入：token_ids 形状 (batch, seq_len)
#    - 获取 batch, seq_len = token_ids.shape
#    - 计算 token_emb = self.token_embedding(token_ids)  # (batch, seq_len, emb_dim)
#    - 生成位置索引 positions = torch.arange(seq_len, device=token_ids.device)  # (seq_len,)
#    - 计算 pos_emb = self.position_embedding(positions)  # (seq_len, emb_dim)
#    - 将 token_emb + pos_emb 广播相加（pos_emb 会自动广播到 batch 维度）
#    - x = self.dropout(x)  # 可选
#    - 依次通过每个 TransformerBlock: for block in self.blocks: x = block(x)
#    - x = self.norm(x)
#    - logits = self.out(x)  # (batch, seq_len, vocab_size)
#    - 返回 logits
    def forward(self , token_ids):
        batch, seq_len = token_ids.shape
        token_emb = self.token_embedding(token_ids)
        positions = torch.arange(seq_len)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        for block in self.blocks: x = block(x)
        x = self.norm(x)
        logits = self.out(x)
        return logits
# 3. 测试：创建小模型，输入随机 token IDs，打印输出形状
#    例如：vocab_size=1000, emb_dim=64, num_heads=4, num_layers=2, max_seq_len=32
#    随机输入 token_ids = torch.randint(0, vocab_size, (2, 16))
#    前向传播，打印 logits.shape
# ========== 测试 GPTModule ==========
if __name__ == "__main__":
    # 超参数设置
    vocab_size = 1000      # 词汇表大小
    emb_dim = 64           # 嵌入维度
    num_heads = 4          # 多头注意力头数
    num_layers = 2         # Transformer 块堆叠层数
    max_seq_len = 32       # 最大序列长度（位置编码表大小）
    batch_size = 2
    seq_len = 16

    # 创建模型实例
    model = GPTModule(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )

    # 生成随机 token IDs (batch, seq_len)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits = model(token_ids)

    # 打印输出形状
    print(f"输入 token_ids 形状: {token_ids.shape}")
    print(f"输出 logits 形状: {logits.shape}")   # 应为 (2, 16, 1000)