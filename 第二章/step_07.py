# ========== 2.7 使用滑动窗口进行数据采样 ==========
from torch.nn.functional import embedding

with open("text.txt", "r" , encoding="utf-8") as f:
    raw_text = f.read()
text = raw_text
# 1. 使用 tiktoken 获取 GPT-2 分词器，并对整个 raw_text 进行编码，得到 full_ids
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
full_ids = tokenizer.encode(text)
# 2. 设置滑动窗口参数：
#    max_length = 4   # 每个输入样本的 token 数
#    stride = 1       # 窗口每次移动步长
max_length = 4
stride = 1
# 3. 创建空列表 inputs 和 targets
inputs = []
targets= []
# 4. 用 for 循环 i 从 0 到 len(full_ids)-max_length，步长为 stride：
#    4.1 input_chunk = full_ids[i : i+max_length]
#    4.2 target_chunk = full_ids[i+1 : i+max_length+1]
#    4.3 inputs.append(input_chunk)
#    4.4 targets.append(target_chunk)
for i in range(0 , len(full_ids) - max_length , stride):
    input_chunk = full_ids[i: i + max_length]
    target_chunk = full_ids[i + 1: i + max_length + 1]
    inputs.append(input_chunk)
    targets.append(target_chunk)
# 5. 打印生成的样本数量
print(len(inputs) , len(targets))
# 6. 打印第一个样本的输入 ID、目标 ID，以及用 tokenizer.decode 解码后的文本
#    例如：print("输入文本:", tokenizer.decode(inputs[0]))
#          print("目标文本:", tokenizer.decode(targets[0]))
print("输入文本:", tokenizer.decode(inputs[0]))
print("目标文本:", tokenizer.decode(targets[0]))
# 7. （选做）将上述逻辑封装成 PyTorch Dataset 类，参考书本 GPTDatasetV1


# ========== 2.8 创建词嵌入 ==========
# 目标：将 token ID 转换为稠密向量（嵌入）

# 1. 假设我们已经有一个 token ID 张量（例如从滑动窗口得到的第一个输入 inputs[0]）
#    先将其转换为 PyTorch 张量：import torch; sample_ids = torch.tensor(inputs[0])
import torch

sample_ids = torch.tensor(inputs[0])
# 2. 定义嵌入参数：
#    vocab_size = 50257   # GPT-2 词汇表大小（使用 tiktoken 的分词器词汇量）
#    emb_dim = 256        # 嵌入维度（书中示例较小，可以先用 256）
vocab_size = tokenizer.n_vocab
emb_dim = 256
# 3. 创建嵌入层：embedding_layer = torch.nn.Embedding(vocab_size, emb_dim)
embedding_layer = torch.nn.Embedding(vocab_size , emb_dim)
# 4. 将 token ID 张量传入嵌入层：embedding_layer(sample_ids)
#    输出形状应为 (max_length, emb_dim) 即 (4, 256)
token_embeddings = embedding_layer(sample_ids)
# 5. 打印嵌入向量的形状和前几行数值（可选）
print(token_embeddings)

# ========== 2.9 编码词位置 ==========
# 目标：为序列中的每个位置添加位置信息（绝对位置嵌入）

# 1. 定义最大上下文长度 context_length = max_length（例如 4）
context_length = max_length
# 2. 创建位置嵌入层：pos_embedding_layer = torch.nn.Embedding(context_length, emb_dim)
pos_embedding_layer = torch.nn.Embedding(context_length , emb_dim)
# 3. 生成位置索引：position_ids = torch.arange(context_length)   # 例如 [0, 1, 2, 3]
position_ids = torch.arange(context_length)
# 4. 获取位置嵌入：pos_embeddings = pos_embedding_layer(position_ids)  # 形状 (context_length, emb_dim)
pos_embedding = pos_embedding_layer(position_ids)
# 5. 将词嵌入和位置嵌入相加：input_embeddings = token_embeddings + pos_embeddings
#    注意：PyTorch 会自动广播，确保形状一致 (4, 256)
input_embeddings = pos_embedding + token_embeddings
# 6. 打印最终输入嵌入的形状和第一行（可选）
print(input_embeddings)
print(input_embeddings[: 1])