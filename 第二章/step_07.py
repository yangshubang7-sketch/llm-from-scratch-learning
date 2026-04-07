# ========== 2.6 使用滑动窗口进行数据采样 ==========
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