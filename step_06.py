# ========== 2.5 字节对编码（BPE） ==========
# 目标：使用 tiktoken 库实现 GPT-2 的 BPE 分词器
with open("text.txt" , "r" , encoding="utf-8") as f:
    raw_text = f.read()
text = raw_text
# 1. 安装并导入 tiktoken 库（需要先 pip install tiktoken）
import tiktoken
# 2. 获取 GPT-2 的分词器：tokenizer = tiktoken.get_encoding("gpt2")
tokenizer = tiktoken.get_encoding("gpt2")
# 3. 对一段文本进行编码：ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
ids = tokenizer.encode(text , allowed_special={"<|endoftext|>"})
# 4. 打印编码结果和对应的 token 数量
print(ids)
print(len(ids))
# 5. 解码：decoded = tokenizer.decode(ids)
decoded = tokenizer.decode(ids)
print(decoded)
# 6. 测试未知词：例如 "Akwirw ier"，观察 BPE 如何将其拆分为子词（不产生 <|unk|>）
string = "Akwirw ier"
test_ids = tokenizer.encode(string , allowed_special={"<|endoftext|>"})
test_decoded = tokenizer.decode(test_ids)
print(test_ids)
print(test_decoded)
