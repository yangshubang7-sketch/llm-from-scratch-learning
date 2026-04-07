# 任务2：将文本片段拆分成单词和标点符号的列表（tokens）

with open("text.txt", "r" , encoding="utf-8") as f:
    raw_text = f.read()
# 1. 导入 re 模块（正则表达式）
import re
# 2. 取 raw_text 的前 100 个字符作为样例，赋值给 sample_text
#    （注意：raw_text 是上一步已经读取的文本内容）
sample_text = raw_text[:100]
# 3. 定义正则表达式模式（用于匹配标点、破折号和空白字符）：
#    模式字符串为 r'([,.:;?_!"()\']|--|\s)'
#    含义：匹配逗号、句号、冒号、分号、问号、下划线、感叹号、双引号、括号、单引号、两个连字符、空白字符
pattern = r'([,.:;?_!"()\']|--|\s)'
# 4. 使用 re.split(模式, sample_text) 进行拆分，结果赋值给 tokens_raw
tokens_raw = re.split(pattern , sample_text)
# 5. 打印 tokens_raw，观察包含空白的拆分结果
print(tokens_raw)
# 6. 去除空白字符：遍历 tokens_raw，对每个 token 调用 strip() 去除首尾空白，
#    只保留 strip() 后不为空字符串的 token，生成新列表 tokens
#    （提示：可以用列表推导式 [t.strip() for t in tokens_raw if t.strip()]）
tokens = [t.strip() for t in tokens_raw if t.strip()]
# 7. 打印 tokens 列表
print(tokens)
# 8. 打印 tokens 列表的长度
print(len(tokens))


# ========== 任务3：构建词汇表 ==========

# 1. 使用原始 tokens 列表（不是被覆盖后的）获取唯一 token 并排序，赋值给 unique_token
unique_tokens = sorted(list(set(tokens)))
#    注意：如果之前 tokens 已经被覆盖，请重新运行任务2来得到原始的 tokens 列表
# 2. 构建词汇表字典：vocab = {token: idx for idx, token in enumerate(unique_tokens)}
vocab = {token : idx for idx , token in enumerate(unique_tokens)}
# 3. 打印 len(vocab) 和 vocab 的前几项（可选）
print(len(vocab))

# ========== 任务4：编码和解码函数 ==========

# 1. 构建反向映射 idx_to_token：遍历 vocab.items()，将 (token, idx) 反转为 {idx: token}
idx_to_token = {idx : token for token , idx in vocab.items()}
# 2. 定义 encode(text, vocab, pattern)：
#    2.1 使用 re.split(pattern, text) 得到 tokens_raw
#    2.2 过滤掉空字符串，保留其他所有token：tokens = [t for t in tokens_raw if t != ""]
#    2.3 创建一个空列表 ids
#    2.4 遍历 tokens 中的每个 token：
#         - 如果 token 在 vocab 中，将 vocab[token] 添加到 ids
#         - 否则，将 -1 添加到 ids
#    2.5 返回 ids
def encode(text, vocab, pattern):
    tokens_raw = re.split(pattern, text)
    tokens_raw = [t.strip() for t in tokens_raw if t.strip()]
    ids = []
    for token in tokens_raw:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(unk_id)
    return ids
# 3. 定义 decode(ids, idx_to_token)：
#    3.1 创建一个空列表 tokens
#    3.2 遍历 ids 中的每个 id：
#         - 将 idx_to_token[id] 添加到 tokens
#    3.3 将 tokens 列表中的所有字符串拼接成一个字符串（直接拼接，因为空格已在 token 中）并返回
def decode(ids , idx_to_token):
    tokens = []
    for id in ids:
        tokens.append(idx_to_token[id])
    unique_string  = " ".join(tokens)
    return unique_string
# 4. 测试：
#    test_text = "I HAD always thought"
#    ids = encode(test_text, vocab, pattern)
#    print("编码结果:", ids)
#    decoded = decode(ids, idx_to_token)
#    print("解码结果:", decoded)
test_text = "I HAD always thought"
ids = encode(test_text, vocab, pattern)
print("编码结果:", ids)
decoded = decode(ids, idx_to_token)
print("解码结果:", decoded)


# ========== 任务5：添加特殊标记 <|unk|> 和 <|endoftext|> ==========

# 1. 获取当前词汇表的最大 ID 值（max_id = max(vocab.values())）
max_id = max(vocab.values())
# 2. 在 vocab 字典中添加新键 "<|unk|>"，值为 max_id + 1
vocab["<|unk|>"] = max_id + 1
# 3. 在 vocab 字典中添加新键 "<|endoftext|>"，值为 max_id + 2
vocab["<|endoftext|>"] = max_id + 2
# 4. 更新反向映射 idx_to_token：加入这两个新标记（ID -> 标记字符串）
idx_to_token[max_id + 1] = "<|unk|>"
idx_to_token[max_id + 2] = "<|endoftext|>"
# 5. 定义 unk_id = vocab["<|unk|>"]
unk_id = vocab["<|unk|>"]
# 6. 修改 encode 函数：将原来添加 -1 的地方改为添加 unk_id
#    注意：ids.append(-1) 改为 ids.append(unk_id)

# 7. 测试：对包含未知词的文本（例如 "Hello world"）进行编码，打印编码结果
#    预期：未知词对应的 ID 应为 unk_id

# 8. 测试解码：将编码结果解码，观察输出是否显示 "<|unk|>"
test_text_2 = "Hello world"
ids_2 = encode(test_text_2, vocab, pattern)
print("编码结果:", ids_2)
decoded_2 = decode(ids_2, idx_to_token)
print("解码结果:", decoded_2)