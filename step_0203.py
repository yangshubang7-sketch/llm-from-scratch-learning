# 任务2：将文本片段拆分成单词和标点符号的列表（tokens）

with open("text.txt" , "r" , encoding="utf-8") as f:
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


# ========== 任务3：构建词汇表（2.3节 第24-26页） ==========

# 1. 从 tokens 中获取所有唯一的 token，并排序
tokens = sorted(list(set(tokens)))
# 2. 构建词汇表字典：{token: 整数ID}
vocab = {s : i for i , s in enumerate(tokens)}
# 3. 打印词汇表大小
print(len(tokens))