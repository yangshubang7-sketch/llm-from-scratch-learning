# 任务1：从网络下载《The Verdict》短篇小说文本，并读取内容
# 1. 导入 urllib.request 模块（用于下载文件）
import urllib.request
# 2. 定义文件的 URL 地址（使用下面这个链接）：
#    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
ur1 = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
# 3. 定义本地保存的文件名，比如 "the-verdict.txt"
file_path = "text.txt"
# 4. 使用 urllib.request.urlretrieve(url, 本地文件名) 下载文件
urllib.request.urlretrieve(ur1 , file_path)
# 5. 使用 open() 函数以只读模式打开文件，编码为 "utf-8"，并用变量 f 表示
with open("text.txt", "r" , encoding="utf-8") as f:
# 6. 读取文件全部内容，赋值给变量 raw_text
    raw_text = f.read()
# 7. 打印 raw_text 的字符总数（用 len()）
print(len(raw_text))
# 8. 打印 raw_text 的前 200 个字符（切片 [:200]）
print(raw_text[:200])