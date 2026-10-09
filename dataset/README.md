\# 数据集说明



本目录用于存放训练/验证数据。\*\*数据集本身不提交到 git\*\*（见 `.gitignore`）。



\## 预训练语料



\- 文件名：`pretrain\\\_t2t\\\_mini.jsonl`

\- 格式：每行一个 JSON 对象

\- 字段：`{"text": "..."}`



示例：



```json

{"text": "人工智能是计算机科学的一个分支..."}

{"text": "中国的历史可以追溯到..."}

```



参考来源：



\- \[匠数大模型数据集](https://www.modelscope.cn/datasets/)

\- \[Magpie-Align](https://huggingface.co/datasets/Magpie-Align)

\- \[COIG](https://huggingface.co/datasets/BAAI/COIG)

\- \[R1-Distill-SFT](https://huggingface.co/datasets/)



\## SFT 对话数据



\- 文件名：`sft\\\_t2t.jsonl` 或 `sft\\\_t2t\\\_mini.jsonl`

\- 格式：每行一个 JSON 对象

\- 字段：`conversations` 数组，支持多轮对话 + Tool Calling



基础对话：



```json

{

\&#x20; "conversations": \\\[

\&#x20;   {"role": "user", "content": "你好"},

\&#x20;   {"role": "assistant", "content": "你好！"}

\&#x20; ]

}

```



带 Tool Calling：



```json

{

\&#x20; "conversations": \\\[

\&#x20;   {"role": "system", "content": "# Tools ...", "tools": "\\\[...]"},

\&#x20;   {"role": "user", "content": "把'你好世界'翻译成english"},

\&#x20;   {"role": "assistant", "content": "", "tool\\\_calls": "\\\[{\\\\"name\\\\":\\\\"translate\\\_text\\\\",\\\\"arguments\\\\":{...}}]"},

\&#x20;   {"role": "tool", "content": "{\\\\"translated\\\_text\\\\":\\\\"Hello World\\\\"}"},

\&#x20;   {"role": "assistant", "content": "Hello World"}

\&#x20; ]

}

```



\## Tokenizer



放在 `model/` 目录下，包含：



\- `tokenizer.json`

\- `tokenizer\\\_config.json`

\- `vocab.json`

\- `merges.txt`



可以从 \[MiniMind](https://github.com/jingyaogong/minimind) 获取。

