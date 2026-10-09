\# 复现指南



\## 环境要求



\- Python 3.11

\- CUDA 12.x（BF16 需要 Ampere 以上 GPU）

\- GPU 显存 ≥ 8 GB



\## 安装



```bash

git clone https://github.com/<your-username>/minimind-kda.git

cd minimind-kda

pip install -r requirements.txt

```



\## 数据准备



1\. 从 \[MiniMind](https://github.com/jingyaogong/minimind) 下载：

&#x20;  - Tokenizer 文件 → 放到 `model/`

&#x20;  - `pretrain\\\_t2t\\\_mini.jsonl` → 放到 `dataset/`

&#x20;  - `sft\\\_t2t.jsonl` → 放到 `dataset/`



2\. 确认目录结构：



```text

minimind-kda/

├── model/

│   ├── model\\\_minimind.py

│   ├── kda\\\_attention\\\_v3.py

│   ├── attention\\\_residual.py

│   ├── tokenizer.json

│   └── tokenizer\\\_config.json

└── dataset/

\&#x20;   ├── pretrain\\\_t2t\\\_mini.jsonl

\&#x20;   └── sft\\\_t2t.jsonl

```



\## 复现步骤



\### Step 1: 基础正确性



```bash

python tests/test21\\\_basic\\\_forward.py

```



预期：所有检查通过，梯度有限，参数更新正常。



\### Step 2: Chunked 数值等价



```bash

python tests/test23\\\_chunked\\\_kda\\\_equivalence.py

```



预期：



```

chunk\\\_size=64:  diff \\\~2e-7, speedup \\\~8×

chunk\\\_size=32:  diff \\\~2e-7, speedup \\\~4×

```



\*\*注意\*\*：`chunk\\\_size=128` 在 fp32 下会失败，这是已知的 subnormal 陷阱。



\### Step 3: 真实文本架构对照



```bash

python tests/test22\\\_real\\\_text\\\_comparison.py

```



耗时约 15 分钟（3 个变体 × 200 步）。



预期结果：



| 变体 | Val Loss |

|---|---:|

| Baseline | \~5.69 |

| KDA V3 | \~4.77 |



\### Step 4: 缓存一致性



```bash

python tests/test29\\\_bf16\\\_cache\\\_regression.py

```



预期：所有 prefill 长度在 FP32 对照下 `argmax=100%`。



\### Step 5: 真实语料训练



```bash

python tests/test34\\\_train\\\_real\\\_corpus.py

```



耗时约 20 分钟（3 个变体 × 100 步）。



预期结果：



| 变体 | Val PPL |

|---|---:|

| Baseline | \~1500 |

| KDA V3 | \~930 |



\### Step 6: 全量验证



```bash

python tests/test35\\\_full\\\_validation\\\_benchmark.py

```



耗时约 3 分钟（加载 checkpoint + 全量验证 + 推理 benchmark）。



预期结果：



| 变体 | Val PPL | Accuracy |

|---|---:|---:|

| Baseline | \~1568 | 5.30% |

| KDA V3 | \~943 | 9.64% |



\### Step 7: SFT 训练



```bash

python tests/test37\\\_sft\\\_training.py

```



耗时约 40 分钟（KDA 单变体 2000 步）。



预期结果：



| 变体 | Val Loss | NaN skips |

|---|---:|---:|

| KDA V3 | \~4.74 | \*\*0\*\* |



\## 常见问题



\### Q: `RuntimeError: Validation input device mismatch: cuda:0`



\*\*A\*\*: `torch.device("cuda") != torch.device("cuda:0")`。确保 `DEVICE = torch.device("cuda:0")`，并在比较时用 `next(model.parameters()).device`。



\### Q: `UserWarning: Mismatch dtype between input and weight`



\*\*A\*\*: BF16 autocast 下 RMSNorm 权重是 fp32，输入是 bf16。在 `nn.RMSNorm` 调用前显式 `x.float()`。



\### Q: KDA 训练在 800 步左右崩溃



\*\*A\*\*: 参数进入永久性坏区。检查：



\- `max\\\_state\\\_norm` 是否 ≤ 20

\- `diag\\\_damping` 是否 ≥ 1.05

\- `LR` 是否 ≤ 2e-5



\### Q: chunk\_size=128 时 chunked 输出错误



\*\*A\*\*: 已知 fp32 subnormal 陷阱。`D\\\_i / D\\\_{j+1}` 在 `chunk\\\_size > 64` 时掉入 subnormal 区。使用 `chunk\\\_size ∈ {32, 64}`。



\### Q: 显存不足



\*\*A\*\*: 减小 `BATCH\\\_SIZE` 或 `MAX\\\_SEQ\\\_LEN`，或增加 `GRAD\\\_ACCUM\\\_STEPS` 补偿。

