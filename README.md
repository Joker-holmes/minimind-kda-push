\# MiniMind-KDA



在 \[MiniMind](https://github.com/jingyaogong/minimind) 上实现并对照评估 \*\*Kimi Delta Attention (KDA) V3\*\* —— 一种基于 delta-rule 的线性注意力机制。



在真实中文语料上完成 9 组受控实验，覆盖架构对比、数值稳定性、缓存一致性、生成质量与 SFT 训练全链路。



\## 亮点



\- \*\*KDA V3 vs 标准注意力\*\*：在 27K 真实预训练语料上，验证 loss 低 \*\*0.48\*\*，PPL 低 \*\*40%\*\*，next-token accuracy 从 \*\*5.30% 翻倍至 9.64%\*\*

\- \*\*Chunked 并行化\*\*：把 KDA 的串行时间步递推改写成分块并行形式，单算子加速 \*\*8.4×\*\*，端到端训练差距从 31× 压到 \*\*1.8×\*\*

\- \*\*数值稳定性\*\*：通过 fp64 累积衰减、state 范数裁剪、对角线阻尼三项修复，SFT 训练从 step 847 崩溃变为 \*\*2000 步全程 0 NaN skip\*\*

\- \*\*缓存一致性\*\*：FP32 对照路径下，prefill + incremental decode 与全序列前向的 \*\*argmax 一致率 100%\*\*，最大误差 `< 5e-6`



\## 核心结果



\### Pretrain 对照（TEST 34/35）



在 27,000 条真实预训练记录、5.5M tokens 上训练 100 步：



| 变体 | 参数量 | Val Loss | Val PPL | Accuracy | 推理吞吐 | 显存 |

|---|---:|---:|---:|---:|---:|---:|

| Baseline Attention | 34.4M | 7.357 | 1568 | 5.30% | 42,029 tok/s | 179 MB |

| \*\*KDA V3\*\* | 36.9M | \*\*6.849\*\* | \*\*943\*\* | \*\*9.64%\*\* | 15,965 tok/s | 189 MB |

| KDA V3 + AttnRes | 46.3M | 6.844 | 938 | 9.45% | 14,913 tok/s | 225 MB |



\### SFT 训练（TEST 37）



在 `sft\_t2t.jsonl`（60K 条对话）上训练 2000 步：



| 变体 | LR | Val Loss | Val PPL | NaN skips |

|---|---:|---:|---:|---:|

| Baseline Attention | 5e-5 | \*\*3.944\*\* | \*\*51.6\*\* | 0 |

| KDA V3 | 2e-5 | 4.743 | 114.8 | \*\*0\*\* |



> KDA 需要降低学习率以保持数值稳定。稳定性修复的代价是收敛速度，可通过后续的 chunk 级数值优化缩小。



\## 架构



```

                    ┌────────────────────────────┐

                  │      Input Hidden States    │

                    └──────────────┬─────────────┘

                                   │

            ┌─────────────────────┴─────────────────────┐

             │                                           │

   ┌─────────▼─────────┐                     ┌───────────▼───────────┐

   │  Standard Attention│                     │       KDA V3           │

   │  (QKV proj → SDPA) │                     │  (Conv → Proj → Norm   │

   │                    │                     │   → Chunked Delta Rule)│

   └─────────┬──────────┘                     └───────────┬───────────┘

             │                                           │

             └─────────────────────┬─────────────────────┘

                                  │

                    ┌──────────────▼─────────────┐

                    │         MLP (SwiGLU)        │

                    └──────────────┬─────────────┘

                                   │

                    ┌──────────────▼─────────────┐

                    │  Block Attention Residual   │

                    │        (optional)           │

                    └──────────────┬─────────────┘

                                   │

                             Output

```



\### KDA V3 内部流程



```

x  ──►  ShortConv1d (Q, K, V)  ──►  Linear Projections

    ──►  RMSNorm + L2 Normalize

    ──►  α, β gates (sigmoid)

    ──►  Chunked Delta-Rule Recurrence

        ┌─────────────────────────────────────────────┐

        │ Per chunk (C=64):                            │

        │   D\_cum = cumprod(α)          \[fp64]         │

        │   M = β ⊙ (K K^T) ⊙ ratio    \[unit lower tri]│

        │   M\_full = M + 1.05 I         \[diagonal damp]│

        │   E = solve(M\_full, V - D S K)\[torch.linalg] │

        │   out = D Q S + A\_intra E                    │

        │   S = D\_C S + (coeff ⊙ K) E^T                │

        │   S ← clip(S, max\_norm=20)   \[state clip]    │

        └─────────────────────────────────────────────┘

    ──►  Output Projection

```



\## 快速开始



\### 环境



```bash

\# 推荐 Python 3.11

pip install -r requirements.txt

```



关键依赖：



```

torch>=2.11.0

transformers>=4.40.0

```



\### 数据准备



```bash

\# 预训练语料（JSONL，每行 {"text": "..."}）

dataset/pretrain\_t2t\_mini.jsonl



\# SFT 对话数据（JSONL，每行 {"conversations": \[...]}）

dataset/sft\_t2t.jsonl



\# Tokenizer

model/

```



\### 复现命令



按顺序运行：



```bash

\# 1. 单元测试

python tests/test21\_basic\_forward.py



\# 2. 真实文本对照（3 个变体 × 200 步）

python tests/test22\_real\_text\_comparison.py



\# 3. Chunked 数值等价验证

python tests/test23\_chunked\_kda\_equivalence.py



\# 4. 分阶段计时（CUDA Event）

python tests/test26\_phase\_timing.py



\# 5. 缓存 + 增量解码回归（BF16 + FP32 对照）

python tests/test29\_bf16\_cache\_regression.py



\# 6. 真实语料训练

python tests/test34\_train\_real\_corpus.py



\# 7. 全量验证 + 推理 benchmark

python tests/test35\_full\_validation\_benchmark.py



\# 8. 生成质量回归

python tests/test36\_generation\_quality.py



\# 9. SFT 训练

python tests/test37\_sft\_training.py

```



所有结果自动写入各 `test\*\_outputs/` 目录下的 `results.json` / `summary.csv`。



\## 测试矩阵



| Test | 目标 | 关键指标 | 状态 |

|---|---|---|---|

| 21 | 前向/反向/优化器 | 梯度有限、参数更新 | ✅ |

| 22 | 小语料架构对照 | loss 差 −0.91 | ✅ |

| 23 | Chunked 数值等价 | 误差 1e-7 | ✅ |

| 26 | 分阶段耗时 | 训练速度 1.8× | ✅ |

| 29 | 缓存 + 增量解码 | FP32 argmax 100% | ✅ |

| 34 | 真实语料训练 | loss 差 −0.48 | ✅ |

| 35 | 全量验证 + 推理 | PPL −40%、acc ×1.8 | ✅ |

| 36 | 生成质量 | KDA 采样 logp 领先 | ✅ |

| 37 | SFT 稳定性 | 2000 步 0 NaN | ✅ |



\## 技术难点



\### 1. Chunked Delta-Rule 的精确推导



Per-timestep 递推：



```

S\_{t+1} = α\_t S\_t - β\_t k\_t k\_t^T S\_t + β\_t k\_t v\_t^T

```



通过定义预测误差 `E\_t = v\_t - k\_t^T S\_t`，可以把整块的递推重写为\*\*下三角线性系统\*\*：



```

E\_i + Σ\_{j<i} (D\_i / D\_{j+1}) β\_j (k\_j · k\_i) E\_j = v\_i - D\_i S\_0^T k\_i

```



一次 `torch.linalg.solve` 解出所有 `E\_i`，串行深度从 `T` 降到 `T/C`。



\### 2. fp32 subnormal 陷阱



`D\_i / D\_{j+1}` 当 `chunk\_size > 64` 时会掉进 fp32 subnormal 区（`2^-127 ≈ 6e-39`），精度只剩 1\~3 位有效数字。修复方法：



\- 累积衰减和比值全部在 \*\*fp64\*\* 里算，只在矩阵乘法前 cast 回 fp32



\### 3. `cuda` vs `cuda:0` 设备比较



PyTorch 里 `torch.device("cuda") != torch.device("cuda:0")`。用 `next(model.parameters()).device` 作为唯一真值来源，避免误报设备不匹配。



\### 4. BF16 autocast 下的 RMSNorm dtype



`nn.RMSNorm` 的权重是 fp32，autocast 下输入变 bf16，触发 "Mismatch dtype" 并回退到非融合实现。修复：在 norm 前显式 `x.float()`。



\### 5. State 范数爆炸与 SFT 崩溃



KDA 的 state 是 `\[B, H, D, D]` 的 fp32 张量，当 `α` gate 接近 1 时会无限增长，`torch.linalg.solve` 的条件数爆炸，反向传播产生 NaN 梯度。修复三项：



\- `max\_state\_norm = 20`（Frobenius 范数裁剪）

\- `diag\_damping = 1.05`（抬高 solve 的最小奇异值）

\- `grad\_clip = 0.5` + `LR = 2e-5`



\## 项目结构



```

minimind-kda/

├── model/

│   ├── model\_minimind.py        # 主干（Config + Attention + KDA 分支）

│   ├── kda\_attention\_v3.py      # KDA V3（chunked + reference）

│   └── attention\_residual.py    # Block Attention Residual

├── tests/                        # 9 组独立测试

├── docs/

│   ├── architecture.md

│   ├── experiment\_results.md

│   └── reproduction.md

├── TECHNICAL\_REPORT.md          # 详细技术报告

└── README.md

```







\## 许可



Apache-2.0

