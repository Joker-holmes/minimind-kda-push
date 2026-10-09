\# 简历项目描述



\## 项目：MiniMind-KDA —— 在小型语言模型上实现并评估 Delta-Rule 线性注意力



\*\*技术栈\*\*：PyTorch 2.11, CUDA 12.8, Transformers, Python 3.11



\*\*项目背景\*\*：在 MiniMind（\~35M 参数）上实现 Kimi Delta Attention（KDA V3），一种基于 delta-rule 的线性注意力机制，与标准 Transformer 注意力做系统性对照，覆盖架构对比、性能、数值稳定性、缓存一致性与 SFT 训练全链路。



\### 核心工作



\*\*1. Chunked Delta-Rule 并行化\*\*



\- 把 KDA 的 per-timestep 递推重写为分块并行形式：块内所有位置通过一次下三角线性求解同时得出，块之间才串行传递状态

\- 严格保持 delta-rule 语义，与参考实现误差 < 1e-7

\- 单算子加速 \*\*8.4×\*\*，端到端训练从慢 31× 压到 \*\*1.8×\*\*



\*\*2. 数值稳定性工程\*\*



\- 定位并修复 fp32 subnormal 陷阱：`D\\\_i / D\\\_{j+1}` 在长 chunk 下进入 subnormal 区，破坏三角矩阵近对角元

\- 通过 \*\*fp64 累积衰减 + state 范数裁剪 + 对角线阻尼\*\*三项修复，SFT 训练从 step 847 崩溃变为 2000 步 \*\*0 NaN skip\*\*



\*\*3. 缓存一致性验证\*\*



\- 设计 \*\*FP32 对照 + BF16 语义判定\*\*的双层回归测试

\- FP32 下 prefill + incremental decode 与全序列前向的 \*\*argmax 一致率 100%\*\*，最大误差 `< 5e-6`

\- 证明 KDA 的 `recurrent\\\_state` 和 `conv\\\_state` 在两条路径上数值等价



\*\*4. 真实语料对照\*\*



\- 在 27K 条真实中文预训练记录（5.5M tokens）上做受控对比

\- KDA V3 vs Baseline: val loss \*\*−0.48\*\*，PPL \*\*−40%\*\*，next-token accuracy 从 \*\*5.30% 翻倍到 9.64%\*\*



\### 关键数字



| 指标 | Baseline | KDA V3 | 变化 |

|---|---:|---:|---:|

| Val Loss（真实语料） | 7.357 | \*\*6.849\*\* | −6.9% |

| Val PPL | 1568 | \*\*943\*\* | −40% |

| Next-token Accuracy | 5.30% | \*\*9.64%\*\* | +82% |

| 训练步耗时 | 75 ms | 73 ms | ≈ 持平 |

| 推理吞吐 | 42k tok/s | 16k tok/s | 2.6× |

| 推理显存 | 179 MB | 189 MB | +10 MB |



\### 技术难点



\- Delta-rule 递推的块内下三角系统推导与 `torch.linalg.solve` 应用

\- fp32 浮点精度边界（subnormal 区）对训练稳定性的影响

\- BF16 autocast 与 RMSNorm 权重 dtype 不匹配导致的 kernel 回退

\- PyTorch `cuda` 与 `cuda:0` 设备比较陷阱

\- `torch.linalg.solve` 反向传播的病态条件数问题



\---



\*\*字数\*\*：约 600 字

\*\*适合\*\*：LLM 算法工程师 / 推理优化 / 大模型训练岗位简历

