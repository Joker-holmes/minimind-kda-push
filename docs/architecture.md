\# 架构详解



\## 目录



1\. \[整体结构](#整体结构)

2\. \[KDA V3 模块](#kda-v3-模块)

3\. \[Chunked 递推](#chunked-递推)

4\. \[缓存格式](#缓存格式)

5\. \[与标准注意力的接口对齐](#与标准注意力的接口对齐)



\## 整体结构



MiniMind 是一个标准 decoder-only Transformer：



```

Input IDs

\&#x20; │

\&#x20; ▼

Embedding

\&#x20; │

\&#x20; ▼

┌──────────────────────────────────┐

│  Transformer Block × 4           │

│  ┌────────────────────────────┐  │

│  │ Input LayerNorm            │  │

│  │      ▼                     │  │

│  │ Attention Backend:         │  │

│  │   - Standard Attention     │  │

│  │   - OR KDA V3              │  │

│  │      ▼                     │  │

│  │ Residual + BlockAttnRes    │  │

│  │      ▼                     │  │

│  │ MLP (SwiGLU)               │  │

│  └────────────────────────────┘  │

└──────────────────────────────────┘

\&#x20; │

\&#x20; ▼

Final RMSNorm

\&#x20; │

\&#x20; ▼

LM Head

```



\## KDA V3 模块



```python

class KDAAttentionV3(nn.Module):

\&#x20;   def \\\_\\\_init\\\_\\\_(self, config):

\&#x20;       # 1. Q/K/V Linear Projections (after conv)

\&#x20;       self.q\\\_proj, self.k\\\_proj, self.v\\\_proj = ...

\&#x20;       self.o\\\_proj = ...

\&#x20;       

\&#x20;       # 2. Q/K RMSNorm

\&#x20;       self.q\\\_norm = nn.RMSNorm(head\\\_dim)

\&#x20;       self.k\\\_norm = nn.RMSNorm(head\\\_dim)

\&#x20;       

\&#x20;       # 3. Depthwise ShortConv1d (K=4, BEFORE projection)

\&#x20;       self.q\\\_conv = ShortConv1dV3(hidden\\\_size, kernel\\\_size=4)

\&#x20;       self.k\\\_conv = ShortConv1dV3(hidden\\\_size, kernel\\\_size=4)

\&#x20;       self.v\\\_conv = ShortConv1dV3(hidden\\\_size, kernel\\\_size=4)

\&#x20;       

\&#x20;       # 4. Gates

\&#x20;       self.alpha\\\_down = nn.Linear(hidden\\\_size, num\\\_heads)

\&#x20;       self.alpha\\\_up = nn.Linear(num\\\_heads, num\\\_heads)

\&#x20;       self.beta\\\_proj = nn.Linear(hidden\\\_size, num\\\_heads)

\&#x20;       

\&#x20;       # 5. Recurrence config

\&#x20;       self.use\\\_chunked = True

\&#x20;       self.chunk\\\_size = 64

\&#x20;       self.max\\\_state\\\_norm = 20.0

\&#x20;       self.diag\\\_damping = 1.05

\&#x20;   

\&#x20;   def forward(self, x, position\\\_embeddings=None,

\&#x20;               past\\\_key\\\_value=None, use\\\_cache=False,

\&#x20;               attention\\\_mask=None):

\&#x20;       # Conv → Proj → Norm → Recurrence

\&#x20;       ...

```



\### 前向流程



```

x \\\[B, T, D]

\&#x20; │

\&#x20; ├─► q\\\_conv(x) ──► q\\\_proj ──► \\\[B, T, D]  ──► split\\\_heads ──► q\\\_norm ──► L2 norm

\&#x20; ├─► k\\\_conv(x) ──► k\\\_proj ──► \\\[B, T, D]  ──► split\\\_heads ──► k\\\_norm ──► L2 norm

\&#x20; └─► v\\\_conv(x) ──► v\\\_proj ──► \\\[B, T, D]  ──► split\\\_heads ──► v

\&#x20;                                               │

\&#x20; α = sigmoid(alpha\\\_up(alpha\\\_down(x)))         │

\&#x20; β = sigmoid(beta\\\_proj(x))                     │

\&#x20;                                               │

\&#x20; ┌─────────────────────────────────────────────▼────────────────────┐

\&#x20; │  if use\\\_chunked and T > chunk\\\_size:                              │

\&#x20; │      kda\\\_recurrence\\\_chunked(q, k, v, α, β, state)                │

\&#x20; │  else:                                                           │

\&#x20; │      kda\\\_recurrence\\\_script(q, k, v, α, β, state)                 │

\&#x20; └─────────────────────────────────────────────┬────────────────────┘

\&#x20;                                               │

\&#x20; output \\\[B, H, T, D\\\_h]                         │

\&#x20; state \\\[B, H, D\\\_h, D\\\_h]                        │

\&#x20;                                               │

\&#x20; output = output.transpose(1, 2).reshape(B, T, D)

\&#x20; output = o\\\_proj(output)

\&#x20; return output, (state, new\\\_conv\\\_state) if use\\\_cache else (output, None)

```



\## Chunked 递推



见 \[TECHNICAL\_REPORT.md](../TECHNICAL\_REPORT.md) 第 3 节。



核心步骤：



```python

for c in range(n\\\_chunks):

\&#x20;   # 1. 累积衰减（fp64）

\&#x20;   D\\\_cum = cumprod(\\\[1, α\\\_c])

\&#x20;   

\&#x20;   # 2. Gram 矩阵

\&#x20;   G = K\\\_c @ K\\\_c.T

\&#x20;   

\&#x20;   # 3. 下三角矩阵

\&#x20;   M = β\\\_c\\\[None, :] \\\* G \\\* ratio\\\_m \\\* mask\\\_lower

\&#x20;   M\\\_full = M + eye \\\* diag\\\_damping

\&#x20;   

\&#x20;   # 4. 解预测误差

\&#x20;   E = torch.linalg.solve(M\\\_full, V\\\_c - D\\\_i \\\* K\\\_c @ S)

\&#x20;   

\&#x20;   # 5. 输出

\&#x20;   out = D\\\_j1 \\\* (Q\\\_c @ S) + A\\\_intra @ E

\&#x20;   

\&#x20;   # 6. 状态更新 + 裁剪

\&#x20;   S = D\\\_C \\\* S + (coeff \\\* K\\\_c).T @ E

\&#x20;   S = clip\\\_norm(S, max\\\_state\\\_norm)

```



\## 缓存格式



KDA 和标准注意力的缓存格式\*\*不同\*\*：



| 后端 | 缓存内容 | 形状 |

|---|---|---|

| Standard Attention | `(K\\\_cache, V\\\_cache)` | `\\\[B, T, KV\\\_heads, D]` × 2 |

| KDA V3 | `(recurrent\\\_state, conv\\\_state)` | `\\\[B, H, D\\\_h, D\\\_h]` 和 `\\\[B, K-1, D]` |



两者都返回 `(state, new\\\_conv\\\_state)` 格式的元组，上层 `MiniMindBlock` 不关心内部结构。



\## 与标准注意力的接口对齐



```python

class Attention(nn.Module):

\&#x20;   def forward(self, x, position\\\_embeddings,

\&#x20;               past\\\_key\\\_value=None, use\\\_cache=False,

\&#x20;               attention\\\_mask=None):

\&#x20;       ...

\&#x20;       return output, (xk, xv) if use\\\_cache else (output, None)





class KDAAttentionV3(nn.Module):

\&#x20;   def forward(self, x, position\\\_embeddings=None,

\&#x20;               past\\\_key\\\_value=None, use\\\_cache=False,

\&#x20;               attention\\\_mask=None):

\&#x20;       ...

\&#x20;       return output, (state, new\\\_conv\\\_state) if use\\\_cache else (output, None)

```



两者签名一致，可以直接在 `MiniMindBlock` 里互换。

