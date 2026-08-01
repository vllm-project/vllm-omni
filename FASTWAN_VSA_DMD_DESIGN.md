# FastWan2.2 TI2V-5B：VSA、gate_compress 与 DMD 适配说明

> 当前状态：DMD 与 VSA 已整合进现有 `Wan22Pipeline`，gate 由 checkpoint 自动决定；单元测试与 B300 官方小猫 demo 均已跑通。

## 1. 当前修改概览

### 1.1 合并最新主仓

目标分支原本基于较早版本的 `main`。目前已经：

- 添加主仓 remote：`upstream = https://github.com/vllm-project/vllm-omni.git`
- 拉取最新 `upstream/main`
- 合并到 `dev/fastvideo-vsa-wan22`
- 解决唯一冲突：diffusion attention backend registry
- 同时保留主仓新增 backend 和 `FASTVIDEO_VSA`

合并暂未 commit，因为模型适配工作尚未完成。

### 1.2 PR #4820 已有的 VSA 实现

[PR #4820](https://github.com/vllm-project/vllm-omni/pull/4820) 原本修改了 5 个文件：

- 新增 FastVideo VSA backend
- 注册 `FASTVIDEO_VSA`
- Wan2.2 self-attention 传递三维视频网格信息
- Wan2.2 VACE 做同类适配
- 添加基础 backend 测试

关键入口：

- `vllm_omni/diffusion/attention/backends/fastvideo_vsa.py`
- `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py`
- `vllm_omni/diffusion/attention/backends/registry.py`

Wan transformer 会把 token 序列对应的三维形状传给 backend：

```python
vsa_dit_seq_shape = (
    post_patch_num_frames,
    post_patch_height,
    post_patch_width,
)
```

即：

```text
T × H × W
```

VSA backend 根据该几何结构把一维 token 序列重新划分成时空 tile。

### 1.3 新增 `to_gate_compress`

仅当 self-attention backend 是 `FASTVIDEO_VSA` 时创建：

```python
self.to_gate_compress = ColumnParallelLinear(
    dim,
    inner_dim,
    bias=True,
    gather_output=False,
)
```

它对应 FastWan checkpoint 中的权重：

```text
blocks.N.attn1.to_gate_compress.weight
blocks.N.attn1.to_gate_compress.bias
```

该层初始化为零，以兼容不同 checkpoint：

- FastWan checkpoint：加载训练好的 gate 权重，得到 learned gate。
- 普通 Wan checkpoint：没有这两个参数，保持零初始化，等价于 zero gate。
- Dense attention：不创建该层，不改变普通 Wan 模型结构。

forward 中增加：

```text
gate_compress = W_g x + b_g
```

随后 reshape 为与 Q/K/V 相同的布局：

```text
[B, S, H, D]
```

并通过以下 metadata 传给 VSA backend：

```python
AttentionMetadata.extra["gate_compress"]
```

### 1.4 VSA backend 的 gate 模式

设计为三种模式：

```text
checkpoint-driven
```

不再向用户暴露 `auto`、`zero` 或 `none` 参数：checkpoint 含 `to_gate_compress` 权重时自动使用 learned gate；不含时保持零初始化，自动得到 sparse-only 输出。`none` 只保留为解释 kernel 数学语义的概念，不是运行配置。

## 2. Dense attention 的数学原理

输入 hidden states：

$$
X \in \mathbb{R}^{B\times S\times C}
$$

Wan self-attention 首先计算：

$$
Q=XW_Q+b_Q
$$

$$
K=XW_K+b_K
$$

$$
V=XW_V+b_V
$$

经过多头 reshape 后：

$$
Q,K,V\in\mathbb{R}^{B\times S\times H\times D}
$$

其中：

$$
C=H\cdot D
$$

标准 attention 为：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}\right)
$$

$$
Y=AV
$$

对于每个 head，attention score 矩阵大小为：

$$
S\times S
$$

主要计算复杂度为：

$$
O(BHS^2D)
$$

attention 中间量的显存开销近似为：

$$
O(BHS^2)
$$

视频 token 数通常为：

$$
S=T_pH_pW_p
$$

随着帧数和分辨率增加，$S$ 快速增大，因此 dense attention 成为视频 DiT 的主要瓶颈之一。

## 3. VSA 如何减少计算

VSA 不再让每个 query token 与全部 $S$ 个 key token 计算 attention，而是先按视频的三维结构分块。

当前默认 tile 为：

```text
(4, 8, 8)
```

每个完整 tile 包含：

$$
4\times8\times8=256
$$

个 token。

若 patch 后的视频网格为：

$$
(T_p,H_p,W_p)
$$

tile 数约为：

$$
N_b=
\left\lceil\frac{T_p}{4}\right\rceil
\left\lceil\frac{H_p}{8}\right\rceil
\left\lceil\frac{W_p}{8}\right\rceil
$$

VSA 对每个 query block 仅选择 top-k 个 key blocks。

设每个 block 有 $M=256$ 个 token。dense block attention 需要处理约：

$$
N_b^2M^2
$$

个 token pair；VSA 只处理：

$$
N_b\cdot k\cdot M^2
$$

个 token pair。

理想计算比例近似为：

$$
\frac{k}{N_b}
$$

理论加速比近似为：

$$
\frac{N_b}{k}
$$

实际加速还会受到以下开销影响：

- block 选择
- tile/untile
- padding
- kernel launch
- 不规则边界 block
- Q/K/V 内存搬运

因此实际速度不会等于理论上限。

## 4. 三维 tile、padding 与顺序恢复

原始 Wan attention 输入是一维序列：

```text
[B, S, H, D]
```

VSA 必须知道 token 在视频中的原始位置，因此需要额外传递：

```text
(T_p, H_p, W_p)
```

backend 的数据流为：

```text
原始一维 token 顺序
    ↓
恢复 T/H/W 坐标
    ↓
按 (4, 8, 8) 收集 tile
    ↓
不足 256 token 的边界 tile 补零
    ↓
调用 video_sparse_attn_bshd
    ↓
去除 padding
    ↓
恢复原始 token 顺序
```

相关索引的职责：

- `tile_partition_indices`：从原始顺序重排为 tile 顺序。
- `variable_block_sizes`：记录边界 block 的真实 token 数。
- `non_pad_index`：记录真实 token 在 padding buffer 中的位置。
- `untile_combined_index`：把 kernel 输出恢复为原始顺序。

最终 VSA 输出仍保持：

$$
Y\in\mathbb{R}^{B\times S\times H\times D}
$$

因此上层 Wan transformer block 无需感知 attention 内部是 dense 还是 sparse。

## 5. `gate_compress` 的数学意义

FastWan 为每个 self-attention block 增加一个可训练投影：

$$
G=XW_G+b_G
$$

其中：

$$
G\in\mathbb{R}^{B\times S\times H\times D}
$$

代码中的参数名为：

```text
to_gate_compress
```

`gate_compress` 与 transformer residual 中已有的 `gate_msa` 不同：

- `gate_msa`：控制 attention 输出在 residual 上的幅度。
- `gate_compress`：传给 VSA kernel，逐 token/head/channel 控制 compressed global branch 加回最终输出的权重；它不参与 top-k block 选择。

原有 residual 仍为：

$$
X_{\mathrm{out}}
=
X+
g_{\mathrm{msa}}\odot\operatorname{Attention}(X)
$$

`gate_compress` 工作在 attention kernel 的输出融合阶段。kernel 先用 block-mean Q/K scores 选择 top-k；gate 只调制 compressed global branch，不改变 top-k indices。

FastVideo 的 CUDA kernel 对 Python 侧表现为 opaque custom op。vLLM-Omni 侧可以严格确认的数据关系是：

$$
(Q,K,V,G)
\longrightarrow
\operatorname{video\_sparse\_attn}
$$

具体 block score 的完整 CUDA 公式不在 vLLM-Omni Python 代码中重新实现，因此不应在适配层臆造其内部细节。

从训练目标看，$W_G$ 学到的是 compressed global correction 的逐元素融合权重。top-k block 集合由 block-mean Q/K 相似度决定，而不是由 $W_G$ 决定。

### 5.1 checkpoint 驱动的唯一运行逻辑

用户不需要也不能选择 gate 模式。每个 VSA self-attention block 都创建 `to_gate_compress` 投影并先将权重、bias 置零：

- checkpoint 含 gate 权重：加载器覆盖零值，运行 learned gate；
- checkpoint 不含 gate 权重：仅将这些可选权重视为合法缺省，投影保持全零，运行 sparse-only；
- 其他任何 checkpoint 缺失参数仍由 loader 严格报错。

这等价于内部的 `auto`，但没有用户参数，也不存在配置分支。kernel 的 `compress_attn_weight=None` 语义仍可用于理解公式，却不会由 Wan VSA pipeline 选择。

## 6. DMD 3-step 的数学原理

DMD 是 Distribution Matching Distillation，即分布匹配蒸馏。

DMD 与 VSA 位于两个不同的加速维度：

- VSA 减少每个 diffusion step 内 attention 的计算量。
- DMD 减少 diffusion step 的总数量。

普通 flow-matching 模型在时间 $t$ 上接收：

$$
x_t=(1-\sigma_t)x_0+\sigma_t\epsilon
$$

模型预测 flow/noise：

$$
v_\theta(x_t,t,c)
$$

从当前 noisy latent 估计 clean latent：

$$
\hat{x}_0=x_t-\sigma_t v_\theta(x_t,t,c)
$$

普通 Euler/UniPC 沿连续 ODE 轨迹执行多个小步。例如 Euler：

$$
x_{t_{i+1}}
=
x_{t_i}
+
(\sigma_{i+1}-\sigma_i)
v_\theta(x_{t_i},t_i,c)
$$

原生模型通常需要约 30–40 步，因为它主要保证局部方向足够准确。

DMD 训练学生模型，使少量离散时间点上的输出分布直接逼近教师的最终生成分布。FastWan2.2 使用固定时间点：

```text
[1000, 757, 522]
```

每一步的逻辑是：

1. 学生从当前 $x_{t_i}$ 直接预测 clean video：

   $$
   \hat{x}_0^{(i)}
   =
   x_{t_i}
   -
   \sigma_{t_i}v_\theta(x_{t_i},t_i,c)
   $$

2. 若不是最后一步，重新采样噪声：

   $$
   \epsilon_i\sim\mathcal{N}(0,I)
   $$

3. 将预测的 clean latent 加噪到下一个蒸馏时间点：

   $$
   x_{t_{i+1}}
   =
   (1-\sigma_{t_{i+1}})
   \hat{x}_0^{(i)}
   +
   \sigma_{t_{i+1}}\epsilon_i
   $$

4. 最后一步直接输出：

   $$
   x_{\mathrm{final}}=\hat{x}_0^{(2)}
   $$

因此，仅设置：

```python
num_inference_steps = 3
```

并不等价于 FastWan 的 DMD 3-step。

普通 3-step scheduler 是三个跨度很大的 ODE 更新；FastWan DMD 使用“预测 clean → 按下一个指定时间重新加噪 → 再预测 clean”的路径，并且 checkpoint 专门针对这三个离散时间点训练过。

## 7. 完整 FastWan 加速关系

原生 Wan2.2 的主要路径可概括为：

```text
约 30–40 steps × dense attention
```

FastWan2.2 的目标路径为：

```text
3 DMD steps × VSA sparse attention × learned gate selection
```

粗略计算比例为：

$$
\frac{3}{40}\cdot\frac{k}{N_b}
$$

真实端到端加速不会达到该理论值，因为还包含：

- text encoder
- VAE decode
- transformer MLP
- Q/K/V projection
- gate projection
- tile 重排
- kernel 固定开销

DMD 通常是更主要的加速来源；VSA 在每个剩余 step 上进一步压缩 attention 计算。

## 8. 完整数值计算示例

下面使用 FastWan2.2 TI2V-5B 的典型 `121×704×1280` 生成规格，完整走一遍前面的公式。

### 8.1 从视频尺寸计算 DiT token 数

设 batch size 为 1。Wan2.2 TI2V-5B VAE 的时间压缩倍率为 4、空间压缩倍率为 16。由于第一帧单独保留：

$$
T_{latent}=\frac{121-1}{4}+1=31
$$

$$
H_{latent}=\frac{704}{16}=44,\qquad
W_{latent}=\frac{1280}{16}=80
$$

VAE latent 几何尺寸为 `31×44×80`。Transformer patch size 是 `(1,2,2)`，因此 patch 后的网格为：

$$
(T_p,H_p,W_p)=(31,22,40)
$$

self-attention token 数为：

$$
S=T_pH_pW_p=31\times22\times40=27{,}280
$$

TI2V-5B 有 24 个 heads，每个 head 维度为 128，所以 hidden size 为：

$$
C=24\times128=3072
$$

Q/K/V 的形状为：

```text
[B, S, H, D] = [1, 27280, 24, 128]
```

### 8.2 Dense attention 的实际规模

这一节的核心是：在 dense self-attention 中，每一个 query token 都要与每一个 key token 做一次长度为 128 的向量点积。

#### 8.2.1 Q、K 的矩阵形状

本例的参数为：

```text
Batch B       = 1
Token 数 S    = 27,280
Head 数 H     = 24
Head 维度 D   = 128
Hidden size C = H × D = 3,072
```

线性投影和多头 reshape 后：

```text
Q: [1, 27280, 24, 128]
K: [1, 27280, 24, 128]
V: [1, 27280, 24, 128]
```

不同 heads 之间独立计算。固定一个 batch 和一个 head 后：

```text
Q_head:   [27280, 128]
K_head:   [27280, 128]
K_head.T: [128, 27280]
```

所以矩阵乘法为：

```text
[27280, 128] × [128, 27280]
    → [27280, 27280]
```

因此每个 head 的 attention score 矩阵是 $S\times S$。

#### 8.2.2 为什么有 $S^2$ 个 scores

从单个 query token 看：

```text
q₀ 与 k₀、k₁、...、k₂₇₂₇₉ 分别计算 score
```

一个 query 需要计算 27,280 个 scores，而一共有 27,280 个 queries。因此一个 head 的 score 数量为：

$$
S^2=27{,}280^2=744{,}198{,}400
$$

也可以写成：

$$
\underbrace{27{,}280}_{\text{query 数}}
\times
\underbrace{27{,}280}_{\text{每个 query 访问的 key 数}}
=744{,}198{,}400
$$

#### 8.2.3 一个 score 如何计算

一个 query 和一个 key 都是 128 维向量：

$$
q_i=[q_{i,1},q_{i,2},\ldots,q_{i,128}]
$$

$$
k_j=[k_{j,1},k_{j,2},\ldots,k_{j,128}]
$$

它们的 attention score 为：

$$
s_{ij}=\frac{q_i\cdot k_j}{\sqrt{128}}
$$

其中点积展开为：

$$
q_i\cdot k_j
=q_{i,1}k_{j,1}+q_{i,2}k_{j,2}+\cdots+q_{i,128}k_{j,128}
$$

计算一个 score 约需要：

- 128 次乘法
- 127 次加法
- 一次缩放

GPU 通常把乘法和加法融合为 FMA/MAC，因此一个 score 近似需要 $D=128$ 次 MAC。如果按 FLOPs 统计，一次乘法算 1 FLOP、一次加法算 1 FLOP，则一个 score 约为：

$$
2D=256\text{ FLOPs}
$$

#### 8.2.4 一个 head 的 $QK^\top$ 计算量

一个 head 有 $S^2$ 个 scores，每个 score 需要 128 次 MAC：

$$
S^2D
=744{,}198{,}400\times128
=95{,}257{,}395{,}200
$$

即一个 head 约有 952.6 亿次 MAC。换算成 FLOPs：

$$
2S^2D=190{,}514{,}790{,}400
$$

约为 190.5 GFLOPs。

#### 8.2.5 24 个 heads 的 $QK^\top$

24 个 heads 分别计算：

$$
24\times S^2\times128
=2{,}286{,}177{,}484{,}800
$$

即约 2.286 万亿次 MAC。若按照一次乘法和一次加法合计 2 FLOPs：

$$
2\times24\times S^2\times128
=4{,}572{,}354{,}969{,}600
$$

即一次 transformer layer forward 中，仅 $QK^\top$ 就约为 4.572 TFLOPs。

#### 8.2.6 还需要计算 $AV$

计算 score 和 softmax 后：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}\right)
$$

单个 head 的形状为：

```text
A_head: [27280, 27280]
V_head: [27280, 128]
```

接着计算：

```text
[27280, 27280] × [27280, 128]
    → [27280, 128]
```

也就是 $Y=AV$。$AV$ 的 MAC 数同样为 $S^2D$。所以两次主要矩阵乘法分别为：

$$
QK^\top: BHS^2D\text{ MACs}
$$

$$
AV: BHS^2D\text{ MACs}
$$

若换算成 FLOPs，完整 attention GEMM 约为：

$$
\text{Attention FLOPs}\approx4BHS^2D
$$

代入本例：

$$
4\times1\times24\times27{,}280^2\times128
\approx9.145\times10^{12}\text{ FLOPs}
$$

即一次 transformer layer forward，仅 $QK^\top$ 和 $AV$ 两个 GEMM 就约为 9.15 TFLOPs。这还不包括：

- Q/K/V linear projection
- output projection
- softmax
- RMSNorm 和 RoPE
- FFN
- `gate_compress` projection

#### 8.2.7 一个小矩阵例子

假设只有 3 个 tokens，每个 token 的 head dimension 为 2：

$$
Q=\begin{bmatrix}
q_{11}&q_{12}\\
q_{21}&q_{22}\\
q_{31}&q_{32}
\end{bmatrix},\qquad
K=\begin{bmatrix}
k_{11}&k_{12}\\
k_{21}&k_{22}\\
k_{31}&k_{32}
\end{bmatrix}
$$

则：

$$
QK^\top=
\begin{bmatrix}
q_1\cdot k_1&q_1\cdot k_2&q_1\cdot k_3\\
q_2\cdot k_1&q_2\cdot k_2&q_2\cdot k_3\\
q_3\cdot k_1&q_3\cdot k_2&q_3\cdot k_3
\end{bmatrix}
$$

结果有 $3\times3=9$ 个 scores。每个 score 是长度为 2 的点积，所以 MAC 数为：

$$
3\times3\times2=18
$$

一般化后就是：

$$
S\times S\times D=S^2D
$$

再乘 batch 和 head 数：

$$
BHS^2D
$$

#### 8.2.8 FlashAttention 为什么仍是这个计算规模

FlashAttention 改变的是计算和访存方式，而不是 dense attention 的数学关系。它将 Q/K/V 分块，在片上 SRAM 中完成：

```text
Q block × K block
    → 在线 softmax
    → 立即乘 V block
```

因此它可以：

- 避免把完整 $S\times S$ score 矩阵写入显存
- 显著减少显存流量
- 降低峰值显存
- 提高 GPU 利用率

但是 dense FlashAttention 仍然要遍历所有 query-key pairs，理论计算复杂度仍为：

$$
O(BHS^2D)
$$

VSA 的不同之处是，它真正跳过未进入 top-k 的 block pairs，因此不仅优化访存，也减少实际执行的 query-key 点积。

### 8.3 划分 VSA blocks

这一节把 `(31,22,40)` 的 DiT token 网格如何变成 120 个 VSA blocks 完整展开。

#### 8.3.1 输入不是普通的一维序列

进入 attention 时，hidden states 已经表现为一维 token 序列：

```text
[B, S, C] = [1, 27280, 3072]
```

但是这 27,280 个 tokens 原本来自三维视频 patch 网格：

```text
T_p × H_p × W_p = 31 × 22 × 40
```

原始 flatten 顺序可以写成：

$$
\operatorname{index}(t,h,w)
+=(t\times H_p+h)\times W_p+w
$$

在本例中：

$$
\operatorname{index}(t,h,w)
+=(t\times22+h)\times40+w
$$

例如：

```text
(t=0, h=0, w=0) → index 0
(t=0, h=0, w=1) → index 1
(t=0, h=1, w=0) → index 40
(t=1, h=0, w=0) → index 880
```

其中一个 temporal patch plane 包含：

$$
22\times40=880
$$

个 tokens。

VSA 不能只把这条一维序列每 256 个元素机械切开，因为那样得到的 token 不一定构成局部三维视频区域。它需要利用 `vsa_dit_seq_shape=(31,22,40)` 恢复 `(t,h,w)` 几何关系，再按三维 tile 收集 token。

#### 8.3.2 每个方向如何切分

默认 VSA tile size 为：

```text
(tile_t, tile_h, tile_w) = (4, 8, 8)
```

时间方向有 31 个 positions，每块最多取 4 个：

```text
[0..3], [4..7], [8..11], [12..15],
[16..19], [20..23], [24..27], [28..30]
```

因此时间方向的真实 block sizes 为：

```text
[4, 4, 4, 4, 4, 4, 4, 3]
```

数量为：

$$
N_T=\left\lceil\frac{31}{4}\right\rceil=8
$$

高度方向有 22 个 positions，每块最多取 8 个：

```text
[0..7], [8..15], [16..21]
```

真实 sizes 为：

```text
[8, 8, 6]
```

数量为：

$$
N_H=\left\lceil\frac{22}{8}\right\rceil=3
$$

宽度方向有 40 个 positions，刚好能被 8 整除：

```text
[0..7], [8..15], [16..23], [24..31], [32..39]
```

真实 sizes 为：

```text
[8, 8, 8, 8, 8]
```

数量为：

$$
N_W=\left\lceil\frac{40}{8}\right\rceil=5
$$

所以三维 block 网格为：

```text
8 × 3 × 5
```

总 block 数为：

$$
N_b=N_TN_HN_W=8\times3\times5=120
$$

#### 8.3.3 一个完整 block 包含哪些 token

第一个 block 的三维范围是：

```text
t = 0..3
h = 0..7
w = 0..7
```

它包含的 token 数为：

$$
4\times8\times8=256
$$

这些 token 在原始一维序列中不是单个连续区间。例如 `(t=0,h=0)` 对应 index `0..7`，而 `(t=0,h=1)` 对应 index `40..47`，中间的 `8..39` 属于同一行的其他 width blocks。

因此 backend 必须显式构造索引，把下列三维区域收集到一起：

```text
(t0,h0,w0..7), (t0,h1,w0..7), ... (t0,h7,w0..7),
(t1,h0,w0..7), ...,
(t3,h7,w0..7)
```

`tile_partition_indices` 就是所有 blocks 的这些原始 linear indices 按 tile 顺序串接后的索引数组。

#### 8.3.4 边界 block 为什么不都是 256 tokens

宽度方向没有余数，但时间和高度方向存在边界：

- 时间完整块大小为 4，最后一块大小为 3。
- 高度完整块大小为 8，最后一块大小为 6。
- 宽度所有块大小都为 8。

因此本例一共有四类 block：

| 时间大小 | 高度大小 | 宽度大小 | 单 block token 数 | block 数量 |
|---:|---:|---:|---:|---:|
| 4 | 8 | 8 | $4\times8\times8=256$ | $7\times2\times5=70$ |
| 4 | 6 | 8 | $4\times6\times8=192$ | $7\times1\times5=35$ |
| 3 | 8 | 8 | $3\times8\times8=192$ | $1\times2\times5=10$ |
| 3 | 6 | 8 | $3\times6\times8=144$ | $1\times1\times5=5$ |

block 数量守恒：

$$
70+35+10+5=120
$$

token 数也精确守恒：

$$
70\times256
+35\times192
+10\times192
+5\times144
$$

$$
=17{,}920+6{,}720+1{,}920+720
=27{,}280
$$

这正好等于原始 token 数 $31\times22\times40$，说明分块既没有丢 token，也没有重复 token。

#### 8.3.5 `variable_block_sizes` 如何生成

三个方向的真实 sizes 分别为：

$$
T_{sizes}=[4,4,4,4,4,4,4,3]
$$

$$
H_{sizes}=[8,8,6]
$$

$$
W_{sizes}=[8,8,8,8,8]
$$

每个三维 block 的真实 token 数为：

$$
m_{i,j,k}
=T_{sizes}[i]\times H_{sizes}[j]\times W_{sizes}[k]
$$

backend 对这三个数组做广播乘法并 flatten，得到长度为 120 的：

```text
variable_block_sizes
```

其元素只会取：

```text
256、192、144
```

例如：

```text
(time block 0, height block 0, width block 0) → 4×8×8 = 256
(time block 0, height block 2, width block 0) → 4×6×8 = 192
(time block 7, height block 0, width block 0) → 3×8×8 = 192
(time block 7, height block 2, width block 0) → 3×6×8 = 144
```

CUDA kernel 通过这个数组知道每个 block 中有多少有效 tokens。

#### 8.3.6 为什么需要 padding 到 30,720

FastVideo 的 BSHD VSA kernel 使用最大 block capacity 256。为了把所有 blocks 放入统一规则的 tensor，每个 block 都预留 256 个 slots：

$$
S_{padded}=120\times256=30{,}720
$$

原始序列只有 27,280 个有效 tokens，因此 padding slots 为：

$$
30{,}720-27{,}280=3{,}440
$$

也可以从各类边界 block 分别计算 padding：

- 70 个完整 blocks：每块 padding 0。
- 35 个 `192-token` 高度边界 blocks：每块 padding $256-192=64$。
- 10 个 `192-token` 时间边界 blocks：每块 padding 64。
- 5 个 `144-token` 双边界 blocks：每块 padding $256-144=112$。

所以：

$$
35\times64+10\times64+5\times112
=2{,}240+640+560
=3{,}440
$$

两种算法得到相同结果。

需要注意：`variable_block_sizes` 保证 padding 不参与有效 attention 语义；底层 kernel 是否能完全跳过所有 padding 对应的硬件工作，取决于具体 CUDA kernel 实现，不能仅凭 Python 侧布局断言。

#### 8.3.7 `non_pad_index` 如何把有效 token 放入 slots

每个 block 在 padded buffer 中占据连续的 256 个 slots。第 $b$ 个 block 的起点是：

$$
\operatorname{start}_b=b\times256
$$

若该 block 的真实大小为 $m_b$，有效 slot 为：

$$
[\operatorname{start}_b,
\operatorname{start}_b+1,
\ldots,
\operatorname{start}_b+m_b-1]
$$

例如：

- block 0 大小为 256，有效 slots 是 `0..255`。
- block 1 大小为 256，有效 slots 是 `256..511`。
- 假设某 block 起点为 2560、真实大小为 192，则有效 slots 是 `2560..2751`，剩余 `2752..2815` 是 padding。

所有 blocks 的有效 slots 拼起来，就是 `non_pad_index`。随后执行：

```python
query_tiled[:, non_pad_index] = query[:, tile_partition_indices]
key_tiled[:, non_pad_index] = key[:, tile_partition_indices]
value_tiled[:, non_pad_index] = value[:, tile_partition_indices]
```

含义是：

1. `tile_partition_indices` 从原始序列中按三维 tile 顺序取 token。
2. `non_pad_index` 把这些 token 写入各 block 的有效 slots。
3. 没有被写入的 slots 保持为零 padding。

learned `gate_compress` 也必须使用完全相同的两个索引进行重排，否则 gate 会和对应的 Q/K/V token 错位。

#### 8.3.8 kernel 输出如何恢复原始顺序

kernel 输出仍采用 padded tile 顺序。首先只取有效 slots：

```text
output[:, non_pad_index]
```

此时 token 数恢复为 27,280，但顺序仍是 tile order，而不是 Wan transformer 原始的 `(t,h,w)` flatten order。

`tile_partition_indices` 是从 tile order 指向 original order 的排列。其逆排列可以通过：

```python
torch.argsort(tile_partition_indices)
```

得到。代码把“去 padding”和“逆排列”合并为：

```python
untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
```

最终：

```python
output_original = output_tiled[:, untile_combined_index]
```

输出形状回到：

```text
[1, 27280, 24, 128]
```

而且第 $i$ 个输出 token 重新对应原始第 $i$ 个视频 patch。上层 residual connection 因此可以直接执行：

$$
X_{out}=X+g_{msa}\odot Y
$$

不需要知道 attention 内部曾经做过 tile、padding 和 untile。

#### 8.3.9 完整数据流汇总

本例的 shape 变化为：

```text
原始视频：121 × 704 × 1280
    ↓ VAE (/4 temporal, /16 spatial)
latent grid：31 × 44 × 80
    ↓ transformer patch (1, 2, 2)
DiT grid：31 × 22 × 40
    ↓ flatten
原始 attention sequence：27,280 tokens
    ↓ tile (4, 8, 8)
120 variable-size blocks
    ↓ 每 block 预留 256 slots
padded tile sequence：30,720 slots
    ↓ VSA kernel
padded tile output：30,720 slots
    ↓ remove padding + inverse permutation
原始顺序 output：27,280 tokens
```

因此，8.3 的分块过程本质上不是减少 token 数，而是把 token 重排成具有视频局部性的 blocks。真正减少 attention 计算的是下一节的 top-k：每个 query block 只连接部分 key blocks。

### 8.4 top-k 如何选择，以及为什么默认是 64

#### 8.4.1 VSA 的 block-level 压缩计算

FastVideo VSA 先把每个 block 压缩成一个均值向量。设第 $i$ 个 query block 有 $m_i$ 个有效 tokens，第 $j$ 个 key/value block 有 $m_j$ 个有效 tokens：

$$
\bar Q_i=\frac{1}{m_i}\sum_{p\in block_i}Q_p
$$

$$
\bar K_j=\frac{1}{m_j}\sum_{p\in block_j}K_p
$$

$$
\bar V_j=\frac{1}{m_j}\sum_{p\in block_j}V_p
$$

padding 不参与均值，分母来自 `variable_block_sizes`。随后计算 block-level scores：

$$
S^{block}_{i,j}=\frac{\bar Q_i\bar K_j^\top}{\sqrt D}
$$

本例有 120 个 blocks，因此：

```text
q_c:    [B, H, 120, 128]
k_c:    [B, H, 120, 128]
scores: [B, H, 120, 120]
```

每个 head 只计算 $120^2=14,400$ 个 coarse scores，而非 token-level dense attention 的 $27,280^2=744,198,400$ 个 scores。

#### 8.4.2 top-k 的实际选择过程

对每一个 batch、head 和 query block，都有一行 120 个候选 key-block scores：

$$
[S_{i,0},S_{i,1},\ldots,S_{i,119}]
$$

kernel 沿最后一维选择最大的 64 个：

```python
topk_indices = torch.topk(scores, k=64, dim=-1).indices
mask = torch.zeros_like(scores, dtype=torch.bool)
mask.scatter_(-1, topk_indices, True)
```

mask 的形状为：

```text
[B, H, N_q, N_kv] = [1, 24, 120, 120]
```

每一行严格有 64 个 `True`，token-level sparse kernel 只计算这些 block pairs。

`topk=64` 固定的是数量，不是 block identity。具体选择会随 prompt、输入、diffusion timestep、transformer layer、attention head、query block 和当前 Q/K 表示动态变化。

#### 8.4.3 gate 不参与 top-k

block 选择公式是：

$$
\operatorname{topk}\left(\frac{\bar Q\bar K^\top}{\sqrt D}\right)
$$

`gate_compress` 在 top-k mask 生成后才参与输出融合：

```text
block-mean Q/K → scores → top-k mask → sparse branch O_s
                     ↓
              compression branch O_c
                     ↓
              gate 融合 O_s 与 O_c
```

所以 gate 不会改变某个 query block 选择了哪 64 个 key blocks。

#### 8.4.4 top-k=64 的计算比例

本例 $N_b=120$。sparse block pairs 为：

$$
N_bk=120\times64=7{,}680
$$

dense block pairs 为：

$$
N_b^2=120^2=14{,}400
$$

保留比例：

$$
\frac{k}{N_b}=\frac{64}{120}\approx53.3\%
$$

block sparsity：

$$
1-\frac{64}{120}\approx46.7\%
$$

理想 attention 部分加速比约为：

$$
\frac{N_b}{k}=\frac{120}{64}=1.875
$$

若保守地把所有边界 blocks 都按 256 tokens 估算，sparse token-pair 上界为：

$$
120\times64\times256^2=503{,}316{,}480
$$

相对真实 dense token pairs 的比例约为：

$$
\frac{503{,}316{,}480}{744{,}198{,}400}\approx67.6\%
$$

该保守估算包含 padding；真实 kernel 使用 `variable_block_sizes` 处理边界 blocks。

#### 8.4.5 从 VSA sparsity 推导 top-k

FastVideo 原生实现可按目标稀疏率动态计算：

$$
k=\left\lceil(1-\text{VSA\_sparsity})N_{kv}\right\rceil
$$

并限制 $1\le k\le N_{kv}$。本例 $N_{kv}=120$、$k=64$，对应：

$$
\text{VSA\_sparsity}=1-\frac{64}{120}\approx0.4667
$$

如果目标 sparsity 是 80%，则 $k=\lceil0.2\times120\rceil=24$，但更激进的 sparsity 通常会带来更明显的质量损失。

#### 8.4.6 为什么 PR 默认选择 64

64 不是数学上证明的最优值，而是 PR #4820 的 benchmark 经验折中：

| top-k | 稀疏程度 | 结果倾向 |
|---:|---|---|
| 32 | 很激进 | 最快，质量损失最大 |
| 48 | 较激进 | 有明显加速，但质量弱于 64 |
| 64 | 中等 | 质量与速度的最佳经验折中 |
| 96 | 较保守 | 质量略好，但测试中慢于 dense baseline |

该结论来自 Wan2.2 A14B 的特定尺寸和硬件，不保证对 TI2V-5B、其他分辨率或其他 GPU 仍最优。

#### 8.4.7 固定 top-k 与动态比例

固定 64 在不同 block 总数下代表不同 sparsity：

| 总 blocks | top-k | 保留比例 | sparsity |
|---:|---:|---:|---:|
| 80 | 64 | 80.0% | 20.0% |
| 120 | 64 | 53.3% | 46.7% |
| 240 | 64 | 26.7% | 73.3% |

更稳健的方式是固定保留比例 $r$，动态计算 $k=\lceil rN_{kv}\rceil$。例如保持约 53%：

```text
N=80  → k=43
N=120 → k=64
N=240 → k=128
```

当前 PR backend 使用固定 `topk`；后续可增加 `VSA_sparsity` 配置，并禁止与固定 top-k 同时指定。

### 8.5 kernel 中 `zero`、`none` 与 learned gate 的数学区别（非用户选项）

#### 8.5.1 VSA 的两条输出分支

FastVideo kernel 同时产生：

- $O_s$：top-k token-level sparse branch。
- $O_c$：block-mean dense compression branch。

compression branch 使用所有 block-level scores：

$$
A^{block}=\operatorname{softmax}(S^{block})
$$

$$
\bar O_i=\sum_j A^{block}_{i,j}\bar V_j
$$

然后将 $\bar O_i$ 复制给 query block 内所有 tokens，形成 $O_c$。它以较低成本保留粗粒度全局信息。

#### 8.5.2 `none`

`none` 向 kernel 传 `compress_attn_weight=None`，最终输出：

$$
O_{none}=O_s+O_c
$$

因此 `none` 不是关闭 compression branch，而是没有 learned gate、以固定系数 1 完整加入 $O_c$。

#### 8.5.3 `zero`

`zero` 传递与 Q 同形状的全零 tensor：

$$
G=0
$$

最终输出：

$$
O_{zero}=O_s+0\odot O_c=O_s
$$

所以 `zero` 实际关闭 compression branch，只保留 sparse branch。二者之差为：

$$
O_{none}-O_{zero}=O_c
$$

#### 8.5.4 learned gate

checkpoint 若含 `to_gate_compress`：

$$
G=XW_G+b_G
$$

$$
G\in\mathbb{R}^{B\times S\times H\times D}
$$

最终输出：

$$
O_{learned}=O_s+G\odot O_c
$$

当前实现没有对 $G$ 应用 sigmoid，因此 gate 不限于 $[0,1]$，可以放大、减弱或反向抵消 compression branch。

三种模式汇总：

| 模式 | 传给 kernel | 最终输出 |
|---|---|---|
| `none` | `None` | $O_s+O_c$ |
| `zero` | 全零 tensor | $O_s$ |
| learned | $G=XW_G+b_G$ | $O_s+G\odot O_c$ |

#### 8.5.5 标量数值例子

假设某个 token/head/channel 上 $O_s=0.7$、$O_c=0.2$：

- `none`：$O=0.7+0.2=0.9$。
- `zero`：$O=0.7+0\times0.2=0.7$。
- learned $G=0.4$：$O=0.7+0.4\times0.2=0.78$。
- learned $G=-0.5$：$O=0.7-0.5\times0.2=0.6$。

#### 8.5.6 learned projection 的尺寸例子

本例：

$$
X\in\mathbb{R}^{1\times27{,}280\times3{,}072},\qquad
W_G\in\mathbb{R}^{3072\times3072}
$$

reshape 后：

$$
G\in\mathbb{R}^{1\times27{,}280\times24\times128}
$$

共有 83,804,160 个元素，BF16 逻辑大小约 159.8 MiB。gate 必须使用与 Q/K/V 相同的 tile/padding permutation。

二维 toy projection：

$$
x=[2,-1],\quad
W_G=\begin{bmatrix}0.5&0.2\\-0.3&0.4\end{bmatrix},\quad
b_G=[0.1,-0.2]
$$

$$
g=xW_G+b_G=[1.4,-0.2]
$$

这个 $g$ 用于调制 $O_c$，不是用于生成 top-k mask。

#### 8.5.7 当前公开 checkpoint 的自动行为

已下载的 `FastVideo/FastWan2.2-TI2V-5B-Diffusers` transformer 有 825 个 tensors，`gate_tensor_count=0`。运行时无需选择参数：loader 允许这些零初始化 gate 合法缺省，最终自动得到：

$$
O=O_s+0\odot O_c=O_s
$$

如果以后加载包含 `to_gate_compress` 的 VSA 训练 checkpoint，同一份代码会自动加载 learned projection，得到 $O_s+G\odot O_c$。概念上的 `none`（$O_s+O_c$）不对用户开放。当前实测使用 `topk=64`，与 PR #4820 的起始配置一致。

### 8.6 三步 DMD 的手算例子

为了可以手算，只跟踪 latent 中的一个标量。真实模型会对整个 latent tensor 并行执行同样形式的计算。

固定 timestep 为 `[1000,757,522]`。在简化例子中使用：

$$
[\sigma_0,\sigma_1,\sigma_2]=[1.000,0.757,0.522]
$$

初始 noisy latent 设为 $x_{1000}=0.8$。

#### 第一步：$t=1000$

假设模型预测 $v_\theta=0.3$，则 clean prediction 为：

$$
\hat{x}_0^{(0)}=0.8-1.0\times0.3=0.5
$$

采样 $\epsilon_0=-0.2$，重新加噪到 $t=757$：

$$
x_{757}=(1-0.757)\times0.5+0.757\times(-0.2)
$$

$$
x_{757}=0.1215-0.1514=-0.0299
$$

#### 第二步：$t=757$

假设模型预测 $v_\theta=-0.1$：

$$
\hat{x}_0^{(1)}=-0.0299-0.757\times(-0.1)=0.0458
$$

采样 $\epsilon_1=0.4$，重新加噪到 $t=522$：

$$
x_{522}=(1-0.522)\times0.0458+0.522\times0.4
$$

$$
x_{522}=0.0219+0.2088=0.2307
$$

#### 第三步：$t=522$

假设模型预测 $v_\theta=0.2$：

$$
\hat{x}_0^{(2)}=0.2307-0.522\times0.2=0.1263
$$

这是最后一步，不再重新加噪，因此：

$$
x_{final}=0.1263
$$

这个例子体现了 DMD 与普通 3-step Euler 的核心区别：每一步先直接估计 clean latent，再为下一个指定蒸馏 timestep 重新采样噪声，而不是只做三次大跨度 Euler 更新。

### 8.7 DMD 与 VSA 组合后的理想计算比例

原生模型若使用 40 steps，仅看 step 数，DMD 的比例为：

$$
\frac{3}{40}=7.5\%
$$

再乘以 top-k=64 的 block pair 比例：

$$
7.5\%\times53.3\%\approx4.0\%
$$

因此，在忽略其他计算和固定开销的理想模型中，核心 attention 工作量可能下降到原来的约 4%，即约 25 倍工作量缩减：

$$
\frac{1}{0.04}\approx25
$$

这不是端到端 25 倍速度保证。真实时间还包括 projection、MLP、VAE、text encoder、数据移动和 kernel 调度，必须以实际 GPU benchmark 为准。


## 9. 已完成实现与实际测试

### 9.1 代码实现

- 注册 `WanDMDPipeline`，复用现有 `Wan22Pipeline`。
- 从 `model_index.json` 自动识别 DMD checkpoint。
- DMD 固定使用 `[1000, 757, 522]` 与 scheduler shift `8.0`。
- 每步执行 `predict_clean`；前两步向下一蒸馏时间点重新加噪，最后一步直接输出 clean latent。
- DMD 自动使用 3 steps，并兼容引擎内部的 1-step warmup 请求。
- DMD 默认 guidance 为 `1.0`；原生模型默认行为保持不变。
- gate 不暴露用户参数：有权重则 learned，无权重则 zero/sparse-only。
- gate linear 同时兼容 vLLM 返回 Tensor 或 `(Tensor, bias)` 两种形式。
- 仅允许 `to_gate_compress` 合法缺省，其他缺失 checkpoint 权重仍严格报错。

### 9.2 单元测试

DMD scheduler 数学、完整三步 re-noise 循环、VSA tiling/gate/fallback 共 10 项测试通过：

```text
10 passed
```

### 9.3 B300 官方小猫 demo

配置：官方输入图与 prompt，`704×1280`，121 帧，24 fps，seed 1024。

| 项目 | FastWan DMD + VSA | 原生 Wan 30-step |
|---|---:|---:|
| attention | FASTVIDEO_VSA, topk=64 | TORCH_SDPA |
| denoise steps | 3 | 30 |
| denoise loop | 约 8.0 s | 约 57.3 s |
| generate 调用端到端 | 14.0 s | 63.5 s |
| 输出 | 704×1280, 121 帧 | 704×1280, 121 帧 |

本机 FlashAttention wheel 不包含 B300/SM100 kernel，因此 dense baseline 使用 PyTorch SDPA。FastVideo VSA 0.3.2 已在 `.b_rdma` 安装，并先通过独立 GPU kernel smoke test。实际 DMD+VSA 日志确认：

```text
seq_len=27280
dit_seq_shape=(31, 40, 22)
heads=24
head_size=128
topk=64
block_size=(4, 8, 8)
```

端到端实测加速约：

$$
63.48/14.00\approx4.53\times
$$

denoise loop 实测加速约：

$$
57.3/8.0\approx7.2\times
$$

这不是纯 DMD 的隔离加速比，因为两边同时存在 attention backend 与 CFG 设置差异。画质结论也不能只靠单个样例，应继续做多 prompt、相同 seed 的人工盲评和 VBench/CLIP 类指标。原生模型卡官方 Diffusers 示例为 50 steps、guidance 5.0；本次 30 steps 是按工程对照需求运行。

## 10. 结论

当前同一份 `Wan22Pipeline` 已同时支持：

```text
FastVideo/FastWan2.2-TI2V-5B-Diffusers
    -> 自动 DMD 3-step
    -> FASTVIDEO_VSA topk=64
    -> checkpoint 有 gate: learned
    -> checkpoint 无 gate: zero / sparse-only

Wan-AI/Wan2.2-TI2V-5B-Diffusers
    -> 原生 UniPC/Euler 多步调度
    -> dense attention baseline
```

用户不需要设置 DMD timestep、gate 模式或 gate fallback。模型元数据与 checkpoint 内容决定运行逻辑。
