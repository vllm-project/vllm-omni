# Z-Image（Tongyi-MAI/Z-Image-Turbo）在 vLLM-Omni 的 Tensor Parallel（TP）改造路线

本文档目标是把 **Z-Image Turbo** 在 vLLM-Omni 中逐步改造成 **可用且可维护的 TP=2**（优先）推理能力，并给出每阶段的验收标准与回滚策略。

> 约定：本文的 TP 指 **DiT（transformer）张量并行**；文本编码器/vae 的多卡策略单独讨论。

---

## Phase 0：基线与可复现（先跑通，再改造）

### 0.1 模型与环境准备

- 模型目录（本机示例）：`/root/autodl-tmp/models/z-image-turbo`
- 先确认 2 卡可用（示例）：`nvidia-smi -L`

### 0.2 先用 sglang 做 TP=2 冒烟（只验证“模型+权重+多卡链路”）

建议固定一个小分辨率与少步数，保证快速迭代：

```bash
CUDA_VISIBLE_DEVICES=0,1 sglang generate \
  --model-path /root/autodl-tmp/models/z-image-turbo \
  --tp-size 2 --num-gpus 2 \
  --num-inference-steps 6 \
  --height 512 --width 512 \
  --prompt "a cat reading a book" \
  --output-path outputs --output-file-name sglang_smoke_tp2.png \
  --save-output
```

验收标准：

- 能稳定出图（PNG 可打开，尺寸正确）
- 日志里能看到多卡初始化（`world_size=2` / `nccl`）
- 记录峰值显存（后续对比 vLLM-Omni 的目标）

> 说明：如果 sglang 的 Z-Image TP 跑不通，优先先定位是 **TP 线性层/attention 分片不一致** 还是 **NCCL/端口/环境** 问题；不要直接进入 vLLM-Omni 改造。

---

## Phase 1：vLLM-Omni 单卡（TP=1）跑通与锁定行为

目标：在不改模型代码的情况下，确认 vLLM-Omni 的 Z-Image Turbo **单卡** 推理可用，并把“输入/输出/显存/耗时”固化为基线。

推荐用仓库自带脚本：

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --num_inference_steps 2 \
  --height 256 --width 256 \
  --prompt "a cat reading a book"
```

验收标准：

- 生成成功（输出图片可打开）
- `pytest tests/e2e/offline_inference/test_t2i_model.py`（如环境允许）可跑通至少单卡分支

涉及文件（后续改造会动到）：

- `vllm_omni/diffusion/models/z_image/pipeline_z_image.py`
- `vllm_omni/diffusion/models/z_image/z_image_transformer.py`

---

## Phase 2：先加“约束检查 + 可观测性”（让失败更早、更可读）

目标：TP 改造前先把“会失败的配置”变成 **明确、可定位的报错**，避免跑到深处才爆炸。

建议新增/完善的检查点：

1. **TP 可用性检查（构造期）**
   - `tensor_parallel_size` 是否 > 0
   - `hidden_size(dim)` 是否可被 `tp` 整除（常见线性层需要）
   - `num_heads` 是否可被 `tp` 整除（attention 头分片需要）
   - 关键 projection（例如最终 patch 投影的 out_features）如果采用列并行，需要满足 out_features % tp == 0

2. **配置链路检查**
   - `DiffusionParallelConfig.tensor_parallel_size` 是否真正传到 Z-Image Transformer 构造与线性层（避免 hardcode `disable_tp=True`）

3. **关键日志**
   - 启动时打印：tp、head_dim、local_heads、各大模块是否启用 TP
   - 失败时提示：建议的可用 TP（例如 `tp in {1,2}`）

验收标准：

- TP=3/4 等不支持配置能在初始化阶段直接报出“原因 + 建议值”

---

## Phase 3：DiT（ZImageTransformer2DModel）TP 改造（核心阶段）

目标：让 Z-Image 的 **Transformer/DiT** 在 vLLM-Omni 中真正启用 TP（至少 TP=2）。

### 3.1 改造顺序建议（从最稳定的线性层开始）

1. **FFN（FeedForward）**
   - 使用 vLLM 的 `MergedColumnParallelLinear` + `RowParallelLinear`
   - 去掉 `disable_tp=True`，并确保 `gather_output / input_is_parallel` 的组合一致

2. **Attention 的 QKV**
   - 替换/启用 `QKVParallelLinear` 的 TP（去掉 `disable_tp=True`）
   - 确认每 rank 的 `local_num_heads = num_heads / tp`
   - 注意 `num_kv_heads` 的分片策略（若保持等于 `num_heads`，相对简单）

3. **Attention 的输出投影（to_out）**
   - 用 `RowParallelLinear`（常见做法：输入是并行的 attention 输出）
   - 确保 attention 输出的 hidden 维度在 TP 语义下是 **sharded** 还是 **replicated**
   - 如果 attention 输出仍是 full hidden（replicated），但 to_out 选择 RowParallel，则需要明确 `input_is_parallel=False`（由层内部切分输入）

4. **Embedder / FinalLayer**
   - `all_x_embedder`、`cap_embedder`、`FinalLayer.linear` 要选择合适的并行方式
   - 推荐优先保证正确性：可以先保持这些层 replicated，等主干 TP 稳定后再逐层并行化

### 3.2 权重加载与兼容性

`ZImageTransformer2DModel.load_weights()` 里有 stacked 权重映射（QKV、FFN）。启用 TP 后必须确保：

- 每个并行线性层的 `weight_loader` 与 shard 规则匹配
- 不修改 `state_dict` 形状（除非明确只支持特定 TP 并更新文档/报错）

验收标准：

- TP=1 输出与旧实现一致（至少数值范围/图像结构一致）
- TP=2 能稳定出图（小步数、小分辨率）
- 显存随 TP 有下降趋势（至少 DiT 权重部分）

---

## Phase 4：文本编码器（Qwen3 text encoder）去重/卸载（解决“每 rank 一份 ~15GB”）

目标：避免每个 TP rank 在 GPU 上各放一份 text encoder。

可选方案（按推荐顺序）：

### 方案 A（推荐）：rank0 编码 + broadcast prompt_embeds

- 只在 rank0 执行 text encoder（可放 GPU 或 CPU）
- 将生成的 `prompt_embeds`（以及 negative prompt）广播给所有 rank
- 需要处理 **变长序列**：建议先 pad 到同长度再 broadcast，再按长度切回 list

优点：实现相对直接；text encoder 只跑一次；显存节省巨大。  
风险：需要额外通信；要小心 batch/多 prompt 场景。

### 方案 B：CPU 常驻 + 每次编码时临时搬到主卡

- text encoder 常驻 CPU
- 编码时把模型或部分层搬到 GPU（或直接 CPU 推理）

优点：实现简单；显存最省。  
缺点：延迟明显增加（但通常比 OOM 更好）。

### 方案 C：FSDP/TP 分片 text encoder

优点：理论最优。  
缺点：工程复杂度高（优先级最低）。

验收标准：

- TP=2 时 text encoder GPU 显存不随 rank 线性翻倍
- 单次请求的编码结果在多 rank 一致（数值一致或可接受误差）

---

## Phase 5：测试与回归（让 TP 改造可持续）

目标：最小化引入回归，并让 CI/本地能快速发现问题。

建议补齐：

1. **最小步数推理测试（TP=1/2）**
   - `num_inference_steps=2`
   - `height=256,width=256`
   - 断言：能生成、尺寸正确、无异常

2. **配置错误测试**
   - TP=3/4 等不支持场景必须提前报错，且报错信息包含关键原因（heads 不整除 / dim 不整除 / projection 不整除等）

3. **显存回归（可选）**
   - 记录峰值显存（至少打印日志），用于对比

---

## Phase 6：性能与工程化收尾

- 对比 sglang 基线：吞吐、延迟、峰值显存
- 若 TP=2 稳定后，再考虑支持更多 TP（需要更严格的整除与更多层并行化）
- 最终把支持矩阵更新到文档：`docs/user_guide/acceleration/parallelism_acceleration.md`

