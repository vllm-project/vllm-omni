# LingBot World 2.0 实时推理集成实施计划

## 1. 文档目的

本文是 LingBot World 2.0 在 vLLM-Omni 上实现实时推理的执行文档。它同时承担：

- integration 分支的范围说明；
- 上游 PR 依赖和合并策略；
- 模型、runtime、API 之间的接口约定；
- 分阶段实施顺序；
- 每个阶段的验收标准和测试命令；
- 并行开发和 Codex session 的切分依据。

实现过程中如发现设计需要调整，应先更新本文，再修改代码。本文不把性能优化作为第一阶段目标；第一阶段只建立正确、可测试、可持续演进的实时推理链路。

## 2. 当前基线

### 2.1 Integration worktree

```text
路径：
/Users/chenshengdong/.codex/worktrees/vllm-omni-lingbot-world-v2-integration

分支：
codex/lingbot-world-v2-realtime-integration

起点：
upstream main @ b55120c93b868e8f8c4c34c8e26cb68c2c638340
```

该 worktree 与 `main`、已有 PR review worktree 和旧的 realtime integration worktree相互独立。

### 2.2 上游能力来源

| 层次 | 来源 | 当前用途 |
|---|---|---|
| 通用 AR session/runtime | PR #5271 | `ARDiffusionEngine`、capability、paged KV、session 生命周期 |
| LingBot World 2.0 模型基线 | PR #5022 | checkpoint、T5、VAE、camera、causal DMD、direct request-local KV |
| Prompt/interaction 传递基础 | PR #4652 / RFC #5120 | WebSocket control plane、interaction 消息和 worker 路由的设计参考 |
| LingBot-Video | PR #5035 / #5311 | 只参考 API、image handling 和 parity 方法，不作为 realtime world 依赖 |

### 2.3 “已经支持”的准确含义

PR #5022 已经实现：

- LingBot World 2.0 checkpoint-compatible pipeline；
- image + camera trajectory conditioning；
- 3 latent frames/AR block；
- 4-step causal DMD；
- 单次 `forward()` 内连续生成多个 AR blocks；
- request-local self/cross attention cache；
- 最后一次性 VAE decode。

它尚未实现：

- 每个 AR block 一个独立 engine request；
- 跨 request 的 `session_id`；
- runner-owned/paged AR KV；
- runtime event 和 `event_id`；
- prompt 更新后的 text cross-KV invalidation；
- 增量 VAE decode；
- WebSocket disconnect/reset/close 到 worker session 的生命周期闭环。

因此 PR #5022 是 direct/offline correctness baseline，不是 realtime serving 完成态。

## 3. 目标调用链

```text
WebSocket / Python control client
  -> World session control plane
     -> pending interaction queue
     -> chunk-boundary immutable snapshot
     -> LingBotWorldTickRequest
  -> AsyncOmni / DiffusionEngine
     -> one OmniDiffusionRequest per AR block
  -> ARDiffusionModelRunner
     -> session_id lookup
     -> bind ARDiffusionKVState
  -> LingBotWorldCausalDMDPipeline.forward_tick()
     -> prepare prompt/image/camera conditioning
     -> four-step causal DMD for one AR block
     -> commit self-attention KV once
     -> maintain/invalidate text cross-attention KV
     -> decode current block incrementally
  -> LingBotChunkResult / DiffusionOutput
     -> frame_batch metadata
     -> applied_event_ids
     -> binary payload
```

## 4. 核心身份和时间单位

| 字段 | 标识对象 | 生命周期 |
|---|---|---|
| `session_id` | 持续演化的世界 | init 到 reset/close/eviction/failure |
| `request_id` | 一次 AR block 计算 | 单个 engine request |
| `event_id` | 一次 prompt/action/composite 输入 | session 内唯一，用于去重和 accepted/applied 关联 |
| `chunk_index` | AR block 的逻辑序号 | session 内从 0 单调递增 |
| `transport_chunk_index` | 编码/传输 payload 序号 | stream 内单调递增，未必等于 `chunk_index` |

时间单位必须显式区分：

- pixel frame；
- latent frame；
- AR block；
- diffusion step；
- transport chunk。

一个 AR block 包含 3 个 latent frames。一个 block 的 4 个 diffusion steps 反复更新同一组 latent frames；只有 block 成功完成后才允许提交一次持久 KV 和 world state。

## 5. PR 拆分

### PR A：通用 AR runtime

来源：#5271。

要求：

- generic runtime 不依赖 DreamZero 私有字段；
- capability 描述单/多 branch、TP-local heads、window/sink、named cross-KV；
- session reset/close/LRU/failure cleanup；
- runner 生命周期可以被 serving 层调用；
- 修复或明确 DreamZero state 强引用、共享 local slot 测试和 pool-pressure eviction。

### PR B：LingBot direct/offline baseline

来源：#5022。

要求：

- 保留现有 checkpoint 和模型定义；
- 保留 request-local direct KV；
- 保留完整 offline generation；
- direct path 作为后续 paged/tick path 的 correctness oracle；
- 不把 realtime API 直接塞进现有 `/v1/videos` model extras。

### PR C：LingBot ARDiffusion adapter

范围：

- 将一个 AR block 抽成独立 tick；
- LingBot pipeline 实现 #5271 capability；
- self-attention 使用 runner-owned/paged KV；
- text cross-attention 使用 named session KV；
- 管理 LingBot 非 KV session state；
- direct multi-block 与多 request tick parity；
- reset/close/failure cleanup。

主要目录：

```text
vllm_omni/diffusion/models/wan2_2/
vllm_omni/experimental/ar_diffusion/
tests/diffusion/models/wan2_2/
tests/ar_diffusion/
```

### PR D：Realtime world session/API

范围：

- WebSocket session create/stop/disconnect；
- `session_id`、`event_id`；
- bounded event queue、排序和去重；
- chunk-boundary immutable snapshot；
- interaction accepted 与 chunk applied 分离；
- 将 snapshot 转换为内部 tick request；
- worker reset/close RPC；
- frame metadata 和错误语义。

主要目录：

```text
vllm_omni/entrypoints/
vllm_omni/engine/
vllm_omni/diffusion/stage_*.py
vllm_omni/diffusion/worker/
tests/entrypoints/
tests/engine/
```

PR D 不解析 LingBot camera 数学，不直接调用 pipeline 方法，不持有 CUDA tensor。

### PR E：Streaming VAE/output

范围：

- persistent VAE decoder state；
- `decode_chunk()`；
- overlap/crop 和首帧规则；
- latent block 到新增 pixel frames 的精确映射；
- frame-batch transport。

### PR F：性能优化

在 correctness gates 全部通过后再做：

- TP/USP；
- compile/CUDA graph；
- paged attention kernel；
- VAE parallel decode；
- overlap generation/encode/send；
- profiling 和吞吐/延迟优化。

## 6. PR C 与 PR D 的共享 contract

并发前必须冻结以下最小内部类型。当前 integration 分支已将其实现为
`vllm_omni.experimental.ar_diffusion.tick_protocol` 中的通用类型，避免
API 层和 LingBot 模型层各自发明一套字段：

```python
@dataclass(frozen=True)
class ARDiffusionControlInput:
    track: str
    schema: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class ARDiffusionTickRequest:
    session_id: str
    request_id: str
    chunk_index: int
    applied_event_ids: tuple[int, ...]
    prompt: str | None
    controls: tuple[ARDiffusionControlInput, ...]
    reset: bool = False
    close_session: bool = False


@dataclass(frozen=True)
class ARDiffusionChunkMetadata:
    session_id: str
    request_id: str
    chunk_index: int
    applied_event_ids: tuple[int, ...]
```

序列化后的 tick 统一放在
`sampling_params.extra_args["ar_diffusion_tick"]`，由
`ARDiffusionModelRunner` 校验并解析。`request_id` 必须与
`OmniDiffusionRequest.request_id` 一致。旧的 DreamZero 顶层
`session_id/reset/close_session` 暂时保留兼容。

模型输出仍使用 `DiffusionOutput`；chunk identity metadata 与真正的
latent/pixel payload 分开，后续放入标准 output envelope，而不是新增
一个绕开 diffusion output formatter 的返回类型。

### 6.1 公共层负责

- session/event 身份；
- 排序、去重和 backpressure；
- composite 原子入队；
- chunk 开始前 snapshot；
- accepted/applied/error；
- reset/close；
- 原样保存 schema payload。

### 6.2 LingBot adapter 负责

- schema capability；
- WASD/key state、pose、trajectory 等输入的模型语义；
- camera 坐标系和单位；
- 按 latent frame 构造 conditioning；
- prompt encoding；
- text cross-KV invalidation；
- 模型非 KV session state；
- tick 成功后的模型状态 commit。

公共层不应把 `target/velocity` 的插值和积分提前固化。它们可以是具体 schema 的语义。

## 7. PR C/D 是否可以并发

结论：可以，但只能在“先串行冻结 contract、后并行实现”的方式下进行。

### 7.1 不应立即并发的部分

以下工作应由 integration owner 在当前 session 先完成：

1. 叠加 #5271 和 #5022；
2. 解决编译和 registry 冲突；
3. 确认一块 tick 的 request/output 字段；
4. 冻结 `session_id/request_id/event_id/chunk_index`；
5. 冻结 reset/close/failure 行为；
6. 写下 contract tests。

如果在此之前并发，PR C 会把字段放进 `extra_args`，PR D 可能新增另一套 message/dataclass，最终在 worker 边界重新返工。

### 7.2 可以并发的部分

contract 冻结后：

- PR C 独立开发模型 tick、capability、paged KV 和 model session state；
- PR D 独立开发 WebSocket、event queue、control routing 和 lifecycle RPC；
- 两边通过 fake tick consumer/provider 做 CPU contract test；
- 最后在 integration 分支联调。

### 7.3 Codex session 建议

- 当前 Codex session：负责 integration owner、PR A/B 叠加、共享 contract、PR C 和最终联调。
- 新 Codex session：在 contract commit 后负责 PR D；输入必须指向本文、固定 base commit 和明确验收标准。
- 不建议为 PR C 再开新 session：PR C 与 #5022/#5271 的上下文和冲突最多，由当前 session 连续处理更安全。
- PR E 后续再开独立 session，因为 VAE state/transport 是新的、边界明确的子系统。

每个 session 使用独立 branch/worktree。PR D 分支应基于“包含共享 contract、但不包含 PR C 模型实现”的 integration contract commit，降低代码交叉。

## 8. 实施里程碑

### M0：可构建的叠加基线

任务：

- 将 #5271 和 #5022 叠加到 integration branch；
- 处理最新 main 冲突；
- 不改变模型行为；
- 运行 import、compile 和已有 CPU targeted tests。

验收：

- `LingBotWorldCausalDMDPipeline` 可注册；
- `ARDiffusionEngine` 可注册；
- 普通 diffusion engine routing 不回归；
- direct LingBot unit tests 和 ARDiffusion unit tests 可共同收集。

### M1：冻结 direct oracle

任务：

- 固定 checkpoint revision、initial image、prompt、poses/intrinsics；
- 固定每 block noise 和 4-step schedule；
- 暴露或捕获两个 block 的中间 tensor。

至少记录：

- prompt embedding；
- camera conditioning；
- 每个 diffusion step 的 latent/x0；
- 每个 block clean latent；
- committed KV 的可比较摘要或 tensor；
- full VAE decode。

验收：

- 同 seed 重跑 deterministic；
- fixture 来源、dtype、backend 和容差写入测试；
- fixture 不是仅靠视觉判断。

### M2：Direct single-tick seam

任务：

- 抽取 `forward_tick()`/等价私有方法；
- 一次只生成一个 3-latent-frame block；
- 暂时继续使用 direct contiguous KV；
- offline path 通过循环调用 tick 复用。

验收：

```text
旧 direct multi-block
==
新 direct tick 连续调用 N 次
```

对齐项：

- block latent；
- DMD step trace；
- start frame；
- cache commit；
- full concatenated decode。

### M3：ARDiffusion capability

任务：

- 实现 `ar_diffusion_kv_cache_spec()`；
- 单逻辑 branch `main -> local_index=0`；
- `frames_per_block=3`；
- TP-local KV heads；
- sink/window；
- named text cross-KV；
- bind/reset/close hooks；
- 移除 tick path 对 `allocate_cache()` 的依赖。

验收：

- 多个独立 `request_id` 使用同一 `session_id`；
- 生成结果与 M2 direct tick 对齐；
- reset 后重新从干净状态开始；
- close/LRU/failure 不泄漏 KV 或 model state；
- prompt 变化不会复用旧 text cross-KV。

### M4：最小 realtime control plane

Phase 1 只支持：

- session start/stop；
- next-chunk application；
- prompt hard switch；
- camera schema payload；
- event accepted；
- `applied_event_ids`；
- bounded queue；
- disconnect close。

暂不支持：

- future scheduling；
- easing；
- batch scripts；
- reconnect；
- 多 producer；
- 通用 target/velocity state machine。

### M5：Streaming VAE

任务：

- incremental decoder state；
- chunk decode；
- 精确新增帧；
- output metadata。

验收：

```text
full_decode(concat(latent_blocks))
≈
concat(streaming_decode(block_i).new_frames)
```

### M6：GPU 和性能

顺序：

1. TP=1 eager；
2. TP=1 compile；
3. TP=2；
4. 8+ chunks；
5. 长窗口；
6. 多 session；
7. profiling；
8. 性能优化。

## 9. 测试策略

### 9.1 L1：required

所有纯 contract、状态机和 tensor geometry bug 都必须有 CPU deterministic test：

- capability geometry；
- tick request validation；
- session reuse/reset/close；
- event snapshot；
- duplicate/stale event；
- composite atomicity；
- direct tick trace；
- prompt cross-KV invalidation；
- forward failure cleanup；
- sink/window。

Markers：

```python
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
]
```

L1 使用 `monkeypatch` 或 `mocker`，不使用 `unittest.mock`。

### 9.2 L2/L3：recommended，模型链路完成后 required

- 最小真实 checkpoint offline tick；
- 最小 WebSocket session；
- 两个 chunk；
- 一个 camera event；
- 一个 prompt event；
- reset/close。

Markers：

```python
pytest.mark.core_model
pytest.mark.advanced_model
pytest.mark.diffusion
```

### 9.3 L4：required before claiming realtime support

- 真实 14B checkpoint；
- H100/H200/B300；
- 480x832；
- 4-step DMD；
- 长 session；
- sliding window；
- TP/parallel target configuration；
- direct/reference parity；
- streaming VAE parity；
- latency和显存记录。

## 10. 初始测试命令

在 M0 叠加完成后执行：

```bash
python -m pytest -q \
  tests/ar_diffusion \
  tests/diffusion/models/wan2_2/test_lingbot_world_attention.py \
  tests/diffusion/models/wan2_2/test_lingbot_world_camera.py \
  tests/diffusion/models/wan2_2/test_lingbot_world_transformer.py \
  tests/diffusion/models/wan2_2/test_pipeline_lingbot_world.py
```

静态检查：

```bash
ruff check \
  vllm_omni/experimental/ar_diffusion \
  vllm_omni/diffusion/models/wan2_2 \
  tests/ar_diffusion \
  tests/diffusion/models/wan2_2

git diff --check origin/main...HEAD
```

最小测试的前置条件：

- Python 3.11 或 3.12；
- 与当前 main 匹配的 vLLM；
- CPU unit tests 所需的 diffusion dependencies；
- GPU E2E 另需真实 LingBot World 2.0 checkpoint、初始图像和 camera trajectory。

## 11. 人员分工

### 陈圣东

- 模型提供方和 correctness owner；
- 冻结 official/direct fixture；
- camera/prompt/control contract；
- 模型输出质量和数值验收；
- 决定哪些模型语义可以成为公共 abstraction。

### 黄泽宇

- API 和 session control plane；
- WebSocket lifecycle；
- event queue、accepted/applied、backpressure；
- typed tick request；
- worker lifecycle RPC；
- API contract tests。

### Codex integration owner

- integration worktree；
- #5271/#5022 叠加；
- PR C 实现；
- shared contract；
- direct/paged parity tests；
- PR C/D 联调；
- 记录测试结果和未完成的 GPU 验收。

郑文钢退出后，#5271 到 LingBot 的适配、runtime cleanup 和初始性能诊断不再作为外部隐含依赖，全部纳入本文的显式范围。

## 12. 非目标

第一阶段不做：

- LingBot-Video #5311 功能合并；
- 通用世界模型大一统抽象；
- 多模型 target/velocity 状态机；
- 无 correctness oracle 的性能优化；
- 为追求单个 PR 独立而复制 #5022 transformer；
- 将 realtime protocol 塞入普通 `/v1/videos` extra args；
- 在 FastAPI 进程保存 CUDA/session model state。

## 13. 完成定义

只有同时满足以下条件，才能宣称 vLLM-Omni 支持 LingBot World 2.0 realtime inference：

1. 一个 session 通过多个独立 request 连续生成 AR blocks；
2. direct 与 ARDiffusion tick 在冻结输入下数值对齐；
3. camera/prompt event 只在 chunk boundary 原子生效；
4. prompt 更新正确重建 text embedding/cross-KV；
5. KV、VAE 和模型状态按 session reset/close/failure；
6. 每个成功 chunk 返回正确的 `applied_event_ids`；
7. streaming VAE 与 full decode 在规定容差内对齐；
8. WebSocket disconnect 可以释放 owning worker session；
9. 真实 checkpoint GPU E2E 和长 session 测试通过；
10. 性能指标被记录，但性能优化不改变 correctness baseline。

## 14. Integration 执行记录

### 2026-07-23：基线和 contract

- 新建独立 worktree：
  `/Users/chenshengdong/.codex/worktrees/vllm-omni-lingbot-world-v2-integration`；
- 基于 `origin/main@b55120c9`；
- 先后合入 #5271 head `ecc18c62` 和 #5022 head `be7770f0`；
- 两个 PR 在各自 merge-base 上没有同名变更文件，叠加无冲突；
- 新增通用 typed tick contract、round-trip/negative L1 tests；
- `ARDiffusionModelRunner` 同时支持 typed tick 和旧顶层 session 字段。

已通过：

```bash
python3 -m py_compile \
  vllm_omni/experimental/ar_diffusion/tick_protocol.py \
  vllm_omni/experimental/ar_diffusion/runner.py \
  tests/ar_diffusion/test_tick_protocol.py
ruff check \
  vllm_omni/experimental/ar_diffusion/tick_protocol.py \
  vllm_omni/experimental/ar_diffusion/runner.py \
  tests/ar_diffusion/test_tick_protocol.py
git diff --check
```

本机完整 pytest 当前不能启动：系统 `pytest` 使用 Python 3.9 和旧版
vLLM，导入时缺少 `vllm.v1`；本仓库要求 Python 3.10 及以上并需要与
当前 vLLM-Omni 匹配的 vLLM build。配置正确环境后必须补跑：

```bash
pytest -q \
  tests/ar_diffusion \
  tests/diffusion/models/wan2_2/test_lingbot_world_typing.py \
  tests/diffusion/models/wan2_2/test_lingbot_world_camera.py
```
