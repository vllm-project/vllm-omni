# 用问题完全掌握 LingBot World 2.0 PR C

本文用“提问—回答”的方式解释 PR C：LingBot World 2.0 如何从
request-local direct pipeline 接入通用 `ARDiffusionEngine`。阅读代码时，
建议依次对照：

- `vllm_omni/experimental/ar_diffusion/tick_protocol.py`
- `vllm_omni/experimental/ar_diffusion/runner.py`
- `vllm_omni/experimental/ar_diffusion/kv_cache/state.py`
- `vllm_omni/diffusion/models/wan2_2/pipeline_lingbot_world.py`
- `vllm_omni/diffusion/models/wan2_2/lingbot_world_transformer.py`

## 1. PR C 到底解决什么问题？

PR #5022 已能在一次 `forward()` 中生成多个 AR blocks，但 cache 只属于
该次 request。PR C 把“一次请求生成完整视频”改造成：

```text
同一个 session_id
  request/chunk 0 -> 生成 3 latent frames -> commit KV
  request/chunk 1 -> 复用历史 KV -> 生成 3 latent frames -> commit KV
  request/chunk 2 -> ...
```

每次 request 只计算一个 block，持续世界由 `session_id` 串起来。

## 2. 为什么不重新定义 LingBotChunkRequest？

PR C 和 PR D 共用通用 `ARDiffusionTickRequest`：

```python
sampling_params.extra_args["ar_diffusion_tick"]
```

它只描述 session、chunk、event snapshot 和 model-defined controls。
LingBot adapter 解释 camera schema，但 runtime/API 不理解 camera 数学。
这样下一种世界模型可以复用 session/event 层。

## 3. request_id、session_id、event_id、chunk_index 各自是什么？

- `request_id`：一次 engine 计算；外层
  `OmniDiffusionRequest.request_id` 是权威值。
- `session_id`：持续世界，拥有跨 request KV 和模型状态。
- `event_id`：一次被 API 接受的交互事件；由 PR D 排序和去重。
- `chunk_index`：session 中 AR block 的位置，从 0 严格连续增长。

tick 内嵌的 `request_id` 必须等于外层 request ID，否则 runner 拒绝。

## 4. 一个合法的 tick request 如何构造？

模型 prompt 仍走标准 `request.prompt`，不是藏进 `extra_args`：

```python
tick = ARDiffusionTickRequest(
    session_id="world-42",
    request_id="world-42:chunk-3",
    chunk_index=3,
    applied_event_ids=(17, 18),
    prompt="move through the room",
    controls=(
        ARDiffusionControlInput(
            track="camera",
            schema="lingbot.camera_trajectory.v1",
            data={
                "poses": poses,          # [pixel_frames, 4, 4]
                "intrinsics": intrinsics # [pixel_frames, 4]
            },
        ),
    ),
)

request = OmniDiffusionRequest(
    request_id=tick.request_id,
    prompt={
        "prompt": tick.prompt,
        "multi_modal_data": {"image": initial_image},
    },
    sampling_params=OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=9,
        num_inference_steps=4,
        output_type="latent",
        extra_args=tick.to_extra_args(),
    ),
)
```

当前一个 block 是 3 latent frames。Wan causal VAE temporal factor 为 4，
因此 request 的 pixel-frame geometry 是：

```text
(3 - 1) * 4 + 1 = 9 pixel frames
```

## 5. 为什么 realtime tick 当前要求 output_type="latent"？

直接对每个 block 独立调用 stateless `vae.decode()`，不能保证与
`decode(concat(all_latent_blocks))` 一致。正确的增量 VAE 需要持久
decoder state、overlap 和 crop 规则，属于 PR E。

因此 PR C 先保证 DiT/KV correctness，返回 latent block，不伪装成已经
完成 pixel-frame streaming。

## 6. 输出 envelope 是什么样？

pipeline 返回标准 `DiffusionOutput`：

```python
DiffusionOutput(
    output={
        "payload": {
            "latents": latent_block,  # [1, 16, 3, H_latent, W_latent]
        },
        "metadata": {
            "ar_diffusion": {
                "session_id": "world-42",
                "request_id": "world-42:chunk-3",
                "chunk_index": 3,
                "applied_event_ids": [17, 18],
            },
        },
    },
)
```

PR D 只在 metadata 与其 immutable snapshot 完全一致时提交 accepted
events。它不会从 payload 猜测 event，也不会伪造 metadata。

## 7. `ar_diffusion_kv_cache_spec()` 声明了什么？

LingBot capability 声明：

- 一个逻辑 KV branch：`main -> local_index 0`；
- 40 层（由 checkpoint config 决定）；
- TP-local KV heads；
- head size 128；
- 每个 block 3 个 frame pages；
- text cross-attention 固定 512 tokens；
- sink 和 recent sliding window；
- 固定 realtime resolution。

`tokens_per_frame` 由以下 geometry 计算：

```text
pixel H/W
  / VAE spatial factor 8
  / DiT patch H/W 2
  => post-patch tokens per latent frame
```

## 8. 为什么 realtime resolution 必须固定？

#5271 的 paged pool 在 model load 时预分配，page/block size 必须提前知道。
LingBot 每个 latent frame 的 token 数取决于 H×W，所以同一个 runner
不能在 pool 建好后任意改变 resolution。

部署通过：

```yaml
model_config:
  ar_diffusion_height: 480
  ar_diffusion_width: 832
```

请求 resolution 不一致会在计算前失败。

## 9. sink/window 为什么不是简单照抄 18 和 9？

#5022 direct cache 的总容量是 18 frames，其中前 9 frames 是 sink。
#5271 capability 将二者分开表示：

```text
sink_frames = 9
recent_window_frames = total_capacity 18 - sink 9 = 9
```

paged attention 最终可见集合是 sink + recent tail + 当前未提交 block，
与 direct cache 的保留语义对齐。

## 10. 四个 diffusion steps 如何使用 cache？

一个 block 有四次 noisy DMD probe 和一次 clean commit forward：

```text
step 0: scratch page，commit_current=False
step 1: scratch page，commit_current=False
step 2: scratch page，commit_current=False
step 3: scratch page，commit_current=False
clean:  managed page，commit_current=True
        -> commit_paged_context("main")
```

noisy probe 可以看到“历史 + 当前 noisy block”，但不能污染持久历史。
只有最终 clean latent 对应的 K/V 被提交一次。

## 11. transformer 中 direct 和 paged 两条路径如何共存？

`LingBotSelfAttention.forward()` 接受两种 cache：

- `LingBotAttentionCache`：#5022 direct contiguous tensor；
- `ARDiffusionPagedLayerInputs`：runner-owned paged pool metadata。

direct path 继续调用 `_update_cache()` 和普通 `Attention`；paged path
调用 `paged_write_attn()`，将当前 K/V 写到 engine slot 后按 block table
执行 attention。权重、RoPE、QKV projection、DiT block 数学不变。

## 12. text cross-KV 由谁拥有？

runner 的 `ARDiffusionKVState` 拥有名为 `text` 的 per-session pool。
pipeline 第一次看到 prompt 时逐层投影 K/V 并 populate；后续 tick
直接读取，不再重复投影。

pipeline 只在一次 forward 的 bind context 中临时持有 state 引用，
不能把 runner state 永久保存到模型对象。

## 13. prompt update 会发生什么？

模型侧 `_LingBotARSessionState` 记录上一条 prompt：

```text
prompt 未变 -> 复用 text cross-KV
prompt 改变 -> clear_cross_attention()
             -> 保留 self-KV/world history
             -> 重新 encode 并逐层 populate text K/V
```

prompt 更新描述的是从当前世界历史继续生成时的新条件，不等于重置世界。

## 14. source image 为什么只有 chunk 0 生效？

direct full-video condition 只在第一个 latent frame 设置 temporal mask 和
source-image latent。后续 block 的 source image 信息来自 causal self-KV。

因此 PR C 在首个 tick 将“首帧图像 + 后续零帧”的最大 causal horizon
一次性 VAE encode，保存在模型 session state 中；每个 tick 只取对应的
3-latent-frame slice。这样既不会把每个 request 的 image 当作新初始帧
重复注入，也不会错误丢掉 causal VAE temporal receptive field 在后续
condition latent 中传播的内容。

## 15. RNG 为什么也是 session state？

direct multi-block 在一个 `torch.Generator` 上连续取样。如果每个 request
重新从相同 seed 开始，chunk 1 会错误地复用 chunk 0 的 noise。

PR C 在成功 tick 后保存 `generator.get_state()`，下个 tick 先恢复：

```text
direct: rand block0 -> rand block1 -> rand block2
tick:   request0 rand block0
        request1 restore -> rand block1
        request2 restore -> rand block2
```

这就是“确定性的多 chunk 上游 source parity”的一部分。

## 16. camera control 在哪一层解释？

PR D 只保存：

```text
track="camera"
schema="lingbot.camera_trajectory.v1"
data={...}
```

LingBot preprocess adapter 负责：

- 校验 poses `[frames, 4, 4]`；
- 校验 intrinsics `[frames, 4]`；
- 有限数、frame 数一致；
- 转成 `CameraTrajectory`；
- 插值到 latent frames；
- 生成 Plücker embedding 并 spatial fold。

未来 WASD、pose、velocity 等 schema 也应在模型 adapter 内转换，不能让
公共 API 固化 LingBot 坐标系。

## 17. reset、close、failure 如何清理？

runner 是 lifecycle owner：

- reset：释放 self/cross KV，并调用
  `pipeline.reset_ar_diffusion_session(session_id)`；
- close/LRU：释放 KV，并调用 close hook；
- forward exception：走统一 release path，清理 KV 和模型状态。

pipeline hook 删除：

- `next_chunk_index`；
- prompt；
- RNG state。
- causal image-condition horizon。

reset 后 chunk 从 0 开始。FAILED session 不能原 chunk 就地 retry，只能
显式 reset 或 close；event ID 的 API 去重序列不因 model reset 重用。

## 18. 为什么 chunk_index 必须连续？

paged KV、RoPE temporal position 和 world state 必须指向同一个时间：

```text
expected chunk 3，却收到 chunk 5
```

如果允许，会让 RoPE 跳跃、camera 时间和 KV history 不一致。因此 model
adapter 在任何 tensor 计算前检查 `next_chunk_index`。

## 19. direct path 还保留吗？

保留。没有 typed tick 时：

- 分配 request-local `LingBotTransformerCache`；
- 一次 forward 可生成多个 blocks；
- 最后 full VAE decode；
- 行为继续作为 correctness oracle。

PR C 不强迫普通 offline request 使用 AR engine，也不删除 #5022 的路径。

## 20. direct/tick parity 应比较什么？

固定 checkpoint、dtype、backend、image、prompt、camera、noise source 后：

1. 每个 DMD step 的 flow/x0；
2. 每个 block 的 clean latent；
3. start frame / RoPE position；
4. clean K/V commit 次数；
5. sink/recent visible history；
6. prompt cross-KV；
7. 拼接后的 latent；
8. 最终 full decode（PR E 前由 direct path统一 decode）。

CPU tests覆盖 contract/state/geometry；真实数值结论必须用同一 GPU backend
和真实 checkpoint 验证。

## 21. 当前有哪些重要测试？

- `tests/ar_diffusion/test_tick_protocol.py`
  - typed tick round-trip、非法 ID；
- `tests/ar_diffusion/test_capability_runner.py`
  - capability、session、typed runner parsing；
- `tests/ar_diffusion/test_pipeline_bridge.py`
  - paged commit、sink/window、cross-KV invalidation；
- `tests/diffusion/models/wan2_2/test_lingbot_world_attention.py`
  - direct attention 和 paged routing；
- `tests/diffusion/models/wan2_2/test_pipeline_lingbot_world.py`
  - capability geometry、camera control、单 block envelope、chunk/RNG state。

## 22. 本地如何验证？

快速 L1：

```bash
cd /Users/chenshengdong/.codex/worktrees/vllm-omni-lingbot-world-v2-integration
pytest -q \
  tests/ar_diffusion \
  tests/diffusion/models/wan2_2/test_lingbot_world_attention.py \
  tests/diffusion/models/wan2_2/test_pipeline_lingbot_world.py
```

CI-like：

```bash
pytest -s -v \
  tests/ar_diffusion \
  tests/diffusion/models/wan2_2 \
  -m "core_model and cpu" \
  --run-level=core_model
```

前置条件：Python 3.11/3.12、与当前 vLLM-Omni 匹配且已构建的 vLLM、
diffusion test dependencies。本机若只有未编译的 macOS vLLM source，
只能运行已有 stub 覆盖的测试及静态检查。

## 23. GPU 验收的最小矩阵是什么？

1. TP=1、eager、chunk 0；
2. TP=1、eager、至少 3 chunks；
3. prompt 在 chunk boundary 更新；
4. reset 后重新从 chunk 0；
5. 人工触发 forward failure，确认 pool 恢复；
6. 超过 window 的长 session，确认 sink 和 recent tail；
7. direct 与 tick latent parity；
8. TP=2；
9. compile/CUDA graph。

真实 checkpoint parity 和长 session GPU 测试未通过前，不能宣称 realtime
correctness 完成。

## 24. review 时最应该盯哪些风险？

- `tokens_per_frame` 是否与实际 resolution 一致；
- sink 是否被重复计算进 window；
- noisy DMD probe 是否误 commit；
- chunk>0 是否重复注入 source image；
- RNG 是否跨 request 连续；
- prompt 更新是否只清 cross-KV；
- forward failure 是否清 model + self/cross KV；
- outer request ID 和 tick request ID 是否一致；
- metadata 是否在 postprocess/AsyncOmni 链路中保留；
- 是否把 camera/WASD 语义泄漏到公共 runtime；
- 是否误把 stateless per-block VAE decode 当成 streaming VAE。

## 25. PR C 完成后还缺什么？

- PR D：WebSocket/session/event control plane 与 worker lifecycle RPC；
- PR E：stateful streaming VAE 和 pixel-frame transport；
- 真实 checkpoint direct/tick GPU parity fixture；
- 性能工作：compile、TP/USP、kernel、VAE overlap。

PR C 只负责建立正确的 model/runtime tick bridge，不以 API 或性能优化扩大
review 范围。
