# Native KV Append：正确性与性能复核（2026-09-06）

本轮范围是单张 H200、vLLM 0.28.0、MiniCPM-o 4.5 三阶段 native audio
链路。不是任意硬件/并发的生产签字，也不使用“完成 95%”之类比例代替证据。
保留版本的 native 正常路径和本轮选定的四项实机故障回归均已通过。

## 基准口径纠正

此前 Seed-TTS 试验配置了 `native_duplex=False`，测的是 Realtime chat fallback。
这些结果不能证明 KV Append 或 native Stage0 async 的性能。历史记录保留，相关
性能推论撤回；nightly 中也将 fallback 与 native 两条测试路线分开。

新的 `run_native_duplex_benchmark.py` 使用真实音频输入，并检查：

- 每个 session 的物理 `duplex-s.*` request ID 保持不变，session 之间不共享 ID；
- 固定 replica、递增的 v0.28 append receipt、正的 prompt/computed 差值；
- KV/context 单调且未越界；每轮有真实音频和模型 EOS；
- 所有指定回复完成后才计算有效吞吐，不能把 listen 当成已完成的语音回复。

当前负载：1/2/4 路、每路两轮、每轮取固定 WAV 的前 1400 ms、实时发送；
每档 2 次 warmup、3 次测量。context 只增长约 82→143 tokens，**不是长上下文
性能测试，也不是持续饱和流量的最大容量测试**。完整 WAV 长约 5.47 s。

吞吐计时包括客户端连接、关闭和产物处理，不包括服务器/模型启动。保留原始
responses/s，同时报告生成音频秒数/s，避免输出变短被误读为加速。
客户端 text delta 与音频同步，不能当成 Stage0 token TTFT；后者单独从
`engine_stage_metrics` 统计。H200 C1/C2 有实测基线和 10% 回退门槛；其他硬件和
C4 未录入性能门槛时仅收集证据，仍执行全部正确性断言。

## 已修复的问题

1. **RPC 超时未覆盖提交。** 有界 request queue 满时，原 `put()` 可以永久阻塞，
   后续 reply timeout 和 fatal broadcast 都解救不了提交线程。现在提交/回包共用
   一个单调时钟 deadline，阻塞提交分段检查 fatal/close；native 非阻塞提交不变。
2. **混合温度采样。** 全 batch 的 `all_greedy=False` 会让温度为 0 的 native
   请求随机决定 chunk 边界。边界与正文现在都使用请求自己的 greedy 判断。
3. **native 与普通 chat 混合 batch。** 原先会整体跳过 MiniCPM native policy。
   现在保留 chat 的标准采样/logprobs，覆盖 native 行的策略结果，并克隆标准采样
   中无用的 native generator，避免额外推进请求 RNG。真实 CUDA sampler 对照通过。
4. **入站 mailbox 在字节检查前无界增长。** 现在每个 attachment 限制 256 个外部
   事件、16 MiB 编码后积压，正常流量仍按同一 FIFO。过载明确返回
   `input_backpressure`、以 1013 关闭并回收该 session；不是静默丢输入继续运行。
   单独预留一个小型 EOF 标记，满队列时断连不会丢失。回收/健康 peer 回归通过。
5. **native 基准收集路径可能误选 fallback。** 独立配置/fixture 映射已修正；
   显式传入非 native 配置会在启动模型前失败。

## 保留的低风险优化与当前测量

- 采样从不可变 CPU request snapshot 读取 temperature/top-k/top-p，避免逐行读取
  GPU 标量。每次 append 的参数更新仍会刷新 snapshot。
- Talker 频次惩罚改用固定大小 `scatter_add_`，删除 Tensor 布尔值同步与动态
  `bincount`，保留频次、符号、窗口和分块 workspace 语义；缓存无请求状态的 Sampler。
- 利用已有 `omni_pooler_payload_include_hidden` hook，省掉
  Talker→Code2Wav 不使用的 hidden payload；Thinker→Talker 和设备上的采样 hidden
  states 不变。包含该修改的扩大 CPU 回归 975 项、native C1/C2/C4、四项实机
  故障回归均通过。

惩罚算子单独微基准（H200，float32、vocab=32000、window=64，交替 A/B，
10 组×100 次；不是整模型吞吐）：

| batch | 原实现 ms/次 | 新实现 ms/次 |
| --- | ---: | ---: |
| 1 | 0.1419 | 0.0907 |
| 4 | 0.1792 | 0.1262 |
| 32 | 0.5512 | 0.4716 |

输出逐位一致。虽然该算子单路/四路快约 30%～36%，每步只节约约 0.05 ms。

包含精简 hidden payload 与 mailbox 修复的整链路结果：

| 并发 | 原基线 responses/s | 修复后 responses/s | 原 TTFP ms | 修复后 TTFP ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.2920 | 0.2939 | 388.3 | 378.3 |
| 2 | 0.4758 | 0.4778 | 476.1 | 466.8 |
| 4 | 0.6135 | 0.6301 | 913.2 | 856.0 |

不能据此宣称显著吞吐提升：一、二路基本处于波动范围，四路跨次运行的 TTFP
中位数约 660～913 ms，且生成音频长度也需要共同核对。

## 没有默认启用的实验

- **current-unit embedding 少拷贝：撤回。** 一、二路只约 1% 波动，四路多次有
  一个第二轮回复未出声。真实 forward 的 token/embedding oracle 与原路径完全
  一致，但 strict speech gate 仍失败。现象是模型发出 listen，不是已证明的
  scheduler 死锁、KV 重复追加或 RPC 丢失；也不能据此声称根因已解决。
- **HiFT capture batches [1,2,4]：不改默认。** CUDA eager/graph、padding、输出
  所有权测试通过，native C1/C2/C4 也完成；但相对同轮保守基线没有稳定收益，
  四路 TTFP 反而较差。默认仍为 [1]。
- **Stage0 async-on：保持 opt-in。** 此次是真的 native audio，Stage0 日志确认
  `OmniARAsyncScheduler`。C1/C2 吞吐为 0.2921/0.4767，未优于同源码 async-off；
  双路 Stage0 ITL 约 11.1→9.9 ms，但 TTFP 460.8→469.3 ms。C4 strict speech gate
  失败，不能给这一档签字。关闭调度 lookahead 不会关闭全双工 I/O。

## 为什么 KV Append 不必带来吞吐倍增

v0.28 的底座仍使用 resumable request 保留 scheduler/worker 的请求状态。若原来的
pause/resume 已保留 KV，改为 append 不是把“每轮重算全部历史”变为“零历史成本”。
收益更多体现在显式增量提交、幂等确认、故障边界和多 session 生命周期管理。

当前短负载的 Stage0 首 token 是几十毫秒，而端到端首音是数百毫秒。完整 profile
还显示语音后段、预处理和 CPU/GPU 同步有明显成本；不能把这些时间都归因于 KV。
Stage0/1/2 的 inclusive scope 不能跨线程、跨嵌套直接相加算占比。

下一步应围绕明确 SLO 做长上下文、多轮持续输入和慢消费者压力测试，结合各 stage
的排队/处理耗时来选择优化；不是继续盲目打开 async 或改变精度/采样分布。

## 验证与边界

最终结果：包含 mailbox/精简 payload 的扩大 CPU 回归 975 passed，CUDA 专项
19 passed；另有 2 项故障注入作用域测试通过，禁止在不知道测试服务器 PID 树时
退回全机进程匹配。保留版本 native C1/C2/C4 均通过（含 warmup 共 35 个 session、
70 个语音回复；性能只统计测量轮次）。实机故障 **4 passed，2 deselected**：

- 强制 KV preemption（computed 89→0），历史重算后完成真实语音；
- 编码器坏输入只终止坏请求，健康 peer 正常出声；
- 已提交 append 丢回包后重试只执行一次，再杀死 replica 0 后在 replica 1 有界回放；
- SIGSTOP worker 造成 pending append，分别验证 cancel/close，最后均 SIGCONT 恢复。

本轮未重跑该模块单独的 context rollover 和单独 replica-recovery 两项；不要把
4 项选定回归说成整个 weekly 模块全量通过。测试结束 GPU 1/2 均恢复空闲，
未操作其他项目的 GPU 0 服务。23 个任务源文件的本地/实测快照 SHA256 一致。

原始日志、JUnit、JSON 和实验记录位于 workspace 的
`kv_append_native_perf_20260906/`，被拒绝的结果没有删除。

仍不提供 live KV migration、任意硬件/并发保证或无损无限上下文。
现有有界 replay/rollover 不能等同于完整历史逐位等价的迁移/压缩；H20/H100/NPU
没有本轮实机验收。一个短四路用例完成也不等于所有四路负载都不会饥饿。

## 复现命令

前置：Linux Python 3.12、匹配的 vLLM 0.28.0/torch，已安装测试依赖；CUDA 测试需
空闲 GPU，native E2E 需完整 MiniCPM-o 4.5 权重及 `assets/HT_ref_audio.wav`。
本地 Mac 仅做源码检查；本轮 CPU 逻辑回归也在同一个 H200 0.28 环境运行。

从 repo 根目录运行；测试源码快照复用同一 Python 环境，不是创建多个 vLLM 环境：

CI 的 MiniCPM duplex ready/merge/nightly/weekly job 已明确设置 V1 runner；
ready/merge 依赖列表补入 correlated RPC router。修改 CI 配置不代表 H100 已实跑。

```bash
# Local L1 / CI-like L1
tools/run_fullduplex_028.sh -m pytest -q \
  tests/engine/test_correlated_rpc_client.py \
  tests/worker/test_native_duplex_input_safety.py \
  tests/entrypoints/duplex/test_websocket_actor.py \
  tests/dfx/perf/tests/test_native_duplex_metrics.py \
  -m 'core_model and cpu' --run-level core_model

# CUDA sampler/graph 专项，既有 ready Model Executor CUDA sweep 收集
tools/run_fullduplex_028.sh -m pytest -q \
  tests/model_executor/models/minicpmo_4_5/test_native_sampling_cuda.py \
  tests/model_executor/models/minicpmo_4_5/test_codec_penalty_cuda.py \
  tests/model_executor/models/minicpmo_4_5/test_cuda_graph_wrapper.py \
  -m 'core_model and cuda' --run-level core_model

# Native L4 Perf：先选择空闲 CUDA_VISIBLE_DEVICES，设置 MODEL_PREFIX
export BENCHMARK_DIR=/path/to/fresh/native-results
tools/run_fullduplex_028.sh -m pytest -sv \
  tests/dfx/perf/scripts/run_native_duplex_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_minicpmo_4_5_native_duplex.json

# 独立的真实故障回归，不与性能测量并行
tools/run_fullduplex_028.sh -m pytest -sv \
  tests/dfx/reliability/test_reliability_minicpmo_4_5_duplex.py \
  -k 'preemption or failed_input or lost_reply or pending_append' \
  -m 'slow and H100 and cards_1' --run-level full_model
```
