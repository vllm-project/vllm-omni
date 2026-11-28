# 与 MixtureOfHorizons 对齐的计时与评测字段

## 参照实现与版本

参照 `/home/dong_jing/code/replay/MixtureOfHorizons/src/openpi/policies/policy.py`
的 `Policy.infer()`：输入 transforms、组 batch、Tensor 转换和输入搬入设备后，
紧贴 `_sample_actions(...)` 调用使用 `time.monotonic()` 计时；采样结果返回后结束，
然后才执行输出 `.detach().cpu()`、NumPy 转换与 output transforms。

```python
# 输入 transforms / Tensor / H2D 已完成
start_time = time.monotonic()
action_rets = self._sample_actions(...)
outputs = {"state": inputs["state"], "actions": action_rets["actions"]}
model_time = time.monotonic() - start_time
# 输出 D2H / NumPy / output transforms 在计时之后
outputs["policy_timing"] = {"infer_ms": model_time * 1000}
```

本实现的计时契约为 `moh_sample_actions_v1`，报告 `schema_version=3`。
**一致的是采样调用的测量边界、时钟与同步方式；Cosmos3 与 MoH 的模型结构、
实际采样计算量、动作表示和 horizon 并不因此相同。** 原 MoH 仓库没有直接
汇总 throughput 的公式；本文的模型吞吐量是用其采样耗时作为分母的明确扩展。

## worker 的计时位置

实现位于 `vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`。
普通/原生 LIBERO 动作路径在 prompt tokenize、外部输入处理及输入搬入设备之后，
从 `_prepare_action_latents` 前开始计时，经过 VAE 条件编码、模型 prefix/UND、
全部去噪更新，返回采样张量后结束。RoboLab 策略路径采用相同边界。

| 工作 | 是否进入 `policy_timing.infer_ms` |
| --- | --- |
| 环境观测、图像旋转/resize、PNG/base64、HTTP 上传与解码 | 否 |
| prompt 格式化/tokenize、外部 Tensor 输入准备、输入 H2D | 否 |
| action/video 噪声生成、模型内部 latent 准备 | 是 |
| VAE 条件编码、语言/视觉 prefix、KV、各轮模型前向与采样更新 | 是 |
| 采样内部自然发生的 CPU 判断、设备等待 | 是 |
| 最终动作 D2H、NumPy、物理量反归一化 | 否 |
| 视频输出解码、响应 JSON、文件持久化、job 轮询/删除 | 否 |
| 排队、锁等待、远端通信、环境执行、渲染和保存回放 | 否 |

起止处均不调用 `cuda.synchronize()`、平台 `synchronize()` 或 CUDA Event。
因此它与 MoH 一样是主机侧调用 wall time，**不是严格的 GPU kernel 完成时间**；
异步设备计算可能在计时结束后继续执行。不要为“补齐 GPU 时间”添加同步。
MoH 自己在 `sample_actions` 内部执行的 gate weights `.cpu()` 也会被计入，
这里同样不扣除采样内部的自然等待。

Pipeline Profiler 会在函数前后显式同步，动作采样计时因此拒绝开启
`enable_diffusion_pipeline_profiler`。正式测试也应关闭外部 profiler、视频输出与 dump，
并在两组测试中保持硬件、负载、权重、seed、采样步数和动作执行 horizon 一致。

流式回调会在采样内部执行 CPU 拷贝与 HTTP 投递，原始 `policy_infer_ms` 会包含
这些调用；不通过减去回调时间来伪造连续采样 wall time。流式记录保留作诊断，
**不进入正式 MoH 模型吞吐量**。`fair_eval.py` 对 HAS 与 constant Euler 都使用
非流式完整 chunk；流式功能仍可单独运行 `closed_loop_eval.py --streaming`。

## 服务响应与逐请求字段（时间单位均为 ms）

worker 的动作元数据携带 `policy_timing` 与 `timing_contract`，经过视频 job API、
transport、策略 frontend 原样传递。不能在 frontend 用远端 round trip 重新计时
并覆盖模型结果。批量请求当前逐条执行 job，`policy_infer_ms` 为各条采样时间之和；
它不是“平均每个 chunk 的耗时”。有任一条缺失时，整批该字段为 `null`。

| 响应 / JSONL 字段 | 定义 |
| --- | --- |
| `timing_contract` | `moh_sample_actions_v1`；没有此版本标记的旧值不能当模型时间 |
| `policy_timing.infer_ms` → `policy_infer_ms` | worker 采样主机 wall time；正式模型延迟 |
| `server_timing.infer_ms` → `server_infer_ms` | frontend 策略入口至完整物理动作生成，含预处理、锁等待、远端执行、反归一化；不含响应 JSON 序列化 |
| `timing_ms.full_chunk` → `server_full_chunk_ms` | 与 `server_infer_ms` 相同 |
| `timing_ms.backend_roundtrip` → `server_backend_roundtrip_ms` | 获得策略锁后至远端 job 与回调处理完成；旧版 `policy_infer_ms` 的范围移至这里 |
| `timing_ms.preprocess` → `server_preprocess_ms` | frontend 图像解码和策略输入准备 |
| `timing_ms.queue_wait` → `server_queue_wait_ms` | 输入准备后等待策略锁 |
| `timing_ms.denormalization` → `server_denormalization_ms` | 最终整批动作反归一化；流式 partial 的反归一化在回调内部 |
| `timing_ms.remote_roundtrip` | frontend 输入准备结束至最终动作完成，含锁等待、backend roundtrip 和最终反归一化 |
| `timing_ms.ttfa` → `server_ttfa_ms` | frontend 入口至首个物理动作可交付；非流式等于 `full_chunk` |
| `timing_ms.partial_events` / `partial_events` | frontend / 客户端观察到的 partial 事件数，各自计数 |
| `performance.worker_metrics[].policy_infer_ms` | 每个 job 的 worker 采样时间 |
| `performance.worker_metrics[].worker_inference_ms` | 视频 job generation/output persistence 耗时；与模型采样边界不同 |
| `performance.worker_metrics[].worker_job_latency_ms` | transport 开始提交 job 至轮询收到结果，含 HTTP、队列、轮询间隔；不含最后的删除请求 |
| JSONL `worker_inference_ms` | 一次请求内全部 job 的 `worker_inference_ms` 之和；有缺失则 `null` |
| `worker_metrics[].stage_durations` / `peak_memory_mb` | 原有阶段 profiler 数据 / 显存峰值，诊断用；不作为采样时间的替代值 |
| `client_preprocess_ms` | HTTP 提交前客户端图像准备、PNG/base64 等耗时 |
| `http_total_ms` | 客户端 HTTP 提交至完整调用结束 |
| `client_ttfa_ms` | HTTP 提交至首动作；非流式等于 `http_total_ms` |
| `client_time_to_first_action_ms` | `client_preprocess_ms + client_ttfa_ms` |
| `client_total_ms` / `client_time_to_last_action_ms` / `client_observation_to_completion_ms` | `client_preprocess_ms + http_total_ms`；末动作字段测到调用结束 |
| `client_infer_ms` | 非流式为 `client_total_ms`，流式为 `null`；不能用于模型延迟 |
| `client_last_ready_action_ms` | 最后一个 partial 动作实际到达时刻，相对 HTTP 提交；非流式为响应耗时 |
| `client_action_timings[]` / `server_action_timings[]` | 各动作的 `sample_index`、`action_index`、`ready_ms` 及可用时的 `step`；分别相对客户端 HTTP 提交 / frontend 入口，不跨进程比较绝对时钟 |
| `started_at_s` / `completed_at_s` | 客户端本次评测单调时钟相对起点，后者等于前者加 `http_total_ms/1000` |
| `request` / `endpoint` / `batch_size` / `streaming` / `warmup` | 完成顺序编号、接口、请求环境数、是否流式、是否属于被排除的计时预热 |
| `generated_chunks` / `generated_actions` | 完整返回的 chunk / 动作位置数；一个 16×10D chunk 是 16 actions，不是 160 |
| `worker_metrics[].denoise_steps` / `requested_denoise_steps` / `early_stop_actions` | 实际采样迭代数 / 原始网格步数 / 请求动作前缀长度；不是耗时 |

`server_infer_ms` 保留的是分离式服务的端到端诊断，**不声明与 MoH WebSocket
的 `server_timing.infer_ms` 完全相同**。统一模型评测使用 worker 采样字段。
`worker_inference_ms`、HTTP、TTFA 与模型时间之间不能任意相减推导“纯 GPU 时间”。

## 报告与吞吐量公式

默认排除前两个**完成的请求**（`--timing_warmup_requests 2`），不是额外执行两次
模型，也不是每个 episode 排除两次。`all_requests` 保留所有记录，`steady_state`
排除计时预热。环境起始的 10 个 dummy 动作是另一种稳定过程，不影响这个标记。

### 主指标：`throughput.steady_state_inference`

对正式非流式请求集，令 `T = sum(policy_infer_ms) / 1000`，则：

| 字段 | 含义 / 公式 |
| --- | --- |
| `timing_contract` | 同上，标记模型计时边界 |
| `request_count` | 排除预热后的请求数 |
| `timed_requests` | 带有效、有限、非负采样时间且契约匹配的非流式请求数 |
| `missing_timing_requests` | 非流式请求中缺失、契约不匹配或无效计时的数量 |
| `excluded_streaming_requests` | 该请求集的流式请求数量 |
| `timed_chunks` / `timed_actions` | 有效计时请求返回的 chunk / 动作位置数 |
| `sum_model_time_s` | `T`，只累计采样时间，不合并 HTTP 时间区间 |
| `complete` | 请求集非空、全部请求有效且非流式，并且 `T > 0` |
| `requests_per_s` | 请求数 / `T` |
| `chunks_per_s` | chunk 数 / `T`；单环境一次请求对应一个 chunk |
| `actions_per_s` | 生成动作位置数 / `T`；固定 horizon H 时等于 `H * chunks_per_s` |

仅当 `complete=true` 才输出三种速率，否则为 `null`/`n/a`，不把部分覆盖的数据
当作完整评测结果。不要平均每次请求的 `1000/infer_ms` 来代替“总数 / 总时间”。
若要报告实际执行动作的模型摊销速率，应另行记录每个 chunk 的实际执行数；本字段
采用**生成数**，不会将 episode 结束时尚未执行的后缀误称为已执行动作。

### 辅助吞吐量

| 报告分组 | 分母与范围 |
| --- | --- |
| `steady_state_http.http_active_s` | 有效 HTTP 请求区间的并集，避免并发重叠重复累计 |
| `steady_state_http.active_{requests,chunks,actions}_per_s` | 对应生成计数 / `http_active_s`；包含通信和排队 |
| `steady_state_http.first_to_last_request_s` | 首个请求起点至最后完成请求的跨度，含环境执行间隔 |
| `steady_state_http.sustained_{requests,chunks,actions}_per_s` | 对应计数 / 首尾跨度 |
| `steady_state_worker_jobs` | `sum_job_wall_s` 为已计时 job 的 wall time 总和；`chunks_per_s`、`actions_per_s` 以此为分母，仅诊断 |
| `evaluation_wall` | 整次任务评测耗时 `elapsed_s`，含环境初始化、推理、仿真和清理，不含服务启动；保留全部完成请求和计时预热 |
| `evaluation_wall.generated_{chunks,actions}_per_s` | 完整生成数 / 评测耗时 |
| `evaluation_wall.executed_actions_per_s` | 实际执行的模型动作数 / 评测耗时，排除 dummy 和未执行后缀 |
| `evaluation_wall.environment_steps_per_s` | 所有环境动作数 / 评测耗时，包含 dummy |

`request_timing_summary.throughput` 与主指标同义，`http_throughput` 是 HTTP 诊断。
`by_endpoint_and_batch_size` 分组分别提供这两种统计。延迟字段提供
`count/mean/std/median/min/max/p25/p50/p75/p90/p95/p99`；单位 ms，`std` 为总体标准差。
无有效样本时为 `null`。成功率仍为 LIBERO 成功 episode / 有效 episode；运行错误计
失败，Skipped 不进入分母；报告按总体及 task 分组。

## 使用与旧数据

1. 重启 worker 和策略 frontend，确认代码均来自本 checkout。
2. `/info.timing_contract` 必须为 `moh_sample_actions_v1`，一次 `/predict` 的响应
   必须有非空 `policy_timing.infer_ms`。只有 frontend 新、worker 旧时仍会是 `null`。
3. 运行 `examples/cosmos3/closed_loop_eval.py` 时不传 `--streaming`；保持相同预热、
   batch size、权重、seed、采样/执行 horizon 与硬件设置。
4. 对比 `metrics.json` 的 `policy_infer_ms` 延迟分布和
   `throughput.steady_state_inference`；保留原始 JSONL、配置与成功率。
5. `fair_eval.py` 会检查两端 `/info` 的契约，并在生成 `comparison.json` 前检查
   完整采样计时覆盖。外部 Framework 服务也需要在真正的采样边界实现此契约，
   不能仅给旧计时添加同名标签。本次修改仅涉及本 vLLM-Omni checkout。

schema v2 的 `policy_infer_ms` 是远端往返，`steady_state_inference` 是 HTTP active
吞吐量；schema v3 分别将这些诊断移至 `server_backend_roundtrip_ms` 和
`steady_state_http`。旧数据缺少 worker 原始采样时间，**无法离线精确换算**，需重跑。
`server_infer_ms`、`worker_inference_ms` 与闭环吞吐量的原有含义保留。

## 验证

`test_moh_sampler_timer_excludes_transforms_h2d_decode_and_output_copy` 用可控时钟
分别验证 HAS/constant Euler：人为增加 tokenize、H2D、采样、decode、CPU copy 的
耗时，只有采样部分进入结果；额外同步会直接导致测试失败。HTTP 测试覆盖从 job
元数据到响应的透传、批量求和与旧 worker 缺字段；统计测试覆盖预热、无效时间、
零分母、流式排除、不同吞吐量分母及 JSON/CSV/TXT 导出。

安装项目 dev 测试依赖后，在本 checkout 执行（HTTP 测试需要本机临时端口）：

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -o addopts='' -q \
  examples/cosmos3/tests/test_eval_metrics.py \
  examples/cosmos3/tests/test_fair_eval.py \
  examples/cosmos3/tests/test_policy_http.py \
  examples/cosmos3/tests/test_early_stop.py \
  examples/cosmos3/tests/test_has_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py
```

以上 CPU 回归验证计时边界和协议，不代替真实 checkpoint 的 GPU 吞吐量评测。
