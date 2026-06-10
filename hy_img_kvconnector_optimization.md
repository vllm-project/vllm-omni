# HunyuanImage-3 KV Connector 接收路径优化（Phase 1 / 2 / 3）

> 分支：`hy-img-kvconnector-opt`
> 涉及：`vllm_omni/distributed/omni_connectors/kv_transfer_manager.py`、
> `vllm_omni/diffusion/worker/diffusion_model_runner.py`、（Phase 3）`engine/orchestrator.py`
> 模型背景：HunyuanImage-3.0 是 **AR 与扩散共享同一 transformer** 的统一模型。
> stage-0（AR thinker，TP=2）prefill 产出的 KV 通过 connector 传给 stage-1（DiT，TP=2），
> 在 DiT 的 **self-attention 中作为共享前缀（prefix KV reuse）复用**——不是经典 cross-attention。

---

## 1. 问题与现状

stage-1 每个请求 forward 前要同步接收一份 AR KV。实测接收流水线分解：

| 组件 | 耗时 | 说明 |
|---|---|---|
| `connector.get()` | 102ms | ZMQ 握手 + RDMA 传输 + GPU sync（TCP 回退；配好 RDMA 可降到 ~20ms）|
| CPU-side clone | 40ms | `tensor.clone()`（Phase 1 消除）|
| `aten::copy_`（H2D） | 51ms | CPU→GPU，默认流同步，阻塞主线程 |
| **合计** | **~193ms** | 单图 forward ~500ms，接收占 ~28% |

### 调用链路（请求模式，非 stepwise）

```
DiffusionModelRunner.execute_model
  → kv_transfer_manager.receive_multi_kv_cache_distributed
     → receive_multi_kv_cache → receive_kv_cache → receive_kv_cache_for_request
        → connector.get()           （低层传输，ZMQ req/resp 同步阻塞）
        → from_bytes 反序列化
        → detach（H2D 上 GPU）
  → pipeline.forward
```

### 三条接收分支（`receive_multi_kv_cache_distributed`）

| 场景 | 分支 | 谁接收 | target | 是否本系列优化对象 |
|---|---|---|---|---|
| 单卡 / 纯 TP（**hunyuan**）| A：每 rank 独立接收 | 每 rank | **GPU** | ✅ |
| TP + CFG/SP | B：owner 在 CPU 接收 + 集合广播 | owner | CPU→`_apply_request_kv_payload` | ❌ 同步 |
| 传统 world>1 | C：rank0 在 CPU 接收 + 广播 | rank0 | CPU→`_apply` | ❌ 同步 |

**为什么 TP 不广播、CFG/SP 要广播**：TP 把 KV 沿 attention head 切成**互不重叠**的分片，每 rank 用 rank-aware key 各取各的；CFG（复制整模型跑 cond/uncond 分支）和 SP（切序列维）**不切 KV**，多个 rank 要的是同一份（或按分支拆的）数据，只能一个 owner 取一次再集合广播。集合通信不能在后台线程发起 → 这是 Phase 3 只覆盖 Branch A 的根因。

---

## 2. 关键内存事实（贯穿三个 Phase）

1. **CPU pool 是 pinned 的**（`mooncake_transfer_engine_connector.py` `pool = torch.empty(...).pin_memory()`），因为要给 RDMA 用；GPU pool 是 device tensor。
2. `connector.get()` 返回时，**数据已 RDMA 写入本地 pinned pool**；返回的 `ManagedBuffer` 只是指向那块的**零拷贝句柄**。
3. `from_bytes` 用 `torch.frombuffer` 产出**零拷贝视图**（仍指向 pool）；`from_bytes_device`（GPU pool）内部已 `.clone()`。
4. `ManagedBuffer.release()` 把内存还回 allocator free list，**下一次 `get()` 会复用并覆写**。
5. **pinned vs pageable**：`to(device, non_blocking=True)` 只有源是 **pinned** 时才真异步；pageable 源 CUDA 会先**同步**拷到 pinned 暂存区再 DMA → `non_blocking` 静默失效。这决定了哪条路能异步。

---

## 3. Phase 1 —— 消除冗余 CPU clone（已实现，commit `11e1d2e9`）

### 问题
旧 CPU-pool 路径：`from_bytes`（零拷贝视图）→ `_clone_received_payload_tensors`（**40ms**，复制成 pageable）→ `release()` → 之后再 `.to(GPU)`（**51ms**）。**搬了两趟**：`pool → pageable CPU → GPU`。clone 是为了能安全 release（否则 release 后视图悬空 = use-after-free）。

### 改动
GPU 目标路径**删除 clone**，改为**延迟 release**：保留 pinned pool buffer 存活，直接 `.to(GPU)`（这步本就要做、本就把数据拷出 pool），拷完再 release。`pool → GPU` 一趟，省掉 40ms。

- `pending_release_buffers`：暂存待释放的 pool buffer，detach 完成后统一 `release()`，`finally` 兜底（超时/异常也不泄漏）。
- CPU 目标路径（Branch B/C 的 owner 接收）仍 `clone()`——它没有 `.to()` 帮忙搬出 pool，且本就只一次拷贝、无冗余。
- GPU pool 路径（`from_bytes_device` 已内部 clone）行为不变，立即 release。

### 收益 / 风险
- GPU 目标主路径 **省 ~40ms**，零行为变化、零网络依赖。
- 删除了不再使用的 `_clone_received_payload_tensors` 及其孤儿测试。

---

## 4. Phase 2 —— H2D 拷贝异步化（已实现，commit `96f32691`）

### 问题
Phase 1 后 detach 仍是默认流上的**同步** `.to()`，阻塞主线程 ~51ms。因源已是 pinned pool，具备真异步前提。

### 设计：独立 stream + event + 延迟释放

- **专用 copy stream**：`tensor.to(target_device, non_blocking=True)` 在独立流上异步入队，与「接收→forward 之间的 CPU setup（generator / cache refresh）」并行。
- **CUDA event 的两种用法**：
  - `wait_kv_copy()`（forward 前）：`current_stream().wait_stream(copy_stream)` —— **GPU 侧排序**，让 forward 等拷贝，**不阻塞 CPU**；
  - `event.synchronize()`（下次接收的 drain 里）：**CPU 侧确认**拷贝完成后才敢 release pinned pool（否则 DMA 还在读就被复用 → 损坏）。
- **延迟释放（零阻塞）**：detach 时不 release，把 buffer 转入 `_kv_inflight_buffers`、记 event；release 推迟到**下个请求接收开始**的 `_drain_inflight_buffers()`——那时上个 forward(~500ms) 早跑完、拷贝必结束，`event.synchronize()` 瞬时返回。

### CPU 入队顺序 vs GPU 执行顺序（要点）
`with torch.cuda.stream(copy_stream)` 只把「不点名流」的 `.to()` 入队到 copy_stream；`copy_stream.record_event()` 点名了流，放 with 外也准确排在拷贝**之后**（FIFO）。CPU 把「入队拷贝 → 记事件 → 挪引用」瞬间跑完就往下走，GPU 才慢慢拷、最后点亮 event。`extend/clear` 只动 Python 引用、不动数据，安全。

### 安全边界（opt-in，避免 race）
异步路径靠调用方在 forward 前调 `wait_kv_copy`。`receive_kv_cache_for_request` 还被 `OmniConnectorModelRunnerMixin`（AR 接收端）调用，**它不 wait**。故：

- **per-manager opt-in**：`_kv_async_copy` 默认 False；只有接好 `wait_kv_copy` 的 **diffusion runner** 调 `enable_async_kv_copy()` 开启。其它接收端保持同步。
- `gpu_async` 门槛：`_kv_async_copy and copy_stream and target_device.type != "cpu" and needs_buffer_detach`（即仅 pinned-pool 源 + GPU 目标）。
- 全局熔断：`VLLM_OMNI_KV_ASYNC_COPY=0`。
- 异常路径：若异步拷贝半途异常，`finally` 里先 `copy_stream.synchronize()` 再 release，避免 use-after-free。

### 作用域与 ROI
- 只命中 Branch A（hunyuan 主路径，源 pinned）；CFG/SP（源 pageable、且要广播）走同步。
- **单请求**只隐藏「receive→forward 间 CPU setup」那部分 51ms（接收时 GPU 空闲，无 GPU 计算可重叠）。
- 真正大头靠 Phase 3：这套 **stream/event/drain 基建即为 Phase 3 复用**。

---

## 5. Phase 3 —— 后台预取下一请求 KV，与上一请求 forward 重叠（详细设计 / 已实现一版）

> 配置开关：`enable_kv_async_prefetch`（默认 False，opt-in）。CLI `--enable-kv-async-prefetch`。
> 默认关 → 全部新增路径是 no-op，零行为变化。

### 5.1 目标与作用域
- **目标**：请求 N+1 的 KV 接收（~102ms 的 get，配 RDMA 后 ~20ms）在请求 N 的 forward(~500ms) 期间于**后台 CPU 线程**完成,使深队列里 2nd+ 请求的接收等待 ≈ 0。
- **仅 Branch A**（`cfg_parallel_size<=1 且 sequence_parallel_size<=1`）：CFG/SP 走集合通信、不能后台发起 → 退回同步。由 **runner** 按 `od_config.parallel_config` 在 `__init__` 判定后 gate。
- **仅 request 模式**：KV transfer 只存在于 request 模式;该模式 `max_num_running_reqs` 被强制为 1（`diffusion_engine.py:166`）→ forward 严格串行 → 「下一个请求 = `_waiting[0]`」唯一确定、同时最多一个预取在飞。

### 5.2 关键事实：触发与执行跨进程
- **「看得到下一个请求」在 engine 进程**：`_busy_loop`（`diffusion_engine.py:471`）持有 `scheduler._waiting`；`DiffusionRequestState.req` 带 `kv_sender_info`。
- **「有 connector、能拉 KV」在 worker 进程**：`DiffusionModelRunner.kv_transfer_manager`。
- 两者经 `collective_rpc("execute_model", args=(req, od_config))`（`multiproc_executor.py:304`，`exec_all_ranks=True`，每个 TP rank 各跑一份）连接。
- ⇒ 预取的**触发信息**（下一请求的 `request_id` + `kv_sender_info`，打包成轻量 dict `prefetch_stub`）必须**随 execute_model RPC 一起下发**给 worker，由 worker 侧 manager 真正发起;**不动 orchestrator**。

### 5.3 数据流与时间线（2nd+ 请求）
```
engine (_busy_loop)                     worker (runner + manager)
schedule() → req_N
peek _waiting[0]=req_{N+1}
  prefetch_stub = {request_id, kv_sender_info}
  collective_rpc("execute_model",
      args=(req_N, od_config, prefetch_stub)) ──▶ execute_model(req_N, prefetch_stub):
                                                    1) data = get_loaded_kv(rid_N)      # forward N-1 时已后台拉好
                                                       └ miss → 同步 receive(req_N)      # 首个请求/未命中 fallback
                                                    2) apply → req_N (H2D, Phase2 流)
                                                    3) start_load_kv(prefetch_stub)     # 后台开拉 N+1（非阻塞返回）
                                                    4) wait_kv_copy(); forward N  ← 后台在此拉 N+1
```
`start_load_kv` 只把任务塞进线程池立即返回;真正拉取发生在第 4 步 forward N 期间（后台线程，get/IO 释放 GIL）。

### 5.4 线程切分：后台只 CPU/IO，主线程独占 CUDA
| 段 | 内容 | 线程 | 重叠 |
|---|---|---|---|
| ① get + 反序列化 | ZMQ/RDMA + `from_bytes`（纯 CPU/IO，释放 GIL）| **后台** | ✅ 与 forward N |
| ② pin + release pool | 视图 → 自有 pinned CPU 张量；立即 `release()` pool buffer | **后台** | ✅ |
| ③ H2D | pinned → GPU（复用 Phase 2 异步流 + event）| **主线程** | 消费 N+1 时 |

后台返回「自有 pinned CPU data」：不跨线程持 pool、不跨线程碰 stream/event、源仍 pinned 保证主线程 H2D 真异步。

### 5.5 配置项（沿用 Phase 2 的 opt-in 落地法）
新增 `enable_kv_async_prefetch: bool = False`：`data.py` 字段 + `serve.py --enable-kv-async-prefetch`(store_true) + `arg_utils.OrchestratorArgs` 字段 + `async_omni_engine.stage_engine_args` 透传 + `from_od_config` 构造注入到 manager 的不可变 `_async_prefetch`。H2D 是否异步仍由独立的 `enable_kv_async_copy` 决定（`apply_prefetched_kv` 两条路都支持，async 关时退化为同步 H2D，仍享受「藏 get」主收益）。

### 5.6 Manager 改动（核心）
```python
# __init__(config, *, async_kv_copy=False, async_prefetch=False)
self._async_prefetch = async_prefetch
self._load_executor = None            # 懒建 ThreadPoolExecutor(max_workers=1)
self._load_futures: dict[str, Any] = {}

# receive_kv_cache_for_request(rid, target_device=None, *, sender_info=None, pin=False)
#   - pin=True（后台）：跳过 _drain_inflight_buffers()（drain 含 event.synchronize = CUDA）
#                       不取 copy_stream；target_device=None 分支用 _to_owned_pinned() 代替 clone()
#   - sender_info!=None：从参数解析 base host/port 构 rank_metadata（即便 TP=1 也构），
#                        彻底不读/写 self._sender_base_*（消除后台 vs 主线程的端点竞争）

def start_load_kv(self, prefetch_stub):                 # 主线程调，立即返回
    if not (self._async_prefetch and self.config.need_recv_cache and prefetch_stub): return
    rid, sender_info = prefetch_stub["request_id"], prefetch_stub.get("kv_sender_info")
    if not rid or rid in self._load_futures: return
    self._load_futures[rid] = self._get_load_executor().submit(self._prefetch_payload, rid, sender_info)

def _prefetch_payload(self, rid, sender_info):          # 后台线程：纯 CPU/IO
    return self.receive_kv_cache_for_request(rid, target_device=None, sender_info=sender_info, pin=True)

def get_loaded_kv(self, rid, timeout=None):             # 命中→result；miss/超时/异常→(None,0) 走同步
    fut = self._load_futures.pop(rid, None)
    if fut is None: return None, 0
    try: return fut.result(timeout=timeout or self.config.recv_timeout)
    except Exception: logger.exception(...); return None, 0

def apply_prefetched_kv(self, req, data, target_device):  # 主线程 H2D，复用 Phase 2 流/event
    self._drain_inflight_buffers()                         # 预取消费路径也要 drain（原设计漏点）
    # gpu_async：在 copy_stream 上 .to(non_blocking)，record_event，把 pinned 源存入 _kv_inflight_buffers 保活
    # 否则：同步 .to()
    self.apply_kv_cache_to_request(req, data)

def abort_load_kv(self, rid): ...                        # cancel 未启动的；已启动的让其跑完
```
`_drain_inflight_buffers` 泛化：对没有 `.release()` 的元素（pinned 源张量列表）只 drop 引用。`_to_owned_pinned(t) = t.contiguous().pin_memory()`（见 5.8 R1）。

### 5.7 调度 / 执行链改动（全部 default-off 守卫）
- `DiffusionSchedulerOutput` 加 `prefetch_stub: dict | None = None`。
- `_BaseScheduler.initialize` 读 `self._prefetch_enabled = od_config.enable_kv_async_prefetch`；`schedule()` 末尾若启用且 `_waiting` 非空，从 `_request_states[_waiting[0]].req` 取 `{request_id, kv_sender_info}`。
- `multiproc_executor.execute_request`：`args=(req, od_config, scheduler_output.prefetch_stub)`。
- `DiffusionWorker.execute_model(req, od_config, prefetch_stub=None)` → `runner.execute_model(req, prefetch_stub=prefetch_stub)`。
- `DiffusionModelRunner.__init__`：`self._kv_prefetch_enabled = enable_kv_async_prefetch and Branch_A and need_recv_cache`。
  `execute_model`：先 `get_loaded_kv(rid)` 命中走 `apply_prefetched_kv`，否则同步 `receive_multi_kv_cache_distributed`；随后 `start_load_kv(prefetch_stub)` 触发 N+1。

### 5.8 正确性 / 风险
1. **集合通信**：仅 Branch A 触发 `start_load_kv`；CFG/SP 永不预取。
2. **无跨线程 CUDA**：后台只 CPU/IO（drain 被 `pin` 跳过；不取 copy_stream）；H2D 全主线程,`_kv_copy_event/_kv_inflight_buffers` 单线程读写。
3. **无共享端点（核心修正）**：sender_info 走**参数**进 receive，不再读/写实例 `_sender_base_*` → 后台预取 N+1 与主线程 fallback N 即便来自不同 AR replica 也无竞争。`update_sender_info` 仍只由同步路径用。
4. **window=1**：request 模式串行 + `max_workers=1` → 同时最多一个预取在飞，单 event/单 inflight 槽成立。
5. **一次性消费**：`_load_futures` 去重 + 命中即用，主线程不重复 `get`（ManagedBuffer 一次性）。
6. **连接器并发**：串行保证后台 get 与主线程 get 永不同时发生，connector 无需线程安全。
7. **fallback 闭环**：miss/超时/异常 → 同步接收;abort → `abort_load_kv` + 孤儿在 drain/finish 清理。
- **R1（需实测）**：`pin_memory()` 是 page-locked host alloc，可能触发隐式 device sync 拖慢并发的 forward N。缓解：改用**可复用 pinned staging 缓冲 + host→host memcpy**（window=1 → 一块即可），`_to_owned_pinned` 是唯一替换点。
- **R2（需实测）**：`from_bytes` 逐层 Python 循环持 GIL,与 forward 的 host 端 kernel launch 抢锁;用「forward N wall-time 有/无并发预取的差值」验证重叠率。
- **R3（ROI）**：预取主要藏 **get**，**不藏 H2D**（H2D 仍在消费时入 copy stream、forward 等它）。配好 RDMA 后 get 102→20ms，Phase 3 边际收益大幅缩水 → **把「RDMA 是否可配」作为是否做 Phase 3 的前置判断**。

### 5.9 时间线
```
Req N:   get_loaded_kv(命中~0) → apply(H2D) → wait → [===== forward N (500ms) =====]
                                                       │ 期间后台预取 N+1（get+pin）
后台:                                                  [== get + 反序列化 + pin (~102ms) ==]✓
Req N+1: get_loaded_kv(就绪~0) → apply(H2D) → wait → [== forward N+1 ==]
```

### 5.10 孤儿、超时与「AR 还没产出 KV」分析

**「会不会预取 N+1 时 AR 还没把它的 KV 发出来 → 超时？」**——正常流程**不会**:
- AR→DiT 是 `async_chunk: false` 的非流式管线。一个请求**只有在 AR 阶段把它跑完、并 `put` 了 KV 之后**,orchestrator 才会 `submit_initial` 把它转给 DiT(`_forward_to_next_stage`)。
- 因此当某请求出现在 DiT 调度器 `_waiting` 里(= DiT 已收到它的 `add_request`)时,**它的 AR KV 必然已经 `put` 到 connector**。我们只预取 `_waiting[0]`,所以预取目标的 KV 总是已就绪;`put` 与 `add_request` 之间极小的时序窗口由 receive 的轮询吸收。
- 真正会「等不到」的只剩**异常路径**:该请求随后被 abort(AR 可能不再发 / KV 已过期),或 `prefetch_stub` 的 `kv_sender_info`/key 不对 → get 轮询拿不到 → 超时。

**这些异常路径如何兜底**:
1. **短预取超时(已实现)**:后台 get 用 `_prefetch_timeout = min(recv_timeout, 5s)`,而非 30s 的 `recv_timeout`。超时返回 `(None, 0)`。
2. **不双消费**:超时的 get **没有消费** connector 里的 payload(只在真拿到结果时才消费),所以主线程 fallback 的同步 `receive`(用完整 `recv_timeout`)仍能取到。
3. **主线程不被拖死**:`get_loaded_kv` 的 `fut.result()` 也按 `_prefetch_timeout` 上界等待(+1s slack 以观测后台自身的 `(None,0)` 而非 result 超时);miss → 同步 fallback。
4. **结果**:AR 晚到的请求**最坏退化成基线同步接收**,无额外延迟、无双消费;正常请求仍命中预取。

**孤儿(预取了但永不被消费的请求,如被取消的 N+1)如何处理**:
- **sweep-on-insert(主兜底)**:`start_load_kv` 起新预取前,丢弃 `_load_futures` 里所有 rid≠当前 的旧条目(串行模式同时只该有一条在飞)。把泄漏从「无界累积」降为「最多 1 条瞬时」。
- **`_discard_future` 释放 pinned**:未启动→`cancel()`;已启动/完成→挂 done-callback 取出并丢弃结果,使自有 pinned CPU 张量被 GC 回收,不残留「未取的 future 结果」。
- **shutdown 清理(已接线)**:`manager.shutdown_prefetch()`(取消全部 future + `executor.shutdown(cancel_futures=True)` + drain inflight)挂在 `DiffusionWorker.shutdown()`。
- **`abort_load_kv(rid)`**:对外精确取消入口(走 `_discard_future`)。

**仍存的活性缺口(已知,未做)**:`max_workers=1` 下,一个孤儿的 get 若正卡到 `_prefetch_timeout`,会占住唯一 worker,使紧随的预取被推迟最多 ~5s(只影响一发请求的延迟,不影响正确性/不泄漏)。彻底解法是**跨进程精确 abort**:引擎 abort 已预取的 req 时 RPC 通知 worker `abort_load_kv(rid)`,在 get 启动前取消。改动面大,作为后续;短超时已把影响压到秒级。

---

## 6. 落地顺序与优先级

| 项 | 收益 | 风险 | 状态 |
|---|---|---|---|
| **RDMA 替代 TCP**（配置 `device_name` + RDMA 网口 IP，RoCE 加 `MC_GID_INDEX`）| 102→20ms，~82ms | 仅配置、零代码 | 待配置（需硬件）|
| **Phase 1**：删冗余 clone | ~40ms | 低 | ✅ `11e1d2e9` |
| **Phase 2**：H2D 异步 | 单请求隐藏部分 51ms；为 P3 铺路 | 中 | ✅ `96f32691` |
| **Phase 3**：后台预取 | 2nd+ 请求隐藏 get（~102ms，RDMA 后 ~20ms），深队列吞吐 +40~50% | 高（跨进程链路 + R1/R2 需实测）| 已实现一版（默认关，`enable_kv_async_prefetch`）|

> 注：Phase 1/2 纯软件、不依赖网络硬件；RDMA 配置是最大且最安全的单项收益，但需节点具备 RDMA 网卡。三者收益叠加。
