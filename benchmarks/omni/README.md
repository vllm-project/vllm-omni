# Omni Serving Benchmark (i2t / t2i / it2i)

针对**理解生成合一模型**（如 HunyuanImage-3.0 AR+DiT 统一部署）的混合任务调度评估工具。
在 `vllm serve ... --omni` 启动的统一服务端上，将 **i2t（理解）/ t2i（生成）/ it2i（编辑）** 三类任务按可配置配比混合发送，统计调度与性能指标，用于评估 DTPS 等混合调度优化的收益。

入口脚本：

- `benchmarks/omni/omni_benchmark_serving.py`

与 `benchmarks/diffusion/` 的关系：`diffusion/` 只覆盖纯生成任务（t2i/t2v/i2i）；本工具在其 backend 实现之上复用并扩展，新增三类任务的混合配比、it2i `bot_task` 随机化、真随机发送顺序与按任务/按 bot_task 的分桶统计。

---

## 1. 设计目标与覆盖的能力

| # | 能力 | 说明 |
|---|------|------|
| 1 | **任务配比** | `--num-i2t/--num-t2i/--num-it2i` 显式指定每类请求数量，例如 `14:2:4` 即 7:1:2 的近似配比。 |
| 2 | **it2i bot_task 配比** | `--it2i-bot-task-weights "recaption=2,think=1,think_recaption=1"` 按权重分配思考强度。默认 `--it2i-bot-task-sampling proportional` 用最大余数法做**精确比例分配**（10 任务 2:2:1 → 4/4/2，可复现）；`--it2i-bot-task-sampling random` 改为带权重独立抽样（只在期望上接近比例）；缺省权重等比覆盖三类。 |
| 3 | **真随机发送顺序** | 默认 `--shuffle` 对合并后的请求列表做 `random.shuffle`，三类任务真正交错发送，而不是按类型分块；`--no-shuffle` 则按类型分组发送。`--seed` 同时控制 shuffle 与 bot_task 采样，整个序列可复现。 |
| 4 | **支撑评估的统计输出** | 终端表格 + JSON 文件，包含 overall / per-task / it2i-by-bot_task 三层分桶，覆盖吞吐、延迟分位、per-stage durations、peak memory。 |
| 5 | **文本输出 TTFT/TPOT/ITL** | `--stream` 开启流式，按 vLLM `benchmark_serving` 的口径统计三类任务**文本输出**的首 token 延迟（TTFT）、每 token 延迟（TPOT）、token 间延迟（ITL）：i2t 的 AR 最终回答文本、it2i 的 AR "思考"(CoT) 文本。详见 §2.5。 |
| 6 | **发送顺序 = dry-run 顺序（可证）** | 发送前一次性预读+预编码 i2t/it2i 输入图（base64 / 原始 bytes 缓存到 request 上），发送期 sender 跳过文件 I/O 与 base64；再在每个 `create_task` 后 `await asyncio.sleep(0)` 让 task[i] 抢到 FIFO 信号量并 `post` 之后才创建 task[i+1]。两者合起来保证**客户端实际发送顺序严格等于 dry-run / 列表顺序**，不再被各任务 prep 耗时（t2i < it2i < i2t）抢跑。运行时打印 `Client dispatch order matches dry-run list order: True` 并把 `dispatch_order` 写入 JSON 供与服务端接收日志对照。详见 §2.6。 |
| 7 | **`--no-shuffle` 分组顺序** | `--no-shuffle-order "t2i,it2i,i2t"` 指定三类分组的拼接先后（须为 `i2t,t2i,it2i` 的全排列，默认 `i2t,t2i,it2i`）。仅在 `--no-shuffle` 下生效；`--shuffle` 时该参数被忽略（合并列表整体随机化）。 |
| 8 | **生图分辨率配比** | `--gen-resolution-weights "512x512=2,1024x1024=2,1280x720=1"` 给 t2i/it2i 两类生图任务按比例分配生成分辨率（仅限 `512x512 / 1024x1024 / 1280x720`）。默认 `--gen-resolution-sampling proportional` 最大余数法精确分配（10 请求 2:2:1 → 4/4/2）；`random` 带权重独立抽样。i2t 不受影响；custom 数据集行若显式带 `width/height` 仍优先。 |
| 9 | **输入随机化** | `--randomize-input` 让**输入侧**逐请求随机化，贴近真实请求分布多样性（由 `--seed` 控制，可复现）：① 三类任务的 prompt 从"短→长"池中各取一条，使 prompt 长度随机分布；② i2t/it2i 的输入图分辨率在 `512x512 / 1024x1024 / 1280x720` 间均匀抽取；③ i2t/it2i 的输入图内容逐请求生成不同的随机配色+形状图（不再共用一张纯色占位图）。仅对 `random` 数据集生效（`custom` 行始终优先）；开启时 `--input-image` 与 `--i2t/--t2i/--it2i-prompt` 被忽略。关闭时行为与旧版逐字节一致（不引入额外 RNG 抽取）。 |

此外复用了 `diffusion` 既有能力：请求数量、并发上限（`--max-concurrency`）、Poisson 请求速率（`--request-rate`）、warmup、分位（`--percentiles`）、统一落盘 `--output-dir`、`--disable-tqdm` 等。

---

## 2. 工作机制

### 2.1 任务类型 → 服务端字段的映射

三类任务都走 `POST /v1/chat/completions`，通过 `modalities` 字段区分（与服务端 `serving_chat.py` 中 `omni_task_type` 的解析逻辑对齐）；it2i 另可选 `POST /v1/images/edits`（`--it2i-endpoint images-edits`）。

| 任务 | `modalities` | 输入图 | 额外字段 | 服务端 bucket |
|------|--------------|--------|----------|---------------|
| i2t（理解） | `["text"]` | 有 | — | `ar_only`（AR 阶段即结束） |
| t2i（生成） | `["image"]` | 无 | — | `ar_downstream`（AR→DiT） |
| it2i（编辑） | `["image"]` | 有 | `bot_task` ∈ {recaption, think, think_recaption} | `ar_downstream`（AR→DiT） |

> i2t 与 t2i/it2i 落在 DTPS 的不同调度桶里，混合发送正是为了压测跨桶调度。

### 2.2 请求构造流水线（`omni_dataset.build_requests`）

1. 按数量分别构造 i2t / t2i / it2i 的 `OmniRequest` 列表（数据源见 §3）。
2. 用单个 `random.Random(seed)` 依次抽取：
   - **it2i bot_task**（最先抽，保证不开启分辨率/输入随机化特性时 bot_task+shuffle 序列与旧版逐字节一致）：默认 `proportional`（最大余数法，精确匹配权重，如 10 任务 2:2:1 → 4/4/2），或 `random`（`rng.choices` 带权重独立抽样，只在期望上匹配）。分配结果再用 `rng.shuffle` 打乱，使 bot_task 在 it2i 内部也是随机分布的（`--no-shuffle` 时不会聚块）。
   - **t2i / it2i 生成分辨率**（仅当 `--gen-resolution-weights` 给定）：各自独立按权重抽取（`proportional` 精确 / `random` 期望），允许取值限定 `512x512 / 1024x1024 / 1280x720`。抽取结果再 `rng.shuffle`。i2t 不参与；custom 行显式 `width/height` 仍优先（`gkw.setdefault`，只在行未指定时回填）。
   - **输入随机化**（仅当 `--randomize-input` 且 `dataset=random`）：在上述分配之后、构造每条请求时，按 i2t→t2i→it2i 的顺序逐请求从同一个 `rng` 抽取 prompt（短→长池）+（仅 i2t/it2i）输入图分辨率与内容。关掉时不做任何额外抽取，故旧版序列不变。
3. 按 `group_order`（`--no-shuffle-order`，默认 `i2t,t2i,it2i`）拼接三列表；缺漏的类型防御性追加在后，保证计数不丢。
4. `--shuffle` 时对合并列表 `random.shuffle`（同一个 `rng`，保证整体可复现）。

`OmniRequest` 是 `diffusion.RequestFuncInput` 的子类，仅新增 `task_type` / `bot_task` / `input_image_size` 三个客户端记账字段，**不发送到服务端**。Phase D 在其上新增 `image_data_urls` / `image_bytes` 两个缓存字段（见 §2.6），同样不发送到服务端。

### 2.3 发送与统计（`omni_benchmark_serving.benchmark`）

- `iter_requests`：Poisson 到达（`expovariate`），`--request-rate inf` 时一次性全部发出。
- 并发受 `asyncio.Semaphore(--max-concurrency)` 上限保护。
- **预编码 + 有序派发**（Phase D，见 §2.6）：发送前一次性把 i2t/it2i 输入图读出并缓存 base64 / 原始 bytes 到 request 上；发送期 sender 直接用缓存，跳过文件 I/O 与 base64，事件循环不再被 ~15ms/请求的同步 prep 阻塞。每个 `asyncio.create_task` 后 `await asyncio.sleep(0)`，让 task[i] 在 task[i+1] 创建前就抢到 FIFO 信号量并 `session.post`，从而**客户端实际发送顺序严格等于 dry-run / 列表顺序**。运行结束打印 `Client dispatch order matches dry-run list order: True`，并把 `dispatch_order`（实际进入 sender 的 idx 序列）写入 JSON `config.dispatch_order` 供与服务端接收日志对照。
- `--warmup-requests` 在测量前以 `--warmup-concurrency` 跑一批预热（不计入指标）。
- `--return-stage-metrics` 让服务端返回 `stage_durations`（ar_prefill / ar_decode / dit），统计时做 mean / p50 / p99 聚合。
- 指标分桶：
  - **overall**：全量吞吐与延迟。
  - **per_task**：i2t / t2i / it2i 各自的 qps、延迟分位、stage_durations、success_rate、peak_memory。
  - **it2i_by_bot_task**：按 `bot_task`（recaption / think / think_recaption）再分桶，比较不同思考强度的延迟。
  - per-task qps 以 `total_duration` 为分母，三者相加 = overall qps。
- `--output-dir DIR` 统一落盘：`DIR/result.json` 写指标，`DIR/inputs/` 落 i2t/it2i 的输入参考图，`DIR/outputs/` 落 t2i/it2i 的生成图（i2t 无生成图）。it2i 同时有 `inputs/` 与 `outputs/` 条目，可逐请求前后对比。

### 2.4 it2i 两种 endpoint 的差异

| 项 | `--it2i-endpoint chat`（默认，统一） | `--it2i-endpoint images-edits` |
|----|--------------------------------------|--------------------------------|
| 路径 | `/v1/chat/completions` | `/v1/images/edits`（multipart） |
| `bot_task` | 放在请求体 `extra_body.bot_task` | multipart 字段 `bot_task` |
| `stage_durations` | ✅ 随 chat 响应返回 | ⚠️ edits 响应体不携带 stage_durations，故 it2i 该桶 stage 统计为空 |

> 评估 per-stage 时序优化建议用默认 `chat`；images-edits 用于回归兼容性或与服务端 edits 通路对齐。

### 2.5 文本输出的 TTFT / TPOT / ITL（`--stream`）

参照 vLLM `benchmarks/benchmark_serving`（`serve.py`）对大语言模型的指标口径，本工具在 `--stream` 下用流式请求采集三类任务**文本输出**的 token 级延迟：

| 指标 | 定义 |
|------|------|
| **TTFT** | 从请求发出到第一个文本 token（delta）返回的耗时。 |
| **ITL** | 相邻文本 token 之间的到达间隔（每请求一个 list）。 |
| **TPOT** | 每请求 `sum(itl) / (output_tokens - 1)`，即排除首 token 后的平均每 token 耗时；`output_tokens <= 1` 时记 0。用 ITL 之和（= 末个文本 token 时刻 − TTFT）而非 `latency − ttft`，使 TPOT 只落在**文本流时间窗内**——对 i2t（AR 是终点阶段）与 vLLM 原口径一致；对 it2i 则**自动剔除文本流结束后的 DiT 阶段**，不被算到每 token 头上。 |
| **output_tokens** | 文本 token 数。chat i2t 取最终 `usage.completion_tokens`（缺失则回退为 delta 计数）；it2i edits 取 `ar_delta` chunk 数。 |

聚合时统一换算到毫秒，报告 mean / std / median / p50 / p95 / p99（分位由 `--percentiles` 控制）。**仅对流式采集到文本的请求计入**（`measured_requests`），非流式运行或服务端未流式文本的任务，对应桶 `count=0`、不参与统计（不会把 0 当成零延迟样本拉低均值）。

各任务的文本输出与采集路径：

| 任务 | "文本输出"指什么 | 采集方式 | 备注 |
|------|------------------|----------|------|
| **i2t** | AR 阶段的最终回答文本（AR 是终点阶段） | chat 流式，解析 `choices[0].delta.content` 文本 delta | ✅ 主场景，TTFT/TPOT/ITL 完整 |
| **it2i** | AR 阶段的 "思考/CoT" 文本（图像生成前先吐出的规划文本） | **`--it2i-endpoint images-edits`** 流式，解析 `ImageEditARDeltaChunk`（`type="ar_delta"`）的 `delta` | ✅ 推荐路径；edits 流会逐 token 吐 AR 文本，再吐一个 image chunk |
| **t2i** | AR "思考" 文本 | chat 流式仅吐 image chunk（服务端 `first_iteration_dict` 过滤了 AR 文本） | ⚠️ 文本 TTFT/TPOT 不可用（N/A）；如需 t2i AR 文本延迟，需服务端支持，本工具不修改服务端 |

> 因此 **it2i 文本延迟请配合 `--it2i-endpoint images-edits --stream`** 使用；i2t 文本延迟用默认 chat + `--stream` 即可。流式响应体会在客户端被重构成与非流式等价的 JSON，故既有的 `extract_chat_outputs` / `extract_edits_outputs` / 图片落盘 / `stage_durations` 统计**全部沿用**，`--stream` 只是在其上叠加 token 级延迟指标。

> 与 per-stage durations 的关系：`--return-stage-metrics` 给的是 AR/DiT **整阶段**耗时（ar_prefill / ar_decode / dit）；TTFT/TPOT/ITL 是 AR 文本**逐 token** 的延迟分布，二者互补——前者定位瓶颈阶段，后者刻画用户体感（首字等待、生成流畅度）。可同时开启 `--stream --return-stage-metrics`。

### 2.6 发送顺序：客户端 = dry-run 顺序，服务端 t2i→it2i→i2t 是调度器行为（Phase D）

**症状**：实测中 `--shuffle` 与 `--no-shuffle` 两种模式下，服务端接收/处理请求的顺序都呈现 `t2i → it2i → i2t` 的分组，与 dry-run 展示的请求序号对不上。

**根因结论（客户端无 bug）**：

- 客户端**确实按列表顺序发送**。旧实现里每个 i2t 请求发送前会同步执行 `_encode_image_as_data_url`（`open` + `read` + `base64.b64encode`，无 `await`），这段同步代码会阻塞事件循环 ~15ms，反而把发送**串行化成列表顺序**——也就是说，不存在"客户端按 prep 耗时抢跑导致重排"。
- 服务端看到的 `t2i → it2i → i2t` 是 **DTPS 调度器的派发顺序**：t2i/it2i 落在 `ar_downstream` 桶、i2t 落在 `ar_only` 桶，调度器优先派发 `ar_downstream`（且 i2t 有"安全窗口=未吐字"的抢占设计，会延后处理）。所以即便客户端交错发送，服务端处理日志也按桶归并。**本工具不修改服务端**，这个行为是调度策略本身，正是混合调度评估要测的对象。

**为何仍要修（保真 + 可证）**：旧的同步 prep 虽然凑巧保持了列表顺序，但阻塞事件循环 ~15ms/请求，会扭曲并发下的延迟测量；且"凑巧保持"不可证明。Phase D 改成显式保证：

1. **预编码**：发送前一次性把 i2t/it2i 输入图读出，base64（供 chat sender）与原始 bytes（供 edits sender）缓存到 `OmniRequest.image_data_urls` / `image_bytes`。发送期 sender 命中缓存，跳过文件 I/O 与 base64 → 事件循环零阻塞。
2. **有序派发**：每个 `asyncio.create_task(limited(...))` 后 `await asyncio.sleep(0)`，让刚创建的 task 在下一个 task 创建前就抢到 FIFO 信号量并 `session.post`。配合预编码（无 prep 延迟），发送顺序 = 列表顺序，即使 `--request-rate inf` + 并发也成立。
3. **审计字段 `dispatch_order`**：`limited` 在信号量获取后把 `idx` 追加到共享 `dispatch_order` 列表，运行结束写入 JSON `config.dispatch_order` 并打印 `Client dispatch order matches dry-run list order: True`。把它与服务端接收/处理日志对照，即可把"客户端发送序"与"服务端派发序"区分开，直接回应"对不上"的疑问。

> 对照试验（纯 asyncio 模拟，已通过）：①旧同步 prep → 列表顺序（无重排，但阻塞循环）；②若把 prep 改成异步却不加 `sleep(0)` → 按 prep 耗时重排（t2i 抢跑）；③预编码 + `sleep(0)` → 列表顺序且零阻塞。即②才是真正的"客户端发送顺序 bug"，而旧实现是①、新实现是③。

---

## 3. 数据集

### 3.1 `random`（默认，零外部依赖）

合成 prompt + 一张内置 512×512 占位图（i2t/it2i 输入），prompt 可用 `--i2t-prompt / --t2i-prompt / --it2i-prompt` 覆盖，生成分辨率/步数用 `--width/--height/--num-inference-steps`。足以驱动调度器、收集延迟与吞吐。

### 3.2 `custom`（可选，按需回放真实分布）

`--dataset custom --dataset-path xxx.jsonl`，每行一个 JSON，按 `task` 字段分桶后按数量循环采样。每行字段：

| 字段 | 适用任务 | 必填 | 说明 |
|------|----------|------|------|
| `task` | 全部 | ✅ | `i2t` / `t2i` / `it2i` |
| `prompt` | 全部 | ✅ | 该条请求的 prompt |
| `image_path` / `image_paths` | i2t/it2i | ❌ | 输入图；缺省回退到 `--input-image` 或占位图 |
| `bot_task` | it2i | ❌ | 显式指定；缺省按 `--it2i-bot-task-weights` 随机 |
| `width` / `height` / `num_inference_steps` / `seed` | t2i/it2i | ❌ | 覆盖 CLI 全局默认 |

示例（`custom.jsonl`）：

```json
{"task":"i2t","prompt":"What objects are in this image?"}
{"task":"t2i","prompt":"A serene mountain lake at dawn","width":1024,"height":1024,"num_inference_steps":30}
{"task":"it2i","prompt":"Make the background a starry night","bot_task":"think","width":1024,"height":1024}
{"task":"it2i","prompt":"Recaption this image with a poetic description","bot_task":"recaption"}
```

---

## 4. 快速开始

### 4.1 启动服务端

```bash
vllm serve <hunyuan-image3-ckpt> --omni --port 8099
```

### 4.2 冒烟：dry-run 只看发送计划，不发请求

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 \
    --num-i2t 14 --num-t2i 2 --num-it2i 4 \
    --it2i-endpoint chat \
    --it2i-bot-task-weights "recaption=2,think=1,think_recaption=1" \
    --shuffle --seed 3 \
    --dry-run
```

`--dry-run` 默认只打印前 20 条，加 `--dry-run-preview 0` 打印**全部**请求计划（也接受任意正整数 N 只看前 N 条）：

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --num-i2t 70 --num-t2i 20 --num-it2i 10 \
    --it2i-bot-task-weights "recaption=2,think=2,think_recaption=1" \
    --shuffle --seed 3 --dry-run --dry-run-preview 0
```

输出会先打印汇总，再按发送顺序列出每条请求的 `(idx, task, bot_task, inWxH, genWxH, prompt_words, api_url)`，可直观看到三类任务真随机交错：

```
Prepared 100 omni requests: i2t=70, t2i=20, it2i=10
it2i bot_task used: recaption=4, think=4, think_recaption=2
t2i/it2i resolution used: 1024x1024=30

[dry-run] all 100 requests in send order (idx, task, bot_task, inWxH, genWxH, prompt_words, api_url):
    0  it2i   bot_task=recaption  -           1024x1024   9    http://127.0.0.1:8099/v1/images/edits
    1  i2t    bot_task=-          -           -           5    http://127.0.0.1:8099/v1/chat/completions
    ...
   99  i2t    bot_task=-          -           -           5    http://127.0.0.1:8099/v1/chat/completions

[dry-run] No requests sent. Shuffle=True, seed=3, group_order=['i2t','t2i','it2i'].
```

> `it2i bot_task used` 的计数精确匹配权重（2:2:1 → 4/4/2），因为默认 `--it2i-bot-task-sampling proportional` 用最大余数法做精确比例分配。若想要带方差的真实随机抽样（只在期望上接近比例），加 `--it2i-bot-task-sampling random`。`WxH` 列在设了 `--gen-resolution-weights` 时显示 t2i/it2i 的分配分辨率，否则显示 `-`（i2t 始终 `-`）。`group_order` 在 `--no-shuffle` 时为 `--no-shuffle-order` 的值，`--shuffle` 时显示默认值仅供参考（实际整体随机化）。

### 4.3 7:1:2 混合配比 + 权重 bot_task + 并发测量

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 --model default \
    --num-i2t 140 --num-t2i 20 --num-it2i 40 \
    --it2i-endpoint chat \
    --it2i-bot-task-weights "recaption=2,think=1,think_recaption=1" \
    --shuffle --seed 7 \
    --request-rate 10 --max-concurrency 8 \
    --warmup-requests 8 --warmup-concurrency 8 \
    --return-stage-metrics \
    --percentiles 50 95 99 \
    --output-dir omni_7_1_2
```

### 4.4 自定义数据集

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 \
    --dataset custom --dataset-path custom.jsonl \
    --num-i2t 50 --num-t2i 10 --num-it2i 20 \
    --it2i-endpoint chat --it2i-bot-task-weights "recaption=1,think=1,think_recaption=1" \
    --shuffle --seed 42 \
    --max-concurrency 8 --return-stage-metrics \
    --output-dir omni_custom
```

### 4.5 it2i 走 images-edits 端点

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 \
    --num-i2t 10 --num-t2i 6 --num-it2i 8 \
    --it2i-endpoint images-edits \
    --it2i-bot-task-weights "recaption=2,think=1,think_recaption=1" \
    --shuffle --seed 7 \
    --max-concurrency 4 --return-stage-metrics \
    --output-dir omni_edits
```

### 4.6 关闭 shuffle（按类型分组发送，做对照组）

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --num-i2t 14 --num-t2i 2 --num-it2i 4 --no-shuffle --seed 7 --dry-run
```

加 `--no-shuffle-order` 指定分组先后顺序（须为 `i2t,t2i,it2i` 的全排列，默认 `i2t,t2i,it2i`）。例如先发 t2i、再 it2i、最后 i2t：

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --num-i2t 14 --num-t2i 2 --num-it2i 4 \
    --no-shuffle --no-shuffle-order "t2i,it2i,i2t" --seed 7 --dry-run
```

dry-run 输出会先按 `group_order` 分块列出，footer 也会打印 `group_order=['t2i','it2i','i2t']`。`--shuffle` 时该参数被忽略。

### 4.7 流式采集 i2t 文本输出的 TTFT/TPOT/ITL

i2t 的 AR 最终回答文本经 chat 流式逐 token 返回，`--stream` 直接采集：

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 --model default \
    --num-i2t 50 --num-t2i 0 --num-it2i 0 \
    --stream --return-stage-metrics \
    --request-rate 5 --max-concurrency 4 \
    --percentiles 50 95 99 \
    --output-dir omni_i2t_stream
```

### 4.8 流式采集 it2i AR "思考" 文本的 TTFT/TPOT/ITL

it2i 的 AR 思考文本走 **images-edits** 流式（`ImageEditARDeltaChunk`）才能逐 token 拿到，故需 `--it2i-endpoint images-edits --stream`：

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 --model default \
    --num-i2t 0 --num-t2i 0 --num-it2i 40 \
    --it2i-endpoint images-edits \
    --it2i-bot-task-weights "recaption=2,think=1,think_recaption=1" \
    --stream --return-stage-metrics \
    --request-rate 5 --max-concurrency 4 \
    --output-dir omni_it2i_stream
```

### 4.9 混合三类 + 流式（i2t 文本 + it2i AR 文本同时采集）

i2t 走 chat 流式、it2i 走 images-edits 流式，二者在 `--stream` 下各自采集文本延迟；t2i 文本延迟为 N/A（不计入）：

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 --model default \
    --num-i2t 140 --num-t2i 20 --num-it2i 40 \
    --it2i-endpoint images-edits \
    --it2i-bot-task-weights "recaption=2,think=1,think_recaption=1" \
    --shuffle --seed 7 \
    --stream --return-stage-metrics \
    --request-rate 10 --max-concurrency 8 \
    --warmup-requests 8 --warmup-concurrency 8 \
    --output-dir omni_stream
```

### 4.10 生图分辨率配比（t2i / it2i）

`--gen-resolution-weights` 给 t2i 与 it2i 两类生图任务按比例分配生成分辨率，允许 `512x512 / 1024x1024 / 1280x720`。默认 `proportional` 用最大余数法精确匹配权重（10 请求 2:2:1 → 4/4/2），`random` 改为带权重独立抽样。t2i 与 it2i **各自独立**按比例分配；i2t 不受影响；custom 数据集行若显式带 `width/height` 仍优先。

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 --model default \
    --num-i2t 40 --num-t2i 20 --num-it2i 20 \
    --gen-resolution-weights "512x512=2,1024x1024=2,1280x720=1" \
    --gen-resolution-sampling proportional \
    --shuffle --seed 7 --dry-run --dry-run-preview 0
```

dry-run 输出的 `genWxH` 列即每条 t2i/it2i 的生成分辨率，footer 打印 `gen_resolution_weights` 与 `sampling`，运行后 tally 段打印 `t2i/it2i resolution used:` 的实际计数（应精确匹配 2:2:1 → 各 8/8/4）。若想带方差抽样：

```bash
    --gen-resolution-weights "512x512=2,1024x1024=2,1280x720=1" \
    --gen-resolution-sampling random
```

未列出的分辨率按 0 填充（如 `1280x720=1` 只发 1280×720）；所有权重和必须 > 0。省略 `--gen-resolution-weights` 时全部用 `--width/--height`。

### 4.11 发送顺序审计（对照服务端接收日志）

运行结束会打印 `Client dispatch order matches dry-run list order: True`，并把 `config.dispatch_order`（实际进入 sender 的 idx 序列）写入 JSON。把它与服务端接收/处理日志对照，即可区分"客户端发送序"与"服务端派发序"——若 `dispatch_order == [0,1,2,...]` 而服务端日志呈 `t2i→it2i→i2t`，则重排发生在服务端 DTPS 调度器（`ar_downstream` 优先于 `ar_only`），而非客户端。详见 §2.6。

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 --model default \
    --num-i2t 14 --num-t2i 6 --num-it2i 4 \
    --shuffle --seed 7 --max-concurrency 8 \
    --output-dir omni_dispatch
# omni_dispatch/result.json -> config.dispatch_order == [0,1,...,23]
#                                              dispatch_matches_list_order == true
```

### 4.12 输入随机化（贴近真实请求分布）

`--randomize-input` 让**输入侧**逐请求随机化（由 `--seed` 控制，可复现），覆盖三类任务：

- **prompt 长度随机分布**：i2t / t2i / it2i 各有一个"短→长"的 prompt 池（从一句到长描述），逐请求从池中均匀抽取一条，使 prompt 长度真实分布。
- **输入图分辨率随机**：i2t / it2i 的输入图分辨率在 `512x512 / 1024x1024 / 1280x720` 间均匀抽取（与 `--gen-resolution-weights` 控制的**输出**生图分辨率互不影响）。
- **输入图内容随机**：i2t / it2i 的输入图逐请求用 PIL 生成不同的随机配色 + 椭圆/矩形/线条图（每条请求内容不同、可复现），不再共用一张纯色占位图。

```bash
python3 benchmarks/omni/omni_benchmark_serving.py \
    --host 127.0.0.1 --port 8099 --model default \
    --num-i2t 40 --num-t2i 20 --num-it2i 20 \
    --randomize-input --shuffle --seed 7 --dry-run --dry-run-preview 0
```

dry-run 预览表头变为 `idx, task, bot_task, inWxH, genWxH, prompt_words, api_url`：

- `inWxH`：i2t/it2i 的**输入图**分辨率（随机化时每条不同；t2i 恒为 `-`）；
- `genWxH`：t2i/it2i 的**输出生图**分辨率（由 `--gen-resolution-weights` 或 `--width/--height` 决定；i2t 恒为 `-`）；
- `prompt_words`：该请求 prompt 的词数，可直观看到长短分布。

tally 段额外打印 `prompt word count (randomized): i2t=[min-max, avg=...], t2i=..., it2i=...` 与 `i2t/it2i input-image resolution used: 512x512=N, 1024x1024=M, 1280x720=K`。JSON 的 `config` 新增 `randomize_input` / `input_resolution_actual` / `prompt_word_count`，`requests[].input` 新增 `prompt_word_count` / `input_image_size`。

注意：仅对 `random` 数据集生效（`--dataset custom` 的行始终优先，随机化被跳过）；开启时 `--input-image` 与 `--i2t/--t2i/--it2i-prompt` 被忽略。关掉时（默认）行为与旧版逐字节一致——共用一张 512×512 纯色占位图、三类各用一条固定 prompt，且不引入任何额外 RNG 抽取。`--no-randomize-input` 可显式关闭。

---

## 5. 输出说明

### 5.1 终端表格

依次输出：运行配置 → overall（吞吐/延迟/peak memory/stage durations）→ per-task（三类各自，含 stage durations）→ it2i by bot_task（思考强度分桶）。`--stream` 会在 overall / per-task / it2i-by-bot_task 各层追加 **Text-Output Streaming** 段（仅当该桶确有流式文本采集时打印）。

```
 Mix (i2t:t2i:it2i):            14:2:4
 bot_task weights:              recaption=2,think=1,think_recaption=1
 Shuffle:                       True   Seed: 7
 No-shuffle group order:        i2t,t2i,it2i          # 仅 --no-shuffle 时显示
 Gen resolution weights:        512x512=2,1024x1024=2,1280x720=1   # 仅设定时显示
 Gen resolution sampling:       proportional
 Randomize input:               True                   # --randomize-input 时为 True
 Output dir:                    omni_stream           # 仅 --output-dir 时显示，结果落在其下 result.json + inputs/ + outputs/
 Successful requests:           20/20
 Request throughput (req/s):    ...
 t2i/it2i resolution used:      512x512=8, 1024x1024=8, 1280x720=4   # 仅设定时显示
 prompt word count (randomized): i2t=[3-45, avg=22.1], t2i=[5-40, avg=20.3], it2i=[4-38, avg=18.7]   # 仅 --randomize-input 时显示
 i2t/it2i input-image resolution used: 512x512=9, 1024x1024=6, 1280x720=3                                # 仅 --randomize-input 时显示

Client dispatch order matches dry-run list order: True

Stage Durations (overall, seconds):
  ar_prefill   n=20  mean=0.0312 p50=0.0301 p99=0.0410
  ar_decode    n=20  mean=0.1804 p50=0.1782 p99=0.2205
  dit          n=6   mean=1.9034 p50=1.8810 p99=2.0102

Text-Output Streaming (overall):
  text-output (n=18, out_tokens mean=42.3 median=40.0 max=88):
    TTFT  n=18  mean=128.45ms median=121.30ms p50=121.30ms p95=178.92ms p99=192.04ms
    TPOT  n=18  mean=21.76ms  median=20.91ms  p50=20.91ms  p95=29.33ms  p99=31.50ms
    ITL   n=18  mean=21.40ms  median=20.88ms  p50=20.88ms  p95=28.70ms  p99=30.91ms

 [i2t] total=14 completed=14 failed=0 success_rate=100.00% qps=...
   i2t  n=14  mean=... p50=... p95=... p99=...
     stage ar_prefill   ...
     stage ar_decode    ...
     text-output (n=14, out_tokens mean=45.1 ...):
       TTFT  n=14  mean=125.20ms ...
       TPOT  n=14  mean=19.80ms  ...
       ITL   n=14  mean=19.50ms  ...
 [t2i] total=2  ...  stage ar_prefill / ar_decode / dit   (text-output: 无，N/A)
 [it2i] total=4 ...

 it2i by bot_task:
   [recaption] total=...  ...
     text-output (n=..., ...): TTFT ... TPOT ... ITL ...
   [think] total=...  ...
   [think_recaption] total=...  ...
```

`--output-dir` 时运行末尾还会打印落盘摘要：

```
Saved 18 input image(s) to omni_stream/inputs.
Saved 6 generated image(s) to omni_stream/outputs.
Metrics saved to omni_stream/result.json
```

> `n` 即 `measured_requests`：实际流式采集到文本的请求数。t2i（chat 流式不吐 AR 文本）与未开 `--stream` 的运行，对应行不打印（`n=0`）。

### 5.2 JSON（`--output-dir`，根目录 `result.json`）

```json
{
  "config":      { "output_dir": "omni_stream",
                   "endpoint_i2t_t2i": "...", "endpoint_it2i": "...", "mix_i2t_t2i_it2i": "14:2:4",
                   "bot_task_weights": "...", "bot_task_actual": {"recaption":2,"think":2},
                   "shuffle": true, "seed": 7, "dataset": "random", "model": "default",
                   "request_rate": "inf", "max_concurrency": 4, "stream": true,
                   "group_order": ["i2t","t2i","it2i"],
                   "gen_resolution_weights": {"512x512":2.0,"1024x1024":2.0,"1280x720":1.0},
                   "gen_resolution_sampling": "proportional",
                   "gen_resolution_actual": {"512x512":8,"1024x1024":8,"1280x720":4},
                   "randomize_input": true,
                   "input_resolution_actual": {"512x512":9,"1024x1024":6,"1280x720":3},
                   "prompt_word_count": {"i2t":{"min":3,"max":45,"avg":22.1},
                                          "t2i":{"min":5,"max":40,"avg":20.3},
                                          "it2i":{"min":4,"max":38,"avg":18.7}},
                   "dispatch_order": [0,1,2,...,19],
                   "dispatch_matches_list_order": true },
  "overall":     { "count":20, "completed":20, "failed":0, "throughput_qps": ..., "duration": ...,
                   "mean":..., "p50":..., "p95":..., "p99":...,
                   "peak_memory_mb": {"max":..., "mean":...},
                   "stage_durations": {"ar_prefill":{...}, "ar_decode":{...}, "dit":{...}},
                   "text_stream": {
                     "measured_requests": 18,
                     "output_tokens": {"total": 760, "mean": 42.3, "median": 40.0, "max": 88},
                     "ttft": {"count":18, "mean_ms":128.45, "std_ms":..., "median_ms":121.30,
                              "p50_ms":121.30, "p95_ms":178.92, "p99_ms":192.04},
                     "tpot": {"count":18, "mean_ms":21.76, "std_ms":..., "median_ms":20.91,
                              "p50_ms":20.91, "p95_ms":29.33, "p99_ms":31.50},
                     "itl":  {"count":742, "mean_ms":21.40, "std_ms":..., "median_ms":20.88,
                              "p50_ms":20.88, "p95_ms":28.70, "p99_ms":30.91}
                   } },
  "per_task": {
    "i2t":  { "total":..., "completed":..., "success_rate":..., "throughput_qps":...,
              "mean":..., "p50":..., "p95":..., "p99":...,
              "stage_durations": {"ar_prefill":{...}, "ar_decode":{...}},
              "text_stream": {"measured_requests":14, "ttft":{...}, "tpot":{...}, "itl":{...},
                              "output_tokens":{...}} },
    "t2i":  { ..., "stage_durations": {"ar_prefill":{...}, "ar_decode":{...}, "dit":{...}},
              "text_stream": {"measured_requests":0, "ttft":{"count":0,...}, ...} },
    "it2i": { ..., "stage_durations": {"ar_prefill":{...}, "ar_decode":{...}, "dit":{...}},
              "text_stream": {"measured_requests":4, "ttft":{...}, "tpot":{...}, "itl":{...},
                              "output_tokens":{...}} }
  },
  "it2i_by_bot_task": { "recaption": {..., "text_stream":{...}},
                        "think": {..., "text_stream":{...}},
                        "think_recaption": {..., "text_stream":{...}} },
  "requests": [
    { "index":0, "task_type":"i2t",
      "input":  { "endpoint":"/v1/chat/completions", "model":"default", "prompt":"...",
                   "prompt_word_count":22, "input_image_size":"1024x1024",
                   "image_paths":["<tmp>/omni_bench_input_0.png"],
                   "input_images_saved":["omni_stream/inputs/req_0000_i2t_input.png"],
                   "modalities":["text"], ... },
      "output": { "success":true, "latency_s":1.023, "returned_text":"...", "num_images":0,
                  "image_paths_saved":[],
                  "stage_durations":{...}, "peak_memory_mb":...,
                  "text_stream": { "ttft_s":0.121, "output_tokens":45, "tpot_s":0.0202,
                                   "itl_s":[0.0198,0.0201,...], "generated_text":"..." } } },
    { "index":1, "task_type":"it2i",
      "input":  { "endpoint":"/v1/images/edits", "prompt":"...", "prompt_word_count":14,
                   "input_image_size":"512x512", "bot_task":"think",
                   "image_paths":["<tmp>/omni_bench_input_1.png"],
                   "input_images_saved":["omni_stream/inputs/req_0001_it2i_input.png"] },
      "output": { "success":true, "latency_s":2.451, "cot_output":"...", "num_images":1,
                  "image_paths_saved":["omni_stream/outputs/req_0001_it2i_0.png"],
                  "stage_durations":{...}, "peak_memory_mb":...,
                  "text_stream": { "ttft_s":0.083, "output_tokens":32, "tpot_s":0.0759,
                                   "itl_s":[0.072,0.074,...], "generated_text":"..." } } }
  ]
}
```

> `it2i` 走 `images-edits` 时，`per_task.it2i.stage_durations` 与对应 bot_task 桶的 stage 统计在**非流式**下为空（edits 响应体不携带 stage_durations）；但 `--stream` 下 edits 流的 image chunk 携带 `metrics`，故 `stage_durations` 会随流式补全（见 §2.5）。chat 端点则始终完整。
>
> `requests[].output.text_stream` 仅在该请求确实采集到文本流时出现（`ttft_s>0`）；非流式或 N/A 请求无此字段。`generated_text` 与 `returned_text`（i2t）/ `cot_output`（it2i edits）一致，是流式拼接的原文，便于直接核对 TTFT/TPOT。
>
> 落盘布局（`--output-dir omni_stream`）：`omni_stream/result.json` 写指标；`omni_stream/inputs/req_{idx}_{task}_input.png` 落 i2t/it2i 的输入参考图（逐请求从源文件复制）；`omni_stream/outputs/req_{idx}_{task}_{img}.png` 落 t2i/it2i 的生成图。it2i 同时有 `inputs/` 与 `outputs/` 条目，按 idx 对齐即可前后对比；i2t 只有 `inputs/`、t2i 只有 `outputs/`。`input.image_paths` 是源路径，`input.input_images_saved` 是落盘副本；`output.image_paths_saved` 是生成图落盘路径。

---

## 6. 参数速查

| 参数 | 说明 |
|------|------|
| `--base-url` / `--host` `--port` | 服务端地址，`--base-url` 优先 |
| `--num-i2t` / `--num-t2i` / `--num-it2i` | 三类任务请求数（配比来源） |
| `--it2i-endpoint {chat,images-edits}` | it2i 走统一 chat 端点（默认）或 edits 端点 |
| `--it2i-bot-task-weights` | 如 `recaption=2,think=1,think_recaption=1`；缺省等权重；名字须在三者内；权重和 > 0 |
| `--it2i-bot-task-sampling {proportional,random}` | `proportional`（默认）：最大余数法精确分配，计数严格匹配权重；`random`：带权重独立抽样，仅期望匹配 |
| `--shuffle` / `--no-shuffle` | 真随机交错发送（默认）或按类型分组发送 |
| `--no-shuffle-order ORDER` | `--no-shuffle` 下三组的拼接顺序，须为 `i2t,t2i,it2i` 的全排列，如 `t2i,it2i,i2t`；默认 `i2t,t2i,it2i`；`--shuffle` 时忽略 |
| `--seed` | RNG 种子（shuffle + bot_task + 分辨率采样共用，可复现） |
| `--dataset {random,custom}` / `--dataset-path` | 数据源；custom 需 JSONL |
| `--input-image` | random 模式下 i2t/it2i 输入图，缺省用占位图 |
| `--i2t-prompt` / `--t2i-prompt` / `--it2i-prompt` | 覆盖 random 模式的默认 prompt |
| `--gen-resolution-weights WEIGHTS` | t2i/it2i 生图分辨率配比，如 `512x512=2,1024x1024=2,1280x720=1`；允许 `512x512/1024x1024/1280x720`；缺省则全用 `--width/--height` |
| `--gen-resolution-sampling {proportional,random}` | `proportional`（默认）：最大余数法精确分配；`random`：带权重独立抽样，仅期望匹配 |
| `--randomize-input` / `--no-randomize-input` | 仅 `--dataset random` 下生效。开启后每条请求的**输入**侧随机化：prompt 长度按任务从内置池随机抽取、输入图分辨率在 `512x512/1024x1024/1280x720` 间随机、输入图内容每张独立生成（非同一张纯色图）。由 `--seed` 驱动可复现；开启时忽略 `--input-image` 与 `--i2t/--t2i/--it2i-prompt`；默认关（关闭=与历史字节一致） |
| `--model` `--width` `--height` `--num-inference-steps` `--seed-gen` | 生成参数（`--width/--height` 在设了 `--gen-resolution-weights` 时被 t2i/it2i 逐请求覆盖） |
| `--request-rate` | Poisson 速率（req/s），`inf` 一次性全发（默认） |
| `--max-concurrency` | 在途请求上限 |
| `--warmup-requests` `--warmup-concurrency` | 预热 |
| `--return-stage-metrics` | 请求服务端返回 per-stage durations |
| `--stream` | 流式采集文本输出 TTFT/TPOT/ITL（i2t AR 回答；it2i AR 思考需配 `--it2i-endpoint images-edits`）；不开则 `text_stream` 各桶 `count=0` |
| `--percentiles` | 报告的分位（默认 50 95 99） |
| `--output-dir DIR` | 统一落盘目录：`DIR/result.json`（指标，固定名）+ `DIR/inputs/`（i2t/it2i 输入参考图，逐请求复制）+ `DIR/outputs/`（t2i/it2i 生成图）；it2i 前后可按 idx 对齐对比。旧 `--output-file`/`--save-dir` 仍接受（弃用，分别映射到其父目录/该目录） |
| `--disable-tqdm` `--dry-run` | 关进度条 / 只打印发送计划不发请求 |
| `--dry-run-preview N` | 配合 `--dry-run`，打印前 N 条请求计划；`0` = 全部（默认 20） |

### bot_task 权重说明

合法取值（与服务端 `_BOT_TASK_PRESETS` 对齐）：`think`、`recaption`、`think_recaption`。

- `--it2i-bot-task-weights "recaption=2,think=1,think_recaption=1"` 设定权重。
- `--it2i-bot-task-sampling proportional`（默认）：用最大余数法把 it2i 任务**精确**按权重分配到三类，计数严格匹配（如 10 任务 2:2:1 → 4/4/2，100 任务 → 40/40/20），跨 seed 可复现；适合需要可控、可复现负载的 benchmark。
- `--it2i-bot-task-sampling random`：每个 it2i 任务按权重独立抽样（`rng.choices`），只在**期望**上接近权重，会有方差（如 10 任务 2:2:1 可能抽到 6/3/1）；适合需要真实随机波动的压测。
- 未列出的取值权重按 0 填充；所有权重和必须 > 0。
- 省略 `--it2i-bot-task-weights` 时三值等权重。
- 自定义数据集中若某 it2i 行显式带 `bot_task`，则该行用显式值，不参与上述分配/采样（这会使实际计数略微偏离设定比例）。

### 生图分辨率配比说明

合法取值：`512x512`、`1024x1024`、`1280x720`（与 `omni_dataset.GEN_RESOLUTIONS` 对齐）。

- `--gen-resolution-weights "512x512=2,1024x1024=2,1280x720=1"` 设定权重。t2i 与 it2i **各自独立**按该权重分配生成分辨率（各算一份最大余数法 / 独立抽样）。
- `--gen-resolution-sampling proportional`（默认）：最大余数法**精确**匹配权重，10 请求 2:2:1 → 各 4/4/2，跨 seed 可复现。
- `--gen-resolution-sampling random`：每个 t2i/it2i 请求按权重独立抽样，仅期望匹配，会有方差。
- 未列出的分辨率按 0 填充；所有权重和必须 > 0。
- 仅作用于 t2i / it2i；i2t 始终用 `--width/--height`（理解任务不生图）。
- 自定义数据集行若显式带 `width`/`height`，则该行用显式值（`gkw.setdefault`，仅当行未指定时才回填分配到的分辨率），会使实际计数略微偏离设定比例。
- 省略 `--gen-resolution-weights` 时全部用 `--width/--height`，与旧版完全一致。

---

## 7. 评估优化结果的建议用法

1. **对照组**：`--no-shuffle`（按类型分块）vs `--shuffle`（真随机交错），同样配比/并发，比较 overall qps 与 per-task p99 —— 体现混合调度在交错负载下的优化收益。进一步用 `--no-shuffle-order` 切换分组先后（如 `t2i,it2i,i2t` vs `i2t,t2i,it2i`），观察"先发 ar_downstream"与"先发 ar_only"对调度器排队的影响。
2. **配比扫描**：固定总并发，跑 `7:1:2`、`5:3:2`、`3:3:4` 等不同 `--num-*` 组合，看 DTPS 在不同 i2t/t2i/it2i 负载下的吞吐与尾延迟。
3. **生图分辨率配比**：`--gen-resolution-weights "512x512=2,1024x1024=2,1280x720=1"` 给 t2i/it2i 混入不同分辨率，评估 DiT 阶段在大/小图混合下的吞吐与尾延迟（大图 1280×720 的 dit 耗时显著高于 512×512）。
4. **思考强度影响**：对比 `it2i_by_bot_task` 各桶延迟，评估 recaption/think/think_recaption 对 AR 阶段耗时与整体调度的影响。
5. **per-stage 定位**：`--return-stage-metrics` 下观察 `ar_prefill / ar_decode / dit` 各阶段 p99，定位优化点落在 AR 还是 DiT。
6. **文本体感延迟**：`--stream` 下看 `text_stream` 的 TTFT（首字等待）/ TPOT（生成流畅度）p95/p99，配合 `--return-stage-metrics` 的整阶段耗时，区分"AR 预热慢"（TTFT 高、ar_prefill 高）与"AR decode 慢"（TPOT 高、ar_decode 高）。it2i 需 `--it2i-endpoint images-edits` 才能拿到 AR 思考文本的 TTFT/TPOT。
7. **发送/派发顺序对照**：用 `config.dispatch_order`（客户端实际发送序，应 == `[0,1,...,N-1]`）对照服务端接收/处理日志——若服务端序与 `dispatch_order` 不一致，重排发生在 DTPS 调度器（`ar_downstream` 优先于 `ar_only`），正是混合调度评估要测的对象，而非客户端 bug（见 §2.6）。
8. **输入随机化（贴近真实分布）**：`--randomize-input` 让每条请求的输入 prompt 长度、输入图分辨率（512x512/1024x1024/1280x720）、输入图内容各不相同，模拟真实世界请求多样性，评估 DTPS 在输入异构负载下的调度稳定性与尾延迟（对比开关两次 run，相同 seed 可复现）。注意 `--randomize-input` 控制的是**输入**侧；生图**输出**分辨率仍由 `--gen-resolution-weights` 控制，两者正交，可叠加使用。
9. 结果用 `--output-dir` 统一落盘（`result.json` + `inputs/` + `outputs/`），便于跨 run 画图对比（参考 `diffusion/README.md` 中 performance_dashboard 的做法）；it2i 的 `inputs/` 参考图与 `outputs/` 生成图按 idx 对齐，可直接前后对照编辑效果。
