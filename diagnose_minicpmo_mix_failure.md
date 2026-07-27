# MiniCPM-o 4.5 在线用例 test_mix_to_text_audio_001 失败 — 根因分析

## 0. 现象回顾(来自用户日志,NPU 路径)

```
04:14:38  WARNING mm_outputs.py:54 Error concatenating tensor for key meta.finished; keeping last tensor   (多次)
04:14:39  MiniCPM-o Code2Wav backend was not built during weight loading (load_format=dummy);
           loading Token2wav assets now.
04:14:42  Token2Wav models loaded successfully / Patched Step-Audio2 HiFT linear downsample for Ascend NPU
04:15:12  Error: Stream processing error: timed out
04:24:42  (大量) Aborted request(s) chatcmpl-...
04:25:12  FAILED
```

该用例在 upstream `vllm-omni` 已被 `@pytest.mark.skip(reason=...issues/5437)`(commit `138c6988`)。
本分析针对"如何在 NPU + async_chunk 下真正修复 #5437"给出代码级定位。

---

## 1. 终止信号(meta.finished)完整链路

请求结束标志的传递路径(基于本地最新代码):

1. **Stage1 Talker (TTS)** 在 `make_omni_output` 中产出终止标志:
   `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py:459`
   ```python
   finished = is_eos or reached_limit        # is_eos = 采样到 audio EOS token
   state["finished"] = finished
   terminal_flags.append(torch.tensor(finished, dtype=torch.bool))
   ...
   "meta": {"finished": terminal_flags},      # :481
   ```
   → 这是**整条链路的源头**。若 TTS 永远不采样到 EOS 且不达 max_tokens,`finished` 恒为 False。

2. **Stage1 → Stage2 桥梁** `tts2code2wav_async_chunk`:
   `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py:278`
   ```python
   finished = bool(is_finished or (callable(request_finished) and request_finished()))
   ...
   last_chunk = bool(finished and not pending)   # :305  —— 决定 code2wav 是否收到"末 chunk"
   finished_tensor = torch.tensor(last_chunk, dtype=torch.bool)
   ```
   然后构造 `OmniPayloadStruct(meta=..., finished=finished_tensor, stream_finished=finished_tensor)` (:327-328)。

3. **Stage 间 connector 发送端** `chunk_transfer_adapter._send_single_request`:
   `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py:334`
   ```python
   payload_data.meta.finished = torch.tensor(is_finished, dtype=torch.bool)   # 注入到下一 stage 的 payload
   ```

4. **Stage2 (Code2Wav) 接收端** `omni_connector_model_runner_mixin` (non-ar 分支):
   `vllm_omni/worker/omni_connector_model_runner_mixin.py:1773`
   ```python
   is_finished = self._payload_finished(payload_data)   # 读 meta.finished
   ...
   if is_finished:
       self._chunk_finished_req_ids.add(req_id)
       self._chunk_stream_completed.add(req_id)
   ```
   注意 `minicpmo_4_5_code2wav.py` 的 forward **不产出** `meta.finished`(只产 `meta.tts_is_last_chunk` 等,见 :535-558),
   所以 APIServer 最终看到的 `meta.finished` 是 connector 在 stage2 被 scheduler 标 finished 时注入的,
   而 scheduler 标记 stage2 request finished 的前提是 **stage2 收到了 `last_chunk=True` 的末 chunk**。

5. **Orchestrator → APIServer**:
   `vllm_omni/engine/orchestrator.py:1216` `finished = output.finished`,
   当 stage2(final_output) finished 时 `OutputMessage.finished=True`(:1253)。

6. **APIServer 流终止**:
   `vllm_omni/entrypoints/async_omni.py:850` `if result.finished: break`,
   随后 `serving_chat.py:2206` 下发 `[DONE]`。

**闭环结论**:只要 Stage1 的 TTS 不产出 `meta.finished=True`,后面每一环的终止标志都为 False,
APIServer 永远收不到 `finished=True`,流挂起 → 客户端 120s 超时 → abort → FAILED。
`mm_outputs.py:54` 的 WARNING 只是**症状**(中间 chunk `meta.finished=False` cat 失败 keep-last,但末 True 永远不来),
**不是根因**。

---

## 2. 为什么只有 mix(全模态)用例触发 #5437

测试矩阵(同文件 `test_minicpmo_4_5.py`):
- `test_text_to_text_001`        → **skipped** (也是 #5437)
- `test_text_to_audio_001`       → 单模态(text) 未 skip
- `test_audio_to_text_audio_001` → 双模态(audio) 未 skip
- `test_image_to_text_audio_001` → 双模态(image) 未 skip
- `test_video_to_text_audio_001` → 双模态(video) 未 skip
- `test_mix_to_text_audio_001`   → **skipped** (也是 #5437)  ← 全模态 text+audio+video+image

只有 **mix(4 模态同时输入)** 和纯 text 用例被 skip。纯 text 在 `async_chunk` + NPU 下同样可能不触发 TTS
(没有音频/图像/视频条件),所以 mix 是"全模态混合"这一特定输入形态触发了 TTS 终止信号缺失。

最可疑的代码点:`llm2tts`(`minicpmo_4_5_omni.py`)在把 Thinker(stage0) 输出切成 TTS 条件时:
```python
# :828
if not handoff_ids:
    continue          # ← 若某 segment 的 handoff_ids 为空,直接跳过,
                      #    不会执行后面的 native_turn_end_handoff / turn_end 设置(:831-832)
```
全模态输入下,Thinker 可能产出某些 segment 的 `handoff_ids` 为空(尤其是 async_chunk 的边界切分与
多模态交织时),导致 `turn_end` / `segment_end` 漏设,进而 `tts2code2wav_async_chunk` 中
`state["turn_end"]` 恒为 False,code2wav 的 `last_chunk` 永不为 True。

另一个候选点:`make_omni_output` 里的 `sample_eligible`(:400)。若 mix 下某 request 的
`sample_eligible[index]=False`(:439),会直接 `terminal_flags.append(False)`,永不发 finished。

---

## 3. 时间线证据(指向"终止信号根本没产生",而非性能慢)

- 所有 `meta.finished` WARNING 都集中在 **04:14:38~04:14:42**,即 Token2Wav 加载期间。
  这说明在 Token2Wav 还没 build 完时,stage2 已经在转发 stage1 的 chunk,
  但此时 audio 解码路径阻塞,发出的全是中间态 chunk(`finished=False`),且**没有后续终止 chunk**。
- 第一个 stream timeout 在 **04:15:12**(≈ 加载完 30s 后),说明请求早就进入了音频解码阶段但卡死,
  不是单纯"首 token 慢"。
- **04:24:42 的批量 abort** 距离加载完正好约 10 分钟,对应于 stage 初始化/推理的长超时窗口
  (见 `runtime.py` 的 `stage_init_timeout=600` / `init_timeout=1800`),即进程侧一直没正常结束。

---

## 4. 诊断方案(在 NPU 机器上跑一次,定位信号断点)

在以下 3 个函数加 `logger.info` 追踪 `finished` / `last_chunk` 真实值(只打印 request 末态,避免刷屏):

### (a) Stage1 TTS 是否产出 finished (源头)
`minicpmo_4_5_omni_tts.py` `make_omni_output` 内,`:481` 附近:
```python
any_finished = any(bool(t.item()) for t in terminal_flags)
if any_finished:
    logger.info("[DIAG][TTS] req terminal_flags=%s", [bool(t.item()) for t in terminal_flags])
```

### (b) Stage1→Stage2 桥梁 last_chunk 判定
`minicpmo_4_5_omni.py` `tts2code2wav_async_chunk`,`:305` 附近:
```python
logger.info("[DIAG][bridge] req=%s finished=%s pending=%s last_chunk=%s",
             request_id, finished, len(pending), last_chunk)
```

### (c) Stage2 接收端 is_finished
`omni_connector_model_runner_mixin.py`,`:1773` 附近:
```python
if is_finished:
    logger.info("[DIAG][stage2-recv] req=%s is_finished=True", req_id)
```

**预期判断**:
- 若 (a) 从不打印 `terminal_flags` 含 True → 根因在 **TTS 不采样 EOS / sample_eligible=False**(§2 候选点)。
- 若 (a) 有 True 但 (b) 的 `last_chunk` 恒为 False → 根因在 `tts2code2wav_async_chunk` 的
  `finished`/`last_chunk` 判定(§2 候选点,或 stage1 发来的 `meta.finished` 未被 `_payload_finished` 正确读取)。
- 若 (b) 有 `last_chunk=True` 但 (c) 收不到 `is_finished=True` → 根因在 connector 透传(chunk_transfer_adapter)。

---

## 5. 修复方向(按根因分层)

### 方向 A:若根因是 TTS 不产出 finished(§2 候选点)
- 排查 `sample_eligible` 在 mix 下为何为 False;或确认 mix 下 Thinker 是否确实生成了 audio EOS。
- 兜底:当 `meta.turn_end` 或 `meta.segment_end` 为 True 且 pending 已清空时,强制 `finished=True`,
  避免无限等待 audio EOS(参考 :305 `last_chunk = bool(finished and not pending)` 的语义)。

### 方向 B:若根因是 `llm2tts` 漏设 turn_end(§2 候选点,最可能)
- `minicpmo_4_5_omni.py:828` 的 `if not handoff_ids: continue` 在 mix 全模态下会跳过
  `native_turn_end_handoff` 判定。需确认:当 handoff_ids 为空是因为"该 segment 本就无 TTS 内容"还是
  "async_chunk 边界切分错误吞掉了 handoff"。若是后者,应在 continue 前根据已累积的
  `native_turn_end_handoff` 状态补设 `turn_end`。

### 方向 C:消除 `meta.finished` WARNING 噪音(顺手,非根因)
`mm_outputs.py:42-61` 的 `_consolidate_tensor_list` 对 `meta.finished` 这类布尔标量应直接走
"keep last / any-True" 语义,而非 `cat` → warn → keep last。可加:
```python
if key in ("meta.finished", "meta.is_segment_finished"):
    # 布尔标量:任意 chunk 为 True 即视为完成
    import torch
    return torch.tensor(any(bool(t.item()) for t in tensor_list), dtype=torch.bool)
```

---

## 6. 关键文件清单
- `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py` (TTS finished 源头 :459/:481)
- `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py` (llm2tts :828, bridge :204-341)
- `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` (meta.finished 注入 :334)
- `vllm_omni/worker/omni_connector_model_runner_mixin.py` (_payload_finished :485, stage2 recv :1773)
- `vllm_omni/engine/orchestrator.py` (output.finished :1216, OutputMessage :1253)
- `vllm_omni/entrypoints/async_omni.py` (流终止 :850)
- `vllm_omni/outputs/mm_outputs.py` (WARNING 根 :54)
- 测试:`vllm_omni/tests/e2e/online_serving/test_minicpmo_4_5.py` (L204 skip, L210 test_mix_to_text_audio_001)
- 配置:`vllm_omni/deploy/minicpmo_4_5_batching.yaml` (async_chunk: true, 3 stage)

---

## 7. 用户诊断结果(2026-07-27, NPU 实测)

在 3 个函数加了 `[DIAG-5437]` 日志后在 NPU 重跑 mix 用例,关键输出:

```
[DIAG-5437][TTS] finished flags=[True]      ← TTS 源头确实产出了 finished=True
[DIAG-5437][bridge]   (无输出)               ← Stage1→Stage2 桥梁从未打印
[DIAG-5437][stage2-recv] (无输出)            ← Stage2 接收端从未收到 is_finished=True
```

**这推翻了 §2 的方向 B(llm2tts 漏设 turn_end 导致 last_chunk 恒 False)。**
方向 B 的前提是 bridge 被调用且走到了 last_chunk 判定;但实测 bridge 根本没打印,
说明它要么没被调用,要么每次都在 `tts2code2wav_async_chunk:280` 的
`if not finished and len(pending) < chunk_frames: return None` 早退 —— 而 `finished` 一直为 False。

### 第 1 轮结论(保留)
TTS 已正确报 `meta.finished=True`,但 bridge 在 `if not finished and len(pending) < chunk_frames: return None`
早退(`finished` 永 False)→ stage2 永远收不到 `meta.finished=True` → 流挂起超时。

---

## 8. 第 2 轮修复尝试(已证伪:改错了路径)

曾误判修复点在生产者适配器 `chunk_transfer_adapter._send_single_request`(加了
`_extract_producer_finished` + `[FIX-5437][send]` 日志)。**用户重跑后 `[FIX-5437][send]`
仍不打印**,证明 worker 架构下实际发送走的是
`omni_connector_model_runner_mixin.send_chunk`(:1091) → `_build_custom_process_payload`(:1849)
→ **直接调用 bridge** `tts2code2wav_async_chunk`,**完全不经过 `chunk_transfer_adapter._send_single_request`**。
→ 该路径修改是死代码,**已全部回退**。

---

## 9. 真正的根因(代码级,确定)

worker 侧真实调用链:
```
send_chunk(mixin:1091)
  → _build_custom_process_payload(mixin:1849)
      is_finished = request.is_finished()        # :1873 = stage1 TTS 的 scheduler 级完成标志
      → bridge tts2code2wav_async_chunk(minicpmo_4_5_omni.py:204)
          finished = bool(is_finished or request.is_finished())   # :278 只认这俩
  → connector.put()
```

bridge 的 `finished`(:278)**只认 `is_finished` 形参(= stage1 scheduler 标志)和 `request.is_finished()`,
完全不读 TTS 自己输出到 `multimodal_output` 里的 `meta.finished`**。

在 mix 全模态 + NPU 下:TTS 模型已自报 `meta.finished=True`(第 1 轮 `[DIAG-5437][TTS]` 证实),
但 stage1 的 `request.is_finished()`(scheduler)未能与之同步变 True → bridge 的 `finished` 永 False
→ 在 :320(原 :280)早退 → **最后一包 audio 永远发不出去** → stage2 收不到 `meta.finished=True`
→ 流挂起超时。

根因一句话:**producer 已正确报 finished,但 bridge 只读 scheduler 标志、不读 producer 的
`meta.finished`,而 scheduler 标志在 mix/NPU 下滞后。**

---

## 10. 已落地修复(方向 E:bridge 直接读 producer meta.finished)

**修改文件**: `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`

### (1) 新增模块级 `_producer_finished(multimodal_output)`
读 `meta.finished`(兼容 dict / 含 `multimodal_outputs` 的对象 / tensor / list),
对不输出该字段的模型返回 `False`(no-op)。

### (2) bridge 内并入 producer finished(:303-311 附近)
```python
producer_finished = _producer_finished(multimodal_output)
request_finished = getattr(request, "is_finished", None)
finished = bool(
    is_finished
    or (callable(request_finished) and request_finished())
    or producer_finished
)
```
并在早退**之前**(`:313`)加诊断日志:
```python
logger.info("[FIX-5437][bridge] req=%s is_finished=%s producer_finished=%s "
            "scheduler_finished=%s finished=%s pending=%d chunk_frames=%d", ...)
```
> 此改动仅影响真正输出 `meta.finished` 的模型(MiniCPM-o TTS 目前唯一生产者),
> 对其它 omni 模型是 no-op,不改变既有行为。修复后末包 `last_chunk=True`、`meta.finished=True`
> 能正确下发给 stage2,流终止。

---

## 11. 验证步骤(在 NPU 重跑 mix 用例)

预期日志顺序(修复生效):
```
[DIAG-5437][TTS]        finished flags=[True]                          # TTS 产出(已有)
[FIX-5437][bridge]      ... producer_finished=True finished=True pending=...  # 修复点生效,不再早退
[DIAG-5437][stage2-recv] ... is_finished=True                           # stage2 收到终止
# 不再出现 Stream processing error: timed out,用例 PASS
```

- 若 `[FIX-5437][bridge]` 的 `producer_finished` 仍为 `False` → 说明传给 bridge 的
  `multimodal_output` 里 `meta.finished` 不在 `_producer_finished` 预期位置。把该日志旁的
  payload 实际结构贴回,微调提取器即可。
- 若 `producer_finished=True` 且 `finished=True` 但 `[DIAG-5437][stage2-recv]` 仍收不到 →
  说明 stage2 接收端 `_payload_finished` 没从 payload 读到 `meta.finished`,需查
  `_MiniCPMO45MetaStruct.finished` 的透传链。

---

## 12. 从 [DIAG-5437][TTS] 之后的数据传输顺序(逐跳)

`[DIAG-5437][TTS]` 打印后,数据在以下顺序中流动;**bug 在第 5 跳断开,之后全不发生**。

| # | 阶段 | 文件:行 | 动作 | 备注 |
|---|------|---------|------|------|
| 1 | Stage1 TTS | `minicpmo_4_5_omni_tts.py:459,481` | `make_omni_output` 写 `meta.finished=True` | TTS 自己报的逐 step 标志 |
| 2 | Stage1 Worker | `omni_connector_model_runner_mixin.py:1091` | `send_chunk` 取 `OmniOutput` | 仅 data-transfer rank |
| 3 | Stage1 Worker | `:1849` `_build_custom_process_payload` | 调 bridge,传 `is_finished=request.is_finished()` | scheduler 级,非 `meta.finished` |
| 4 | Stage1 Worker | `minicpmo_4_5_omni.py:232,305` | `tts2code2wav_async_chunk` 读 `producer_finished` | `[FIX-5437][bridge]` 在早退前打印 |
| 5 | **断点** | `minicpmo_4_5_omni.py:320` | `if not finished and len(pending)<chunk_frames: return None` | 修复前 `finished` 恒 False → 早退 |
| 6 | Stage1 Worker | `:353,368` | 修复后构造 `OmniPayloadStruct`,`meta.finished=last_chunk` | 走到 `connector.put` |
| 7 | Connector | `omni_connector_model_runner_mixin.py:1944` | `_send_single_request` → `connector.put` 跨进程 | Stage1 worker → Stage2 worker 队列 |
| 8 | Stage2 | `:1773,1790` | `recv` → `_payload_finished` 读 `meta.finished` | 打印 `[DIAG-5437][stage2-recv]` |
| 9 | APIServer | `async_omni.py:850` / `serving_chat.py:2206` | `OutputMessage.finished` → `break` → `[DONE]` | 客户端收到终止,不再 timeout |

断点机制:第 5 跳早退 → 第 6/7 跳(构造 payload + `connector.put`)永不发生 →
Stage2 收不到任何东西 → `[DIAG-5437][stage2-recv]` 永不打印 → APIServer `result.finished`
永 False → 流挂起超时。这与实测「只打印第一个日志、后面全无、最终 timeout」完全吻合。

> 注意:第 7 跳走的是 **worker 侧** `omni_connector_model_runner_mixin._send_single_request`
> (`:1944`),**不是** `distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`
> 里那个同名函数(死代码路径,第 2 轮已证伪并回退)。

---

## 13. FAQ:Token2Wav "not built during weight loading" 警告是正常的

日志里这条 WARNING 是**设计内的预期行为,不是 bug,也与 #5437 超时无关**:

```
MiniCPM-o Code2Wav backend was not built during weight loading
(load_format=dummy); loading Token2wav assets now.
```

### 为什么"初始化时没装好"

backend 的构建分三步,**初始化(`__init__`)并不构建它**:

1. **`__init__`** (`minicpmo_4_5_code2wav.py:121`):`self.backend = None`。
   构造时 backend 就是 `None`,**没有任何构建动作**。
2. **`load_weights()`** (`:574`):本应是唯一真正构建 backend 的入口——
   它 `for _ in weights: pass`(丢弃传入权重)然后 `self._build_backend()`(`:577`)。
   之所以丢弃权重:Token2Wav 的 `flow.pt`/`hift.pt` **不在 checkpoint 的权重迭代器里**,
   而是单独放在 `assets/token2wav` 目录,由 `_build_backend`(`:606-623`)独立加载。
   所以 code2wav 的 `load_weights` 主要工作就是"构建 backend"。
3. **`load_format=dummy` 会跳过 `load_weights()`**:这是 CI 验证模型结构时常用的格式,
   只用占位 tensor 把计算图跑通,不真正从磁盘加载权重。代码注释(`:408`)明确写了
   *"load_format=dummy (CI core_model runs) skips model.load_weights()"*。
   → 于是 `_build_backend()` 在初始化阶段**没被调用**,backend 保持 `None`。

### 第一次 forward 才 lazy build

`forward`(`:407`)检测到 `self.backend is None`,打印该 WARNING(`:413-416`,含
`load_format=%s`),然后在 `:418-419` 现场 `_build_backend()`。这就是日志里看到的那条。

### 为什么设计成 lazy 而非在 `__init__` 里直接构建

- Token2Wav 的 assets 在 checkpoint 旁边(`assets/token2wav`),不在 weight iterator 内(`:578-582`)。
- lazy build 必须在 `torch.inference_mode(False)` 下构建(`:418` 的 `with` 块)——
  vLLM 在 bf16 模型构造时处于 inference_mode 上下文,Token2Wav 的参数需要是普通 tensor,
  不能在 `__init__` 的 inference_mode 下正确构建。`load_weights` 正常路径会在合适的上下文调用,
  dummy 路径跳过它,就退化成"首次 forward 时脱离 inference_mode 再构建"。

### 与 #5437 超时的关系

**无关。** 时间线佐证:4:14:39 开始 lazy build Token2Wav → 4:14:42 完成(约 3 秒,NPU 正常加载),
而第一个 `timed out` 在 4:15:12,是 build 完成约 30 秒之后。lazy build 的 3 秒延迟不是超时根因,
#5437 的真正根因是 §9 的 bridge `finished` 信号链断开。即便 backend 在初始化时构建好
(生产 `load_format=auto`),#5437 仍会复现。

### 何时不会出现这条 WARNING

生产配置 `load_format=auto`(默认真实加载权重)下,`load_weights()` 被正常调用 → backend 在
初始化阶段就构建好 → `forward` 时 `self.backend is not None` → 不打印。
只有 `load_format=dummy`(CI 结构验证)才会出现。

---

## 14. 真正的根因:`--no-async-chunk` 把用例推入了 miniCPM-o 4.5 **未接线的同步路径**

> 关键实证(用户在 NPU 上验证):去掉 `--no-async-chunk`(即 `async_chunk=True`)后,
> `[DIAG-5437][bridge]` 与 `[DIAG-5437][stage2-recv]` 都能打印 → 说明 **async 路径本身是对的**,
> 卡死发生在 `--no-async-chunk` 强制的 **sync 路径**里。

### 14.1 在线用例实际走的是哪条路

`tests/e2e/online_serving/test_minicpmo_4_5.py`:

- 文件顶部 docstring 自述:`MiniCPM-o 4.5 has async_chunk: false … the vocoder runs
  in-process inside the talker stage rather than as a separate Code2Wav stage.`
  —— 这是**与代码自相矛盾的陈旧描述**(见 14.4)。
- `test_params`(第 23-36 行)对**所有**测试固定带 `--no-async-chunk`,`test_mix_to_text_audio_001`
  用的就是它 → `async_chunk=False` → **sync / full_payload 路径**。

而 `vllm_omni/deploy/minicpmo_4_5_batching.yaml:6` 写的是 `async_chunk: true`,NPU 平台覆盖
(`platforms.npu`)也只改 `max_batched_tokens` / `cudagraph_mode`,**没有改 async_chunk**。
即:部署默认是 async,测试却用 CLI 覆盖成 sync。

### 14.2 sync 路径在哪个环节断掉

stage1(talker)→ stage2(code2wav) 的桥梁函数由 pipeline 决定
(`model_executor/models/minicpmo_4_5/pipeline.py`):

```python
# stage_id=1 (talker)
custom_process_input_func=f"{_PROC}.llm2tts",                              # 输入侧(上游 latent→talker)
async_chunk_process_next_stage_input_func=f"{_PROC}.tts2code2wav_async_chunk",  # 仅 async 输出侧
# 没有 sync_process_input_func / 没有 custom_process_next_stage_input_func
```

- **async 模式**(`config/stage_config.py:787`):`next_stage_proc = async_chunk_process_next_stage_input_func`
  = `tts2code2wav_async_chunk`。该函数在 `engine_args` 里被写成 `custom_process_next_stage_input_func`,
  于是 `_load_custom_func`(`omni_connector_model_runner_mixin.py:2096`)首选它就命中 →
  正确累积 codec、补 left-context、产出带 `finished_tensor` 的 `OmniPayloadStruct` → stage2 收到终止 → 流结束。
- **sync 模式**(`stage_config.py:789`):`elif not async_chunk and ps.sync_process_input_func` ——
  此处只改 `input_proc`,**`next_stage_proc` 落到默认的 `custom_process_next_stage_input_func` = None**。
  `_load_custom_func` 在 `async_chunk=False` 时转而拿 `custom_process_input_func=llm2tts`,
  推导候选:`llm2tts_full_payload` / `llm2tts_batch` / `llm2tts`
  (`mixin.py:2111-2123`)。

  - `llm2tts` 签名(`minicpmo_4_5_omni.py:611`):
    `(source_outputs, prompt, requires_multimodal_data, _streaming_context)` ——
    **不含** `transfer_manager` / `pooling_output` / `request`。
  - `_is_connector_payload_builder`(`mixin.py:2156`)要求这三个参数 ⊆ 签名支持集 → `llm2tts` 判 False。
  - `llm2tts_full_payload` / `llm2tts_batch` 不存在 → 全部落空。
  - **结果:`_load_custom_func` 返回 `(None, None)`,stage1 没有任何"构造下游 payload"的钩子。**

### 14.3 None 之后发生了什么 → 卡死

`send_full_payload_outputs`(`mixin.py:975`):

```python
payload = raw_output
if self._custom_process_func is not None:        # ← sync 下为 None
    payload = self._build_custom_process_payload(...)
    if payload is None:
        continue
if payload is None:
    continue
# 否则直接把 raw_output 发往下一阶段
```

即 sync 模式下,**talker 的"原始 accumulated output"被原样 `connector.put` 给 code2wav,
完全丢失了 `tts2code2wav` 那层 codec 分帧 + `finished_tensor` 终止携带**。
code2wav 拿不到带 `finished` 终止信号的规整 codec 流 → 永远等下一个 chunk → 请求不终结 →
客户端流超时。

这与实测完美对应:
- `[DIAG-5437][TTS]` 能打印(stage1 的 `make_omni_output` 照常产出 `meta.finished=True`)。
- `[DIAG-5437][bridge]` 永不打印 —— 它在 `tts2code2wav_async_chunk` 内,而 sync 路径根本不加载该函数。
- `[DIAG-5437][stage2-recv]` 永不打印 —— 它在 `if self._async_chunk:` 分支(`mixin.py:1790`),
  sync 路径的执行分支里压根没有这段 is_finished 逻辑。
- 客户端最终 `Stream processing error: timed out`。

### 14.4 结论:这条用例的 `--no-async-chunk` 是个错误接线

- miniCPM-o 4.5 的 stage1→stage2 跳**只在 async 下有桥**(`tts2code2wav_async_chunk`);
  sync 路径没有任何对应的 connector payload builder。换句话说,**该模型不被 sync 路径支持**。
- 部署 yaml(`async_chunk: true`)+ pipeline(async 专属 producer)都已指明:miniCPM-o 4.5
  **应当 async 运行**。测试 docstring 里 "async_chunk: false" 与代码现实冲突,是过时描述。
- 因此:
  - 之前对 `tts2code2wav_async_chunk` 加的 `_producer_finished` 修复,位置是对的(async 路径),
    但对**这个用例(走 sync)没有作用**——它根本不会被调用。那处改动可作为 async 下 mix 输入的
    健壮性改进保留,但**不是本用例超时的修复点**。
  - 本用例超时的真实修复杠杆是 **`--no-async-chunk` 这个 flag**。

### 14.5 两个修复选项

**选项 A(推荐,最小且契合既有接线):**
从 `test_minicpmo_4_5.py` 的 `test_params` 里删除 `"--no-async-chunk"`,让用例走
`async_chunk=True`(与 yaml、pipeline、以及用户在 NPU 上的实证一致)。如要一并复活该用例,
再把 `@pytest.mark.skip(reason="#5437")` 摘掉。

**选项 B(让 sync 也能跑,改动大):**
为 miniCPM-o 4.5 的 talker→code2wav 实现一个 sync 版 connector payload builder
(如新增 `tts2code2wav_full_payload`,或在 pipeline 给 stage1 设 `sync_process_input_func`),
把 `tts2code2wav_async_chunk` 里的 codec 分帧 + `finished_tensor` 终止逻辑平移到 whole-payload 场景。
**仅在确有硬性"必须 sync"需求时才值得做**(目前没有任何证据表明 NPU CI 必须 sync;
yaml 默认就是 async)。

> 建议先走选项 A 验证用例能不能过;若确有 sync 需求再补选项 B。

