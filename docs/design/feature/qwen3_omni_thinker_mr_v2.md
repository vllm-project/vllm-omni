# 设计文档：Qwen3-Omni Thinker 迁移至 Model Runner V2

**作者：**（基于 RFC #1770 决策自动生成）
**状态：** 草案
**相关文档：** [RFC #1770 — 迁移至 Model Runner V2](https://github.com/vllm-project/vllm-omni/issues/1770)、[RFC #1595 — 解耦模型特定逻辑](https://github.com/vllm-project/vllm-omni/issues/1595)
**范围：** Qwen3-Omni-30B-A3B-Instruct **仅 Thinker 阶段** 迁移至 MR V2
**目标时间线：** 2026 年 5 月设为默认

---

## 1. 目标

将 Qwen3-Omni Thinker（AR 文本生成阶段）从 v1 的 `OmniGPUModelRunner → GPUARModelRunner` 继承链迁移至 v2 的 `GPUModelRunner` + `ModelState` 架构。Thinker 是首个落地 MR V2 的模型，将作为后续 Talker、TTS、Code2Wav 等模型迁移的模板。

### 非目标（本文档不涉及）

- Talker / TTS / Code2Wav / Diffusion runner 迁移（后续阶段）。
- 修改上游 `GPUModelRunner` 或 `ModelState` 接口。
- NPU / XPU 平台适配。

### 约束条件

- **不修改上游 v2 代码。** 所有扩展通过继承、组合和 `init_model_state()` 注册机制实现。
- Talker、TTS、Code2Wav 必须在同一抽象层上保持可行性。设计决策不能为后续阶段制造障碍。
- 过渡期间 v1 和 v2 runner 通过 `VLLM_OMNI_USE_V2_RUNNER` 环境变量共存。

---

## 2. 背景：当前 v1 架构（Thinker 路径）

### 2.1 类继承层次

```
vllm.v1.worker.gpu_model_runner.GPUModelRunner        (v1, ~6283 行)
  └── OmniGPUModelRunner                               (~1403 行)
        └── GPUARModelRunner                            (~677 行)
```

### 2.2 Thinker 阶段实际需要的功能

Qwen3-Omni Thinker 是一个多模态编码器-解码器（视觉/音频 Transformer + MoE LLM）。在 v1 runner 中，Thinker 路径仅使用了 `OmniGPUModelRunner` 功能的**子集**：

| 功能 | Thinker 是否使用？ | 说明 |
|------|:-:|------|
| `model_intermediate_buffer` | 部分 | 在请求进入时填充（prompt_embeds、additional_information）；每步以 `runtime_additional_information` 形式聚合——但 **Thinker 的 `forward()` 会忽略它**（被 `**kwargs` 吸收）。该缓冲区的存在是为了下游的 Talker/Code2Wav 阶段。 |
| `prompt_embeds` 覆盖 | 否 | Thinker 的 `has_preprocess = False`；`prompt_embeds_cpu` 会被解码并存储，但不会进行覆盖。 |
| M-RoPE 位置编码 | **是** | Thinker 实现了 `SupportsMRoPE` 接口；`get_mrope_input_positions()` 从 `image_grid_thw`、`video_grid_thw`、`audio_feature_lengths` 计算 3D 位置。 |
| `OmniOutput` 包装 | **是** | `make_omni_output()` 将 Thinker 输出 `(text_hidden_states, captured_layer_dict)` 包装为 `OmniOutput`，附带多模态侧载信息（TTS embed BOS/EOS/PAD）。 |
| `extract_multimodal_outputs` | **是** | 将 `OmniOutput` 拆分为 `text_hidden_states` + `multimodal_outputs` 字典。 |
| `has_preprocess` / 自定义预处理 | 否 | Thinker：`has_preprocess = False`。 |
| `has_postprocess` / 自定义后处理 | 否 | Thinker：`has_postprocess = False`。 |
| `talker_mtp` 子模块 | 否 | Thinker：`self.talker = None`，无 talker_mtp。 |
| `pooler_output`（hidden + mm） | **是** | AR runner 将每个请求的 hidden states + 多模态输出切片后组装为 `pooler_output`，供下游阶段使用。 |
| KV 传输 | **是** | `OmniKVTransferManager` 为已完成请求提取 KV 缓存。 |
| Deepstack 视觉 | **是** | Thinker 模型 `forward()` 内部处理，对 runner 透明。 |

**关键洞察：** Thinker 是最简单的 Omni 模型阶段。它不使用 `preprocess`/`postprocess`、`talker_mtp` 或 `prompt_embeds` 覆盖。它**需要**的是 M-RoPE、`OmniOutput` 处理、`pooler_output` 构建和 KV 传输。

---

## 3. 设计概览

### 3.1 目标架构

```
vllm.v1.worker.gpu.model_runner.GPUModelRunner      (v2, ~1168 行, 不修改)
  └── OmniGPUModelRunner                              (薄继承层, ~150 行)
        └── OmniARModelRunner                          (AR 专用, ~350 行)

vllm.v1.worker.gpu.model_states.interface.ModelState  (v2 接口, 不修改)
  └── DefaultModelState                                (v2 默认实现)
        └── OmniModelState                             (通用 Omni 基类, ~300 行)

init_model_state()  ← 通过架构名检查扩展  (v2 工厂函数, 新增 ~5 行)

OmniIntermediateBuffer                                 (新增, ~120 行)
OmniModelStatePlugin                                   (新增, ABC, ~30 行)
PositionEncodingStrategy                               (新增, ABC + 4 种实现, ~200 行)
OmniKVConnector                                        (新增, 封装 OmniKVTransferManager, ~80 行)
```

### 3.2 设计决策（来自 RFC #1770 开放问题）

| # | 决策 | 理由 |
|---|------|------|
| 1 | 通过 `init_model_state()` 工厂函数注册 `OmniModelState` | 比 runner 侧注入更清晰；`init_model_state()` 已有基于架构名的分发逻辑（Whisper），将 Omni 架构加入同一模式。 |
| 2 | 对大型中间张量使用 `UvaBackedTensor` | 减少 `prompt_embeds` 和捕获的 hidden states 的显式 CPU↔GPU 拷贝。与 v2 的内存管理方式一致。 |
| 3 | Generation runner 复用 v2 的 `pool()` 代码路径 | Generation 阶段无需采样；`pool()` 已绕过 `sample_tokens`。适配 `pool()` 的返回语义以匹配 Generation 输出。 |
| 4 | V2 runner 于 2026 年 5 月成为默认 | 功能开关 `VLLM_OMNI_USE_V2_RUNNER` 翻转默认值。V1 保留一个发布周期后移除。 |

---

## 4. 详细设计

### 4.1 `init_model_state()` 扩展

v2 工厂函数位于 `vllm/v1/worker/gpu/model_states/__init__.py`，当前按架构名分发。我们在 `vllm_omni` 中添加 Omni 感知的包装函数来**扩展**（而非修改）它：

```python
# vllm_omni/worker_v2/model_states/__init__.py

from vllm.v1.worker.gpu.model_states import init_model_state as _upstream_init

# 需要使用 OmniModelState 的架构列表
_OMNI_ARCHITECTURES = {
    "Qwen3OmniMoeForConditionalGeneration",
    "MammothModa2ForConditionalGeneration",
    "MiMoAudioForConditionalGeneration",
    # 后续模型在此添加
}

def init_omni_model_state(vllm_config, model, encoder_cache, device):
    """扩展 init_model_state，增加 Omni 架构分发。"""
    archs = set(vllm_config.model_config.architectures or [])
    if archs & _OMNI_ARCHITECTURES:
        from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
        return OmniModelState(vllm_config, model, encoder_cache, device)
    return _upstream_init(vllm_config, model, encoder_cache, device)
```

`OmniGPUModelRunner` 通过覆写 `load_model` 使用 `init_omni_model_state` 替代上游工厂函数。这**不需要任何上游代码修改**。

### 4.2 `OmniModelState` — 通用基类

`OmniModelState` 继承 `DefaultModelState`。它与模型无关：不包含 Qwen3 特定代码，也不包含 Talker 逻辑。它处理**所有** Omni 模型共享的横切关注点。

```python
class OmniModelState(DefaultModelState):
    """通用 Omni ModelState — 适用于所有 Omni 模型阶段。

    职责：
    - 跨阶段中间缓冲区（prompt_embeds、additional_information）
    - OmniOutput → text_hidden + multimodal_outputs 提取
    - model_intermediate_buffer / runtime_additional_information 注入
    - 插件生命周期分发
    """

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        max_num_reqs = self.scheduler_config.max_num_seqs
        self.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)
        self.has_preprocess = getattr(model, "has_preprocess", False)
        self.has_postprocess = getattr(model, "has_postprocess", False)
        self.have_multimodal_outputs = getattr(model, "have_multimodal_outputs", False)
        self.plugins: list[OmniModelStatePlugin] = []

        # 允许模型在初始化时注册插件
        if hasattr(model, "get_omni_plugins"):
            for plugin in model.get_omni_plugins():
                self.register_plugin(plugin)

    def register_plugin(self, plugin: "OmniModelStatePlugin") -> None:
        self.plugins.append(plugin)

    # --- 请求生命周期 ---

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.intermediate_buffer.add_request(req_index, new_req_data)
        for plugin in self.plugins:
            plugin.on_add_request(req_index, new_req_data)

    def remove_request(self, req_index: int) -> None:
        self.intermediate_buffer.remove_request(req_index)
        for plugin in self.plugins:
            plugin.on_remove_request(req_index)

    # --- 输入准备 ---

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        base = super().prepare_inputs(input_batch, req_states)

        # 将中间缓冲区作为模型 kwargs 注入
        # v2 通过 **展开 将此字典合并到 model_inputs 中
        buffer_list = self.intermediate_buffer.gather(input_batch)
        base["model_intermediate_buffer"] = buffer_list
        base["runtime_additional_information"] = buffer_list  # 向后兼容

        # 插件提供的额外输入
        for plugin in self.plugins:
            base.update(plugin.prepare_extra_inputs(input_batch, req_states))

        return base

    # --- 输出处理 ---

    def postprocess_model_output(self, model_output, input_batch, req_states):
        """将原始模型输出转换为 (text_hidden, multimodal_outputs)。

        处理 OmniOutput 解包和 make_omni_output 转换。
        由 OmniARModelRunner.sample_tokens() 调用。
        """
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            buffer_list = self.intermediate_buffer.gather(input_batch)
            model_output = self.model.make_omni_output(
                model_output,
                model_intermediate_buffer=buffer_list,
                runtime_additional_information=buffer_list,
            )

        if self.have_multimodal_outputs and isinstance(model_output, OmniOutput):
            text_hidden = model_output.text_hidden_states
            multimodal_outputs = model_output.multimodal_outputs or {}
        elif isinstance(model_output, (list, tuple)):
            text_hidden = model_output[0]
            multimodal_outputs = {}
        else:
            text_hidden = model_output
            multimodal_outputs = {}

        # 运行插件（例如 Talker 阶段的 TalkerMTPPlugin —— Thinker 不涉及）
        for plugin in self.plugins:
            text_hidden, multimodal_outputs = plugin.postprocess(
                text_hidden, multimodal_outputs, input_batch, req_states
            )

        return text_hidden, multimodal_outputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens) -> dict[str, Any]:
        base = super().prepare_dummy_inputs(num_reqs, num_tokens)
        base["model_intermediate_buffer"] = [{} for _ in range(num_reqs)]
        base["runtime_additional_information"] = base["model_intermediate_buffer"]
        return base
```

#### 4.2.1 对 Thinker 的影响

对于 Thinker 阶段（`has_preprocess=False`、`has_postprocess=False`、无插件）：

- `add_request` → 将 `prompt_embeds` 和 `additional_information` 存入 `intermediate_buffer`（供下游使用）；通过 `super().add_request()` 调用 `rope_state.init_prefill_positions()` 初始化 M-RoPE。
- `prepare_inputs` → 返回 `{"positions": mrope_positions, "model_intermediate_buffer": [...], "runtime_additional_information": [...]}`。Thinker 的 `forward()` 通过 `**kwargs` 静默吸收这些参数。
- `postprocess_model_output` → 通过 `make_omni_output()` 将 `(text_hidden_states, captured_layer_dict)` 转换为 `OmniOutput`，然后提取 `text_hidden_states` + `multimodal_outputs`（捕获的层、TTS embeds）。
- 无插件执行。

### 4.3 `OmniIntermediateBuffer`

基于索引设计，与 v2 的 `RequestState` 槽位管理对齐。对 `prompt_embeds` 使用 `UvaBackedTensor`。

```python
class OmniIntermediateBuffer:
    """多阶段流水线的每请求中间状态。

    使用 req_index（而非 req_id）进行 O(1) 访问，与 v2 RequestState 对齐。
    """

    def __init__(self, max_num_reqs: int):
        self.buffers: list[dict[str, Any]] = [{} for _ in range(max_num_reqs)]

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        """解码并存储 prompt_embeds 和 additional_information。"""
        info: dict[str, Any] = {}

        # 解码 prompt_embeds（序列化张量载荷）
        pe = getattr(new_req_data, "prompt_embeds", None)
        if pe is not None:
            pe_cpu = _resolve_prompt_embeds(pe)
            if pe_cpu is not None:
                info["prompt_embeds_cpu"] = pe_cpu

        # 解码 additional_information（字典或旧版载荷）
        ai = getattr(new_req_data, "additional_information", None)
        if ai is not None:
            info.update(_resolve_additional_information(ai))

        # 存储 mm_features 引用，供需要的模型使用（如 MiMoAudio）
        if new_req_data.mm_features:
            info["mm_features"] = new_req_data.mm_features

        # 存储 req_id，供需要的模型使用
        info["req_id"] = new_req_data.req_id

        self.buffers[req_index] = info

    def remove_request(self, req_index: int) -> None:
        self.buffers[req_index] = {}

    def gather(self, input_batch: InputBatch) -> list[dict]:
        """按批次顺序聚合中间信息，使用 idx_mapping。"""
        return [self.buffers[idx] for idx in input_batch.idx_mapping_np]

    def update(self, req_index: int, updates: dict) -> None:
        """合并更新。张量自动 detach；GPU 常驻键保留在 GPU 上。"""
        existing = self.buffers[req_index]
        for k, v in updates.items():
            if isinstance(v, torch.Tensor):
                existing[k] = v.detach().clone()
            else:
                existing[k] = v
```

#### 4.3.1 Thinker 的缓冲区生命周期

```
 请求到达
     │
     ▼
 add_request(req_index, new_req_data)
     ├── 解码 prompt_embeds → info["prompt_embeds_cpu"]
     ├── 解码 additional_information → info[...]
     ├── info["mm_features"] = new_req_data.mm_features
     └── info["req_id"] = new_req_data.req_id
     │
     ▼  （每步执行）
 gather(input_batch) → list[dict]  →  model_intermediate_buffer kwarg
     │                                    （Thinker 通过 **kwargs 忽略）
     ▼
 请求完成
     │
     ▼
 remove_request(req_index) → buffers[i] = {}
```

对于 Thinker，缓冲区虽被填充但**模型不消费**。它的存在是为了下游的 Talker/Code2Wav 阶段能够通过跨阶段 KV 传输元数据接收数据。

### 4.4 `OmniModelStatePlugin` — 扩展点

模型特定逻辑的抽象接口。**Thinker 不注册任何插件。** 此接口为 Talker（`TalkerMTPPlugin`）和未来模型而设。

```python
class OmniModelStatePlugin(ABC):
    """OmniModelState 中模型特定行为的扩展点。

    生命周期：
      on_add_request()       — 新请求进入批次
      on_remove_request()    — 请求完成
      prepare_extra_inputs() — 向 model_inputs 注入额外 kwargs
      postprocess()          — 处理模型输出（如 talker_mtp 前向传播）
      dummy_run()            — 内存分析 / CUDA 图捕获
    """

    @abstractmethod
    def on_add_request(self, req_index, new_req_data) -> None: ...

    @abstractmethod
    def on_remove_request(self, req_index) -> None: ...

    def prepare_extra_inputs(self, input_batch, req_states) -> dict:
        return {}

    @abstractmethod
    def postprocess(self, text_hidden, mm_outputs, input_batch, req_states):
        return text_hidden, mm_outputs

    def dummy_run(self, state, num_tokens) -> None:
        pass
```

未来 Talker 迁移将实现 `TalkerMTPPlugin`：
- `postprocess()` → 运行 `talker_mtp` 前向传播，将 `code_predictor_codes` 写入缓冲区
- `prepare_extra_inputs()` → 注入预处理结果
- `dummy_run()` → 捕获 talker CUDAGraph

### 4.5 `OmniGPUModelRunner` — 薄继承层

对 v2 `GPUModelRunner` 的最小扩展。处理 `ModelState` 自身无法覆盖的 Omni 特定生命周期钩子。

```python
class OmniGPUModelRunner(GPUModelRunner):
    """v2 GPUModelRunner 上的薄层，用于 Omni 生命周期钩子。"""

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        # 用 Omni 感知版本替换 model_state
        self.model_state = init_omni_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )

    def finish_requests(self, scheduler_output):
        # 在上游移除请求索引之前清理中间缓冲区
        for req_id in scheduler_output.finished_req_ids | scheduler_output.preempted_req_ids:
            if req_id in self.req_states.req_id_to_index:
                idx = self.req_states.req_id_to_index[req_id]
                self.model_state.remove_request(idx)
        super().finish_requests(scheduler_output)
```

**为何覆写 `finish_requests`？** v2 `ModelState` 接口没有 `remove_request()`。我们在 `OmniModelState` 中添加它并从 runner 调用。这是我们需要的唯一 runner 级生命周期钩子。如果上游后续将 `remove_request()` 加入 `ModelState`，我们可以移除此覆写。

### 4.6 `OmniARModelRunner` — Thinker 的 Runner

扩展 `OmniGPUModelRunner`，添加 AR 特定行为：`OmniOutput` 后处理、每请求 `pooler_output` 构建和 KV 提取。

```python
class OmniARModelRunner(OmniGPUModelRunner):
    """AR 阶段 runner。生成每请求的 hidden states + 多模态输出。"""

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(
            self.vllm_config, self.model_config
        )

    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None,
                      dummy_run=False, skip_attn_for_dummy_run=False):
        # 在状态清理之前处理已完成请求的 KV 传输
        self._handle_kv_transfer_pre(scheduler_output)

        # 委托给 v2 GPUModelRunner.execute_model（不修改）
        result = super().execute_model(
            scheduler_output, intermediate_tensors,
            dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn_for_dummy_run,
        )
        return result

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        if self.execute_model_state is None:
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        kv_connector_output = self.execute_model_state.kv_connector_output
        num_tokens_across_dp = self.execute_model_state.num_tokens_across_dp
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            sampled, num_sampled, num_rejected = pp_receive(...)
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)
            return None

        # --- Omni 特有：后处理模型输出 ---
        text_hidden, multimodal_outputs = self.model_state.postprocess_model_output(
            hidden_states, input_batch, self.req_states
        )

        # --- 标准 v2 采样 ---
        sampler_output, num_sampled, num_rejected = self.sample(
            text_hidden, input_batch, grammar_output
        )

        # --- Omni 特有：构建 pooler_output ---
        pooler_output = self._build_pooler_output(
            text_hidden, multimodal_outputs, input_batch
        )

        # --- 后处理（Triton kernel 更新 req_states）---
        self.postprocess(
            input_batch, sampler_output.sampled_token_ids,
            num_sampled, num_rejected
        )

        output = OmniModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(input_batch.req_ids)},
            sampled_token_ids=...,
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            kv_extracted_req_ids=self._kv_extracted_req_ids,
        )

        if self.use_async_scheduling:
            return AsyncOutput(model_runner_output=output, ...)
        return output

    def _build_pooler_output(self, text_hidden, mm_outputs, input_batch):
        """按请求切分 hidden states + 多模态输出。"""
        hidden_cpu = text_hidden.detach().cpu().contiguous()
        mm_cpu = self._copy_mm_to_cpu(mm_outputs, hidden_cpu.shape[0])
        pooler = []
        for i, req_id in enumerate(input_batch.req_ids):
            start = int(input_batch.query_start_loc_np[i])
            end = start + int(input_batch.num_scheduled_tokens[i])
            payload = {"hidden": hidden_cpu[start:end]}
            payload.update(self._slice_mm_cpu(mm_cpu, start, end, i))
            pooler.append(payload)
        return pooler

    def _handle_kv_transfer_pre(self, scheduler_output):
        """在状态清理前为已完成请求提取 KV 缓存。"""
        finished = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished.items():
                meta = self.model.get_kv_transfer_metadata(req_id)
                if meta:
                    existing = data.get("custom_metadata") or {}
                    existing.update(meta)
                    data["custom_metadata"] = existing
        self._kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
        )
```

### 4.7 M-RoPE 集成

v2 的 `DefaultModelState` 已通过 `rope_state` 处理 M-RoPE：

```python
# DefaultModelState.add_request()
if self.rope_state is not None:
    self.rope_state.init_prefill_positions(
        req_index, self.model, new_req_data.prefill_token_ids,
        mm_features=new_req_data.mm_features,
    )

# DefaultModelState.prepare_inputs()
positions = self.rope_state.get_positions(num_tokens_after_padding)
return {"positions": positions}
```

这会委托给模型的 `get_mrope_input_positions()` —— Qwen3-Omni Thinker 已经实现了该方法。**M-RoPE 无需额外工作。** v2 的 `rope_state` 处理：

- 带多模态特征的 Prefill 位置初始化
- Decode 位置更新（线性递增 + position delta）
- GPU 缓冲区的 Staged writes 管理

v1 的 `_init_mrope_positions` / `_calc_mrope_positions` / `_fixup_precomputed_mrope_decode_positions` 逻辑对于 Thinker 而言**完全被 `rope_state` 取代**。（GLM-Image 的预计算 decode 修正可能需要在未来阶段通过 `PositionEncodingStrategy` 覆写来实现。）

### 4.8 `model_inputs` 注入流程

关键的 kwargs 注入通过 v2 现有的展开模式工作：

```python
# v2 GPUModelRunner.execute_model() 第 991-998 行（不修改）
model_inputs = {
    "input_ids": input_batch.input_ids,
    "positions": input_batch.positions,
    "inputs_embeds": inputs_embeds,
    "intermediate_tensors": intermediate_tensors,
    **self.model_state.prepare_inputs(input_batch, self.req_states),
}
```

`OmniModelState.prepare_inputs()` 返回：

```python
{
    "positions": mrope_3d_positions,              # 来自 rope_state
    "model_intermediate_buffer": [dict, dict, ...], # 每请求
    "runtime_additional_information": [dict, ...],  # 向后兼容别名
}
```

这些键展开到 `model_inputs` 中，并传递给 `self.model(**model_inputs)`。Qwen3-Omni 统一的 `forward()` 接受 `runtime_additional_information` 作为显式参数，`**kwargs` 吸收其余部分：

```python
# Qwen3OmniMoeForConditionalGeneration.forward()
def forward(self, input_ids, positions, ...,
            runtime_additional_information=None, **kwargs):
    if self.model_stage == "thinker":
        # 不使用 runtime_additional_information 或 model_intermediate_buffer
        text_hidden_states, captured_layer_dict = self.thinker(
            input_ids=input_ids, positions=positions,
            inputs_embeds=inputs_embeds, **capture_kwargs, **kwargs,
        )
        return text_hidden_states, captured_layer_dict
```

**Thinker 静默吸收额外的 kwargs。** 无需修改模型代码。

### 4.9 `OmniOutput` 处理流水线

```
model(**model_inputs)
  │
  ▼  返回 (text_hidden_states, captured_layer_dict) — 原始元组，不是 OmniOutput
  │
  ▼  GPUModelRunner 将其作为 hidden_states 存入 ExecuteModelState
  │
  ▼  OmniARModelRunner.sample_tokens()
  │
  ▼  model_state.postprocess_model_output(hidden_states, ...)
  │     ├── model.make_omni_output(raw_output, **buffer_kwargs)
  │     │     └── 返回 OmniOutput(text_hidden_states=..., multimodal_outputs={
  │     │           captured layers, tts_bos_embed, tts_eos_embed, tts_pad_embed
  │     │         })
  │     ├── 提取: text_hidden = omni_output.text_hidden_states
  │     └── 提取: mm_outputs = omni_output.multimodal_outputs
  │
  ▼  self.sample(text_hidden, input_batch, grammar_output)
  │     └── compute_logits → sample → sampler_output
  │
  ▼  _build_pooler_output(text_hidden, mm_outputs, input_batch)
  │     └── 每请求: {hidden: slice, **mm_slice}
  │
  ▼  OmniModelRunnerOutput(pooler_output=pooler, ...)
```

**重要边界情况：** v2 的 `execute_model` 期望最后一个 PP rank 的 `hidden_states` 为 `torch.Tensor`。但 Thinker 返回的是 `(Tensor, dict)` 元组。两种方案：

1. **优选方案：** `OmniModelState.prepare_inputs()` 返回一个标志，使模型直接输出 `OmniOutput`（在 `forward` 内部调用 `make_omni_output`）。
2. **兜底方案：** 在 `OmniGPUModelRunner` 中覆写 `execute_model` 拦截模型输出并包装。（约 10 行）

我们采用方案 2 以确保安全，在 `self.model(**model_inputs)` 之后拦截：

```python
# 在 OmniGPUModelRunner 中，仅覆写 execute_model 以包装输出
def execute_model(self, scheduler_output, ...):
    result = super().execute_model(scheduler_output, ...)
    # 拦截：如果 execute_model_state 的 hidden_states 不是 Tensor，
    # 说明模型返回了元组。进行包装和缓存。
    if self.execute_model_state is not None:
        hs = self.execute_model_state.hidden_states
        if not isinstance(hs, torch.Tensor) and hasattr(self.model, "make_omni_output"):
            buffer_list = self.model_state.intermediate_buffer.gather(
                self.execute_model_state.input_batch
            )
            omni_output = self.model.make_omni_output(
                hs, model_intermediate_buffer=buffer_list,
                runtime_additional_information=buffer_list,
            )
            # 用 OmniOutput 替换 hidden_states，供 sample_tokens 使用
            self.execute_model_state = self.execute_model_state._replace(
                hidden_states=omni_output
            )
    return result
```

### 4.10 功能开关共存

```python
# vllm_omni/worker/omni_ar_worker.py
import os

class OmniARWorker:
    def __init__(self, vllm_config, device):
        use_v2 = os.environ.get("VLLM_OMNI_USE_V2_RUNNER", "0") == "1"
        if use_v2:
            from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
            self.model_runner = OmniARModelRunner(vllm_config, device)
        else:
            from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
            self.model_runner = GPUARModelRunner(vllm_config, device)
```

时间线：
1. **现在 → 4 月：** `VLLM_OMNI_USE_V2_RUNNER=1` 选择性启用
2. **2026 年 5 月：** 翻转默认值为 `1`（v1 可通过 `=0` 使用）
3. **2026 年 6 月/7 月：** 移除 v1 runner

---

## 5. 数据流：Thinker 完整路径

```
SchedulerOutput
     │
     ▼
┌─ OmniARModelRunner.execute_model() ─────────────────────────────────┐
│  1. _handle_kv_transfer_pre(scheduler_output)                        │
│     └── 为已完成请求提取 KV 缓存                                      │
│  2. super().execute_model() → v2 GPUModelRunner.execute_model()      │
│     ├── finish_requests()                                            │
│     │   └── model_state.remove_request() → 清理缓冲区                 │
│     ├── free_states()                                                │
│     ├── add_requests()                                               │
│     │   └── model_state.add_request()                                │
│     │       ├── rope_state.init_prefill_positions() → M-RoPE 初始化   │
│     │       └── intermediate_buffer.add_request()                    │
│     ├── update_requests() → block_tables                             │
│     ├── prepare_inputs() → InputBatch [Triton kernels]               │
│     ├── prepare_attn() → block_tables, slot_mappings                 │
│     ├── model_state.prepare_attn() → attn_metadata                   │
│     ├── model_state.get_mm_embeddings() → 视觉/音频编码               │
│     ├── model_state.prepare_inputs() → {                             │
│     │     positions (M-RoPE 3D),                                     │
│     │     model_intermediate_buffer,                                 │
│     │     runtime_additional_information                             │
│     │   }                                                            │
│     ├── model(**model_inputs) → (text_hidden, captured_layers)       │
│     └── ExecuteModelState 已存储                                      │
│  3. 后拦截: 包装输出 → 通过 make_omni_output 生成 OmniOutput           │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ OmniARModelRunner.sample_tokens() ─────────────────────────────────┐
│  1. model_state.postprocess_model_output()                           │
│     ├── OmniOutput.text_hidden_states → text_hidden                  │
│     └── OmniOutput.multimodal_outputs → {captured_layers, tts_*}     │
│  2. self.sample(text_hidden, ...) → sampled_token_ids [Triton]       │
│  3. _build_pooler_output() → 每请求 hidden + mm 切片                  │
│  4. self.postprocess() → Triton kernel 更新 req_states               │
│  5. → OmniModelRunnerOutput                                         │
│     {sampled_token_ids, pooler_output, kv_extracted_req_ids}         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. 与 v1 的对比变化

| 方面 | v1（当前） | v2（本设计） |
|------|-----------|-------------|
| 基类 | `GPUModelRunner` v1（~6283 行） | `GPUModelRunner` v2（~1168 行） |
| `OmniGPUModelRunner` | ~1403 行，覆写 `_update_states`、`_preprocess`、`_model_forward`、`_dummy_run` | ~150 行，覆写 `load_model`、`finish_requests`、`execute_model`（10 行拦截） |
| `GPUARModelRunner` | ~677 行，覆写 `execute_model`、`sample_tokens` | `OmniARModelRunner` ~350 行，覆写 `execute_model`（前置钩子 + super）、`sample_tokens` |
| `model_intermediate_buffer` | `dict[str, dict]` 以 `req_id` 为键 | `list[dict]` 以 `req_index` 索引，通过 `OmniIntermediateBuffer` 管理 |
| M-RoPE | 自定义 `_init_mrope_positions`、`_calc_mrope_positions`、修正逻辑 | 完全委托给 v2 `rope_state` |
| `OmniOutput` 包装 | `_model_forward` 调用 `make_omni_output`，缓存于 `_omni_last_model_output` | `execute_model` 中后拦截，然后在 `sample_tokens` 中通过 `postprocess_model_output` 处理 |
| 输入准备 | Python 密集型 `_preprocess`（~180 行） | v2 Triton kernels + `ModelState.prepare_inputs()` |
| kwargs 注入 | `_model_forward` → `_build_model_kwargs_extra` | `prepare_inputs()` 返回字典，展开到 `model_inputs` |
| 请求清理 | `_update_states` 从字典中弹出 | `finish_requests` → `model_state.remove_request(idx)` |
| Omni runner 总行数 | ~2080（OmniGPU + AR） | ~500（OmniGPU + AR）— **精简 76%** |

---

## 7. 健壮性：未来阶段兼容性

本设计已针对所有已知 Omni 模型阶段的需求进行验证：

### 7.1 Talker（未来阶段）

- **`has_preprocess = True`**：`OmniModelState.prepare_inputs()` 已包含 `model_intermediate_buffer`。Talker 的 `preprocess()` 在 `TalkerMTPPlugin.prepare_extra_inputs()` 钩子内按请求执行。
- **`talker_mtp` 子模块**：`TalkerMTPPlugin.postprocess()` 在 `sample_tokens` 中运行 talker_mtp 前向传播。CUDAGraph 包装独立管理（与 v1 相同）。
- **`prompt_embeds` 覆盖**：`OmniModelState` 可在 `prepare_inputs()` 中实现覆盖，或通过专用的 `PromptEmbedsPlugin` 实现。
- **`gpu_resident_buffer_keys`**：`OmniIntermediateBuffer.update()` 遵循每模型的 GPU 常驻键设置。

### 7.2 TTS（未来阶段）

- 与 Talker 相同，但 `TalkerMTPPlugin` 禁用 CUDAGraph（模型有内部 AR 循环）。
- 通过 `TTSOmniModelState` 子类或插件上的配置标志实现。

### 7.3 Code2Wav / Generation 阶段（未来阶段）

- 使用 `pool()` 路径（决策 #3）：`OmniGenerationModelRunner` 覆写 `pool()` 而非 `sample_tokens()`。
- `runtime_additional_information` 用于获取 `left_context_size` — 已由 `prepare_inputs()` 注入。
- `seq_token_counts` 通过插件 `prepare_extra_inputs()` 注入。

### 7.4 MiMoAudio、MammothModa2、GLM-Image（未来阶段）

- **MiMoAudio**：`mm_features` 和 `req_id` 始终存在于缓冲区中（来自 RFC #1595 的泛化）。
- **MammothModa2**：网格约束的 `generated_len` — 可在 `prepare_inputs()` 中从 `req_states` 计算。
- **GLM-Image**：预计算的 M-RoPE decode 位置 — 需要 `PositionEncodingStrategy` 覆写（Thinker 不需要）。

---

## 8. 文件结构

```
vllm_omni/worker_v2/
├── __init__.py
├── omni_model_runner.py                 # OmniGPUModelRunner (~150 行)
├── omni_ar_model_runner.py              # OmniARModelRunner (~350 行)
├── model_states/
│   ├── __init__.py                      # init_omni_model_state() 工厂函数
│   ├── omni_model_state.py              # OmniModelState (通用基类, ~300 行)
│   ├── intermediate_buffer.py           # OmniIntermediateBuffer (~120 行)
│   ├── plugin.py                        # OmniModelStatePlugin (ABC, ~30 行)
│   └── models/                          # (当前为空, 未来: talker.py, tts.py)
│       └── __init__.py
└── outputs.py                           # 重导出 OmniModelRunnerOutput
```

过渡期间与 `vllm_omni/worker/` 共存。

---

## 9. 迁移检查清单

- [ ] **Phase 0：验证**（1 周）
  - [ ] 确认 v2 `rope_state` 正确处理 Qwen3-Omni M-RoPE
  - [ ] 确认 v2 `DefaultModelState.get_mm_embeddings()` 对 Qwen3 视觉/音频塔正常工作
  - [ ] 确认 `prepare_inputs()` 的 kwargs 无过滤地传递到 `model.forward()`
  - [ ] 确认 `ExecuteModelState` 能容纳非 Tensor 的 `hidden_states`（元组）

- [ ] **Phase 1：核心状态**（2 周）
  - [ ] 实现 `OmniIntermediateBuffer`
  - [ ] 实现 `OmniModelStatePlugin`（ABC）
  - [ ] 实现 `OmniModelState`（继承 `DefaultModelState`）
  - [ ] 实现 `init_omni_model_state()` 工厂函数
  - [ ] 编写缓冲区 add/remove/gather、state prepare_inputs 的单元测试

- [ ] **Phase 2：AR Runner**（2 周）
  - [ ] 实现 `OmniGPUModelRunner`（薄继承）
  - [ ] 实现 `OmniARModelRunner`（Thinker 路径）
  - [ ] 在 `OmniARWorker` 中接入功能开关
  - [ ] 集成测试：纯文本生成
  - [ ] 集成测试：多模态输入（图像、音频、视频）
  - [ ] 集成测试：M-RoPE 位置与 v1 一致
  - [ ] 集成测试：`pooler_output` 内容与 v1 一致
  - [ ] 集成测试：KV 传输

- [ ] **Phase 3：验收**（1 周）
  - [ ] 端到端测试：Qwen3-Omni-30B-A3B-Instruct Thinker
  - [ ] 性能基准：吞吐量、延迟、GPU 内存 vs. v1
  - [ ] 回归测试：所有现有 Thinker 阶段模型
  - [ ] 推测解码（EAGLE）兼容性检查

---

## 10. 单元测试规划

本节规划 `tests/worker_v2/` 目录下的单元测试，遵循现有 `tests/worker/` 的测试约定：
- 所有测试文件标记 `pytestmark = [pytest.mark.core_model, pytest.mark.cpu]`
- 通过 `object.__new__(RunnerClass)` 跳过真实构造函数，仅填充被测方法所需的最小属性
- 使用 `monkeypatch` 替换模块级符号（如 `set_forward_context`）
- 使用简单的 Dummy/Stub 类替代真实的 `InputBatch`、`RequestState` 等
- 不依赖 GPU、调度器或完整推理管线

### 10.1 文件结构

```
tests/worker_v2/
├── __init__.py
├── test_intermediate_buffer.py          # OmniIntermediateBuffer 单元测试
├── test_omni_model_state.py             # OmniModelState 单元测试
├── test_omni_model_state_plugin.py      # OmniModelStatePlugin 接口测试
├── test_omni_gpu_model_runner.py        # OmniGPUModelRunner 单元测试
├── test_omni_ar_model_runner.py         # OmniARModelRunner 单元测试
└── test_init_model_state.py             # init_omni_model_state 工厂函数测试
```

### 10.2 共用桩对象和工具函数

以下桩对象在多个测试文件中复用，可放置于 `tests/worker_v2/conftest.py` 或各文件内联：

```python
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyInputBatch:
    """最小 InputBatch 桩，提供 idx_mapping_np、req_ids 等。"""
    def __init__(self, req_ids, idx_mapping_np, num_tokens=4,
                 num_tokens_after_padding=4):
        self.req_ids = list(req_ids)
        self.idx_mapping_np = idx_mapping_np
        self.num_reqs = len(req_ids)
        self.num_tokens = num_tokens
        self.num_tokens_after_padding = num_tokens_after_padding
        self.query_start_loc_np = np.zeros(len(req_ids), dtype=np.int32)
        self.num_scheduled_tokens = np.ones(len(req_ids), dtype=np.int32)


class DummyRequestState:
    """最小 RequestState 桩。"""
    def __init__(self):
        self.req_id_to_index = {}


class DummyNewReqData:
    """最小 NewRequestData 桩。"""
    def __init__(self, req_id, prompt_embeds=None,
                 additional_information=None, mm_features=None,
                 prefill_token_ids=None):
        self.req_id = req_id
        self.prompt_embeds = prompt_embeds
        self.additional_information = additional_information
        self.mm_features = mm_features or []
        self.prefill_token_ids = prefill_token_ids or [1, 2, 3]


class DummySchedulerOutput:
    """最小 SchedulerOutput 桩。"""
    def __init__(self, finished_req_ids=None, preempted_req_ids=None):
        self.finished_req_ids = set(finished_req_ids or [])
        self.preempted_req_ids = set(preempted_req_ids or [])


class DummyModel(torch.nn.Module):
    """可配置的模型桩。"""
    def __init__(self, have_multimodal_outputs=True,
                 has_preprocess=False, has_postprocess=False):
        super().__init__()
        self.have_multimodal_outputs = have_multimodal_outputs
        self.has_preprocess = has_preprocess
        self.has_postprocess = has_postprocess

    def make_omni_output(self, raw, **kwargs):
        from vllm_omni.model_executor.models.output_templates import OmniOutput
        text_hidden, captured = raw
        return OmniOutput(
            text_hidden_states=text_hidden.reshape(-1, text_hidden.shape[-1]),
            multimodal_outputs=captured,
        )
```

### 10.3 `test_intermediate_buffer.py` — OmniIntermediateBuffer

测试中间缓冲区的核心生命周期：添加、移除、聚合、更新。

| 测试函数 | 测试内容 | 验证要点 |
|----------|---------|---------|
| `test_add_request_stores_req_id` | 基本 `add_request` | `buffers[idx]["req_id"] == new_req_data.req_id` |
| `test_add_request_decodes_prompt_embeds` | 带 `prompt_embeds` 的请求 | `buffers[idx]["prompt_embeds_cpu"]` 是 CPU Tensor，形状正确 |
| `test_add_request_decodes_additional_information` | 带 `additional_information` 字典 | 字典键值正确写入 `buffers[idx]` |
| `test_add_request_stores_mm_features` | 带 `mm_features` 的请求 | `buffers[idx]["mm_features"]` 引用与输入一致 |
| `test_add_request_empty_payload` | `prompt_embeds=None`、`additional_information=None` | `buffers[idx]` 仅有 `req_id`，无多余键 |
| `test_remove_request_clears_slot` | `remove_request` 后检查 | `buffers[idx] == {}` |
| `test_remove_request_idempotent` | 连续两次 `remove_request` | 不抛异常，`buffers[idx] == {}` |
| `test_gather_returns_batch_order` | 3 个请求、乱序 `idx_mapping_np` | 返回列表顺序与 `idx_mapping_np` 一致 |
| `test_gather_empty_batch` | 空 `idx_mapping_np` | 返回空列表 |
| `test_gather_after_remove_returns_empty_dict` | 添加后移除再聚合 | 对应位置返回 `{}` |
| `test_update_merges_new_keys` | 调用 `update` 添加新键 | 新旧键共存 |
| `test_update_overwrites_existing_key` | 同键多次 `update` | 值为最后一次写入 |
| `test_update_detaches_tensors` | `update` 带 GPU-like Tensor | 存储的 Tensor 已 detach 且为 clone |
| `test_slot_reuse_after_remove` | 移除后在同一 index 添加新请求 | 新请求数据正确，无旧数据残留 |

```python
# 示例测试
def test_add_request_stores_req_id():
    buffer = OmniIntermediateBuffer(max_num_reqs=4)
    req_data = DummyNewReqData(req_id="r1")
    buffer.add_request(req_index=0, new_req_data=req_data)
    assert buffer.buffers[0]["req_id"] == "r1"

def test_gather_returns_batch_order():
    buffer = OmniIntermediateBuffer(max_num_reqs=4)
    for i, rid in enumerate(["r0", "r1", "r2"]):
        buffer.add_request(i, DummyNewReqData(req_id=rid))
    batch = DummyInputBatch(
        req_ids=["r2", "r0"],
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
    )
    result = buffer.gather(batch)
    assert result[0]["req_id"] == "r2"
    assert result[1]["req_id"] == "r0"
```

### 10.4 `test_omni_model_state.py` — OmniModelState

测试 ModelState 的核心方法：请求生命周期、输入准备、输出后处理。

| 测试函数 | 测试内容 | 验证要点 |
|----------|---------|---------|
| `test_init_detects_model_flags` | 构造时读取 `has_preprocess`、`have_multimodal_outputs` | 属性正确反映模型标志 |
| `test_add_request_delegates_to_super_and_buffer` | `add_request` 调用链 | `super().add_request` 和 `intermediate_buffer.add_request` 均被调用（monkeypatch 跟踪） |
| `test_remove_request_clears_buffer_and_plugins` | 注册一个 dummy plugin 后 `remove_request` | buffer 已清理，plugin 的 `on_remove_request` 被调用 |
| `test_prepare_inputs_includes_buffer_keys` | 无插件时 `prepare_inputs` | 返回字典包含 `model_intermediate_buffer` 和 `runtime_additional_information` |
| `test_prepare_inputs_merges_plugin_extras` | 注册返回 `{"extra_key": 42}` 的 plugin | 返回字典包含 `extra_key` |
| `test_prepare_inputs_buffer_alias_consistency` | 检查两个别名键 | `base["model_intermediate_buffer"] is base["runtime_additional_information"]` |
| `test_postprocess_wraps_tuple_via_make_omni_output` | 传入 `(Tensor, dict)` 元组 | 调用 `make_omni_output`，返回 `(text_hidden, mm_outputs)` |
| `test_postprocess_passes_through_plain_tensor` | 传入普通 Tensor | `text_hidden` 是原 Tensor，`mm_outputs == {}` |
| `test_postprocess_extracts_omni_output` | 传入 `OmniOutput` 对象 | 正确提取 `text_hidden_states` 和 `multimodal_outputs` |
| `test_postprocess_runs_plugin_chain` | 注册两个 plugin | 两个 plugin 的 `postprocess` 按顺序被调用 |
| `test_register_plugin_appends` | 多次 `register_plugin` | `self.plugins` 长度递增 |
| `test_prepare_dummy_inputs_includes_buffer` | `prepare_dummy_inputs` | 返回字典包含正确长度的 `model_intermediate_buffer` 列表 |

```python
# 示例测试
def test_prepare_inputs_includes_buffer_keys():
    state = _make_omni_model_state(model=DummyModel())
    input_batch = DummyInputBatch(
        req_ids=["r1"], idx_mapping_np=np.array([0], dtype=np.int32)
    )
    result = state.prepare_inputs(input_batch, DummyRequestState())
    assert "model_intermediate_buffer" in result
    assert "runtime_additional_information" in result
    assert isinstance(result["model_intermediate_buffer"], list)

def test_postprocess_wraps_tuple_via_make_omni_output():
    model = DummyModel(have_multimodal_outputs=True)
    state = _make_omni_model_state(model=model)
    raw_output = (torch.randn(4, 8), {"layer_3": torch.randn(4, 8)})
    input_batch = DummyInputBatch(
        req_ids=["r1"], idx_mapping_np=np.array([0], dtype=np.int32)
    )
    text_hidden, mm = state.postprocess_model_output(
        raw_output, input_batch, DummyRequestState()
    )
    assert isinstance(text_hidden, torch.Tensor)
    assert "layer_3" in mm
```

### 10.5 `test_omni_model_state_plugin.py` — 插件接口

验证插件协议的正确性，为 Talker 插件迁移提供信心。

| 测试函数 | 测试内容 | 验证要点 |
|----------|---------|---------|
| `test_plugin_abc_cannot_instantiate` | 直接实例化 `OmniModelStatePlugin` | 抛出 `TypeError`（ABC） |
| `test_concrete_plugin_lifecycle` | 实现一个计数 plugin | `on_add_request` 和 `on_remove_request` 各被调用正确次数 |
| `test_plugin_prepare_extra_inputs_default_empty` | 默认实现 | 返回 `{}` |
| `test_plugin_postprocess_default_passthrough` | 默认实现 | 输入 `(text_hidden, mm)` 原样返回 |
| `test_multiple_plugins_compose` | 注册两个 plugin，一个修改 `mm_outputs` | 最终输出包含两个 plugin 的修改 |

```python
# 示例测试
class CountingPlugin(OmniModelStatePlugin):
    def __init__(self):
        self.add_count = 0
        self.remove_count = 0

    def on_add_request(self, req_index, new_req_data):
        self.add_count += 1

    def on_remove_request(self, req_index):
        self.remove_count += 1

    def postprocess(self, text_hidden, mm_outputs, input_batch, req_states):
        mm_outputs["plugin_marker"] = True
        return text_hidden, mm_outputs

def test_concrete_plugin_lifecycle():
    plugin = CountingPlugin()
    state = _make_omni_model_state(model=DummyModel())
    state.register_plugin(plugin)
    state.add_request(0, DummyNewReqData("r1"))
    state.add_request(1, DummyNewReqData("r2"))
    state.remove_request(0)
    assert plugin.add_count == 2
    assert plugin.remove_count == 1
```

### 10.6 `test_omni_gpu_model_runner.py` — OmniGPUModelRunner

测试薄继承层的关键覆写。

| 测试函数 | 测试内容 | 验证要点 |
|----------|---------|---------|
| `test_load_model_replaces_model_state` | `load_model` 后 `self.model_state` 类型 | `isinstance(runner.model_state, OmniModelState)` |
| `test_finish_requests_calls_remove_request` | 带 finished_req_ids 的 scheduler output | `model_state.remove_request` 按 req_index 被调用（monkeypatch 跟踪） |
| `test_finish_requests_handles_preempted` | preempted_req_ids 非空 | 预抢占的请求也触发 `remove_request` |
| `test_finish_requests_skips_unknown_req_ids` | finished_req_ids 中包含不存在的 req_id | 不抛异常，仅处理已知请求 |
| `test_execute_model_intercepts_tuple_output` | 模型返回 `(Tensor, dict)` 后检查 `execute_model_state` | `hidden_states` 被替换为 `OmniOutput` |
| `test_execute_model_passthrough_tensor_output` | 模型返回普通 Tensor | `hidden_states` 保持原样，不触发拦截 |

```python
# 示例测试
def test_finish_requests_calls_remove_request(monkeypatch):
    runner = _make_v2_runner(req_ids=["r1", "r2"])
    remove_log = []
    monkeypatch.setattr(
        runner.model_state, "remove_request",
        lambda idx: remove_log.append(idx)
    )
    scheduler_output = DummySchedulerOutput(finished_req_ids={"r1"})
    runner.finish_requests(scheduler_output)
    assert len(remove_log) == 1
    assert remove_log[0] == runner.req_states.req_id_to_index["r1"]
```

### 10.7 `test_omni_ar_model_runner.py` — OmniARModelRunner

测试 AR runner 的 `sample_tokens` 和 `pooler_output` 构建。

| 测试函数 | 测试内容 | 验证要点 |
|----------|---------|---------|
| `test_sample_tokens_returns_none_when_no_state` | `execute_model_state = None` | 返回 `None` |
| `test_sample_tokens_calls_postprocess_model_output` | 正常流程 | `model_state.postprocess_model_output` 被调用（monkeypatch 跟踪） |
| `test_sample_tokens_builds_pooler_output` | 2 个请求 | `pooler_output` 长度为 2，每个包含 `hidden` 键 |
| `test_pooler_output_hidden_shape_matches_scheduled_tokens` | 请求各有不同 scheduled_tokens | 每个 `hidden` 切片长度与 `num_scheduled_tokens` 一致 |
| `test_pooler_output_includes_multimodal_slices` | `multimodal_outputs` 含 Tensor | `pooler_output` 中包含对应的多模态切片 |
| `test_pooler_output_includes_list_multimodal` | `multimodal_outputs` 含 list 类型 | 列表元素按 req_index 正确映射 |
| `test_pooler_output_empty_multimodal` | `multimodal_outputs = {}` | `pooler_output` 每项仅含 `hidden` |
| `test_pooler_output_is_none_for_text_engine` | `engine_output_type == "text"` | `pooler_output is None` |
| `test_handle_kv_transfer_pre_extracts_ids` | 带 `finished_requests_needing_kv_transfer` | `_kv_extracted_req_ids` 非空 |
| `test_handle_kv_transfer_pre_noop_when_empty` | 无需传输的请求 | `_kv_extracted_req_ids` 为空列表或 None |
| `test_sample_tokens_output_type` | 正常流程 | 返回类型为 `OmniModelRunnerOutput` |
| `test_sample_tokens_async_mode` | `use_async_scheduling = True` | 返回类型含 `AsyncOutput` 包装 |

```python
# 示例测试
def test_sample_tokens_builds_pooler_output(monkeypatch):
    runner = _make_ar_runner(req_ids=["r1", "r2"], hidden_size=8)
    text_hidden = torch.randn(4, 8)  # 2 tokens per request
    mm_outputs = {}
    monkeypatch.setattr(
        runner.model_state, "postprocess_model_output",
        lambda *a, **kw: (text_hidden, mm_outputs)
    )
    _setup_execute_model_state(runner, text_hidden)
    output = OmniARModelRunner.sample_tokens(runner, grammar_output=None)
    assert output is not None
    assert len(output.pooler_output) == 2
    for item in output.pooler_output:
        assert "hidden" in item
        assert item["hidden"].shape[0] > 0

def test_pooler_output_hidden_shape_matches_scheduled_tokens(monkeypatch):
    runner = _make_ar_runner(req_ids=["r1", "r2"], hidden_size=8)
    # r1 调度 1 个 token，r2 调度 3 个 token
    runner._input_batch_query_start_loc = np.array([0, 1], dtype=np.int32)
    runner._input_batch_num_scheduled = np.array([1, 3], dtype=np.int32)
    text_hidden = torch.randn(4, 8)
    monkeypatch.setattr(
        runner.model_state, "postprocess_model_output",
        lambda *a, **kw: (text_hidden, {})
    )
    _setup_execute_model_state(runner, text_hidden)
    output = OmniARModelRunner.sample_tokens(runner, grammar_output=None)
    assert output.pooler_output[0]["hidden"].shape[0] == 1
    assert output.pooler_output[1]["hidden"].shape[0] == 3
```

### 10.8 `test_init_model_state.py` — 工厂函数

测试架构分发逻辑。

| 测试函数 | 测试内容 | 验证要点 |
|----------|---------|---------|
| `test_omni_architecture_returns_omni_state` | 传入 Qwen3 架构名 | `isinstance(result, OmniModelState)` |
| `test_non_omni_architecture_falls_through` | 传入非 Omni 架构名 | `isinstance(result, DefaultModelState)` |
| `test_whisper_architecture_returns_whisper_state` | 传入 Whisper 架构名 | 不返回 `OmniModelState`（上游逻辑优先） |
| `test_multiple_architectures_with_omni` | `architectures` 列表含 Omni + 其他 | 命中 Omni 分支 |
| `test_empty_architectures_falls_through` | `architectures = []` | 返回 `DefaultModelState` |

```python
# 示例测试
def test_omni_architecture_returns_omni_state():
    vllm_config = _make_dummy_config(
        architectures=["Qwen3OmniMoeForConditionalGeneration"]
    )
    model = DummyModel()
    result = init_omni_model_state(vllm_config, model, None, torch.device("cpu"))
    assert isinstance(result, OmniModelState)

def test_non_omni_architecture_falls_through():
    vllm_config = _make_dummy_config(
        architectures=["LlamaForCausalLM"]
    )
    model = DummyModel()
    result = init_omni_model_state(vllm_config, model, None, torch.device("cpu"))
    assert not isinstance(result, OmniModelState)
```

### 10.9 测试覆盖矩阵

| 被测组件 | 单元测试文件 | 覆盖的关键路径 |
|---------|------------|--------------|
| `OmniIntermediateBuffer` | `test_intermediate_buffer.py` | add / remove / gather / update / 槽位复用 / 空批次 |
| `OmniModelState` | `test_omni_model_state.py` | 请求生命周期 / prepare_inputs / postprocess / 插件分发 / dummy inputs |
| `OmniModelStatePlugin` | `test_omni_model_state_plugin.py` | ABC 约束 / 生命周期回调 / 默认实现 / 多插件组合 |
| `OmniGPUModelRunner` | `test_omni_gpu_model_runner.py` | load_model 替换 / finish_requests 清理 / execute_model 输出拦截 |
| `OmniARModelRunner` | `test_omni_ar_model_runner.py` | sample_tokens 全流程 / pooler_output 构建 / KV 传输前置 / 异步模式 |
| `init_omni_model_state` | `test_init_model_state.py` | 架构分发 / 向后兼容 / 边界情况 |

### 10.10 不在单元测试范围内（需集成/E2E 测试）

以下场景需要真实的模型权重、GPU 和/或调度器配合，不适合纯 CPU 单元测试：

| 场景 | 测试层级 | 说明 |
|------|---------|------|
| M-RoPE 位置正确性 | 集成测试 | 需要真实的 `rope_state` + 多模态特征 |
| 多模态编码器执行 | 集成测试 | 需要 `EncoderRunner` + 实际视觉/音频塔 |
| CUDA Graph 捕获和回放 | GPU 集成测试 | 需要 GPU + `CUDAGraphManager` |
| 完整 `execute_model` → `sample_tokens` 链路 | 集成测试 | 需要真实的注意力后端、KV 缓存 |
| v1 vs v2 输出一致性 | 回归测试 | 需要同一模型在两个 runner 上运行对比 |
| 性能基准 | 基准测试 | 需要 GPU + 真实模型 + 多请求负载 |
| 端到端 Qwen3-Omni Thinker | E2E 测试 | 需要完整的模型权重和推理管线 |

---

## 11. 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| v2 `ExecuteModelState` 期望 `hidden_states: torch.Tensor`，但 Thinker 返回元组 | `sample_tokens` 运行时崩溃 | 在 `execute_model` 中后拦截，存储前包装为 `OmniOutput` |
| v2 异步 `AsyncOutput` 与 `pooler_output` CPU 拷贝竞争 | 数据损坏 | 在同一 `copy_stream` 中序列化 `pooler_output` 拷贝，或在 `postprocess()` 之前同步构建 |
| `prepare_inputs()` 返回字典的键被未来 v2 版本过滤 | kwargs 静默丢失 | 集成测试断言 kwargs 到达 `model.forward()`；兜底：覆写 `execute_model` |
| `rope_state` 不支持 Qwen3-Omni 的 `get_mrope_input_positions` 签名 | 位置错误 | Phase 0 验证；兜底：自定义 `RopeState` 子类 |
| v2 `model_runner.py` CUDA graph 路径绕过 `model_inputs` 字典 | graph 模式下缺少 kwargs | FULL graph 模式回放捕获的缓冲区；`prepare_inputs()` 在捕获期间被调用。Phase 0 中验证。 |

---

## 12. 待定事项

1. **`prompt_logprobs` 支持：** v2 有 `PromptLogprobsWorker`。需验证其与 `OmniOutput` 的兼容性（text_hidden 提取必须在 logprobs 计算之前完成）。
2. **推测解码：** v2 的 `speculator.propose()` 期望标准 `hidden_states`。需验证与后处理后的 `text_hidden` 的兼容性。
3. **v2 中的 `has_preprocess` 路径：** Thinker 不需要，但设计应文档化 Talker 如何在没有 v1 `_preprocess` 循环的情况下处理每请求的 `preprocess()` 调用。候选方案：基于插件的 `prepare_extra_inputs()`，在插件内部进行每请求迭代。
4. **`prompt_embeds` 的 `UvaBackedTensor`：** 需要性能分析以确定 UVA 访问模式是否优于显式异步拷贝。推迟到 Phase 1。
