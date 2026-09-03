# Breeze-TTS-2 适配 vLLM-Omni：从 0 到 1 实施手册

> 当前代码已经实现和未实现的能力、请求输入输出契约及限制，见：[Breeze-TTS-2 当前能力与限制](breeze_tts_2_capabilities.md)。本文保留为从 0 到 1 的适配实施手册。

本文回答一个问题：拿到 `/Users/liuzhiwei08/Desktop/work/Model/breeze-tts` 上游代码和 Breeze checkpoint 后，如何一步一步把它落到 vLLM-Omni。

目标不是一次性移植整个上游 runtime，而是按下面顺序得到一个可验证的结果：

```text
checkpoint config
    -> vLLM 配置注册
    -> stage 0 talker（文本/音频 embedding + Qwen3 backbone）
    -> codebook-0
    -> depth decoder codebook-1..15
    -> (T, 16) audio codes
    -> stage 1 audio codec decoder
    -> 24 kHz waveform
    -> pipeline / serving / streaming / performance
```

每一步必须先通过自己的验收，再进入下一步。第一版只做：单请求、eager、greedy、同步 full payload、无 CFG。continuous batching、streaming、CUDA Graph 和量化都放到最后。

## 0. 先确认模型边界

Breeze-TTS-2 由四个组件组成：

| 组件 | 当前配置 | 责任 | vLLM 落点 |
| --- | --- | --- | --- |
| Qwen3 backbone | 2048 hidden、28 层、16 Q heads/8 KV heads | 对序列做主 Transformer 推理 | stage 0 内的 `Qwen3Model` |
| Breeze codebook-0 head | 输出 2052 类 | 生成第一路 code，以及额外 EOS | stage 0 的 `compute_logits()` |
| depth decoder | 1024 hidden、12 层、16 codebooks | 根据 backbone hidden 生成 codebook-1..15 | stage 0 内的 per-frame decoder |
| T5Gemma2 text encoder | 1152 hidden、26 层 | 把文本 segment 编成 2048 维条件 | stage 0 输入 embedding 路径 |
| Breeze bundled audio tokenizer / Mimi fallback | 24 kHz、codebook size 2048；根配置的 Mimi 为 32 quantizers | audio codes 和 waveform 互转 | stage 1 |

关键数字：

- `backbone_config.vocab_size=151936` 只是 Qwen3 Transformer 配置，不是 Breeze 输出 head 的词表；
- Breeze 的主 `lm_head` 是 `config.vocab_size + 1 = 2052`，其中 `[0, 2050]` 是 codebook-0 候选，`2051` 是额外 EOS；
- depth decoder 的每个 head 输出 2051 类，合法 codec id 只有 `[0, 2047]`，`2048..2050` 必须 mask；
- `num_codebooks=16` 是 Breeze 每帧生成的路数；Mimi `num_quantizers=32` 不能直接代替它；
- 文本特殊 token 的范围到 262157，不能直接拿来查 Qwen3 默认 embedding。

上游参考文件：

```text
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/models/breeze.py
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/models/breeze_base_config.py
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/models/breeze_config.py
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/models/generation_breeze.py
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/models/t5gemma2_compat.py
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/breeze_infer/templates.py
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/breeze_infer/runtime.py
```

## 1. 建立分支、固定依赖和 checkpoint 目录

### 目标

让后续测试可复现，并确认 checkpoint、文本 tokenizer、音频 tokenizer 和 codec 权重齐全。

### 修改/新增文件

本阶段不改模型代码，建议新增：

```text
tests/model_executor/models/breeze_tts_2/conftest.py
tests/model_executor/models/breeze_tts_2/test_checkpoint_inventory.py
```

### 要写什么

在 `test_checkpoint_inventory.py` 中检查：

```text
config.json
tokenizer.json / tokenizer_config.json
audio_tokenizer/
model*.safetensors 或 pytorch_model*.bin
```

用 `safetensors.safe_open()` 输出所有 key 和 shape，保存一份审计结果。不要先凭猜测写 `load_weights()`。

同时记录 vLLM-Omni commit、上游 `transformers==4.57.3`、CUDA/PyTorch/GPU，以及上游固定输入的基准音频和 code。

### 验收

- `config.json` 能读取；
- checkpoint 中能找到 `BreezeForConditionalGeneration` 对应权重；
- `audio_tokenizer` 目录存在；
- 上游 PyTorch runtime 能生成一条基准音频和一份基准 code。

### 进入下一步的条件

基准音频可以重复生成，且 checkpoint key 清单已经保存。若缺 codec 或 tokenizer，先补齐模型材料。

## 2. 实现并注册 Breeze 配置

### 目标

让 vLLM 正确解析 Breeze 根配置，并让 KV cache 使用 Qwen3 backbone 尺寸。

### 修改文件

```text
vllm_omni/model_executor/models/breeze_tts_2/configuration_breeze_tts_2.py
vllm_omni/engine/arg_utils.py
```

### 要写什么

`configuration_breeze_tts_2.py` 保留上游字段名，但按 MOSS-TTS 方式实现：

1. `BreezeTTS2Config(PretrainedConfig)` 放在文件最前面；
2. `backbone_config` 用 `None` / `dict` 判断，字典复制后移除 `model_type`，构造 `Qwen3Config(**backbone_config)`；
3. `text_encoder_config`、`depth_decoder_config` 分别构造本地配置类；
4. 不声明 `sub_configs`；
5. `codec_config` 保留为字典，codec 由 stage 1 加载；
6. 实现 `get_text_config()` 返回 `self.backbone_config`；
7. 保留 `num_codebooks`、`vocab_size`、`text_vocab_size`、音频 token、projection 和 special-token 字段；
8. `layer_types`、`rope_parameters` 为空时使用 checkpoint 的默认 pattern；
9. `preferred_attn_implementation` 只作为 text encoder 元数据，不作为 vLLM 全局 attention 开关。

在 `arg_utils.py` 的 `_register_omni_hf_configs()` 中导入并注册 `"breeze"`、`"breeze_depth_decoder_model"`、`"t5gemma2_text"`，重复注册按现有代码捕获异常。

### 测试文件

```text
tests/model_executor/models/breeze_tts_2/test_configuration.py
```

至少验证：

```python
cfg = BreezeTTS2Config.from_pretrained(checkpoint)
assert cfg.model_type == "breeze"
assert cfg.get_text_config().hidden_size == 2048
assert cfg.get_text_config().vocab_size == 151936
assert cfg.num_codebooks == 16
assert cfg.depth_decoder_config.num_codebooks == 16
assert cfg.depth_decoder_config.vocab_size == 2051
assert cfg.text_encoder_config.vocab_size == 262158
```

再验证 `cfg.to_dict()` 后重新构造不会丢三个嵌套配置。

### 注意

- 外层 `hidden_size=2048` 不能代替嵌套 `backbone_config`；
- 外层 `vocab_size=2051` 不能拿来构造 Qwen3 词表；
- `backbone_config.max_position_embeddings` 当前是 40960，不能被外层默认值 2048 覆盖；
- 上游 `BreezeConfig` 可以用于 reference script，但正式 vLLM 配置必须有 `get_text_config()`。

### 验收

```bash
pytest -q tests/model_executor/models/breeze_tts_2/test_configuration.py
```

## 3. 注册模型架构并建立空 talker

### 目标

让 vLLM 能根据 `architectures: ["BreezeForConditionalGeneration"]` 找到 stage 0 类。

### 修改/新增文件

```text
vllm_omni/model_executor/models/registry.py
vllm_omni/model_executor/models/breeze_tts_2/__init__.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
```

### 要写什么

在 `_OMNI_MODELS` 增加映射，类名必须和 pipeline 的 `model_arch` 一致：

```python
"BreezeForConditionalGeneration": (
    "breeze_tts_2",
    "modeling_breeze_tts_2_talker",
    "BreezeTTS2TalkerForGeneration",
),
```

talker 先实现骨架：

```python
class BreezeTTS2TalkerForGeneration(nn.Module):
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True

    def __init__(self, *, vllm_config, prefix=""): ...
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs): ...
    def compute_logits(self, hidden_states, sampling_metadata=None): ...
    def load_weights(self, weights): ...
```

不要在这里继承 HuggingFace `GenerationMixin`。

### 测试文件和验收

新增 `tests/model_executor/models/breeze_tts_2/test_model_registration.py`，验证 `OmniModelRegistry` 和 upstream `ModelRegistry` 都能解析架构名。空 talker 可以不推理，但架构解析不能失败。

## 4. 实现文本 tokenizer、prompt 和参考音频输入

### 目标

把用户请求变成 Breeze 真实需要的输入。

### 先分清三个组件

- 文本 tokenizer：checkpoint 根目录的 HuggingFace `AutoTokenizer`；
- 参考音频 tokenizer：`audio_tokenizer/` 下的 `Qwen3TTSTokenizer`，waveform -> reference code；
- 生成音频 decoder：优先使用 checkpoint `audio_tokenizer/` 下的 `Qwen3TTSTokenizer.decode()`；缺少该目录时才使用 `codec_config.model_type="mimi"` 的 Stage 1 fallback。根配置中的 Mimi `num_quantizers=32` 不等于 Breeze 生成的 16 路 codes。

MOSS 的 `audio_tokenizer.py` 是神经 codec，不是文本 tokenizer，不能直接替代 Breeze 的 Qwen3/Mimi tokenizer。

### 修改/新增文件

```text
vllm_omni/model_executor/models/breeze_tts_2/prompt_builder.py
vllm_omni/model_executor/models/breeze_tts_2/audio_tokenizer.py
vllm_omni/entrypoints/openai/tts_adapters/breeze_tts_2.py
vllm_omni/model_executor/stage_input_processors/breeze_tts_2.py
```

这里不要创建一个同时叫“tokenizer”的大杂烩文件。Breeze 有三种不同的处理，必须按数据流拆开。`prompt_builder.py` 是 stage 0 的模型专属 prompt 组装器，不是新的 tokenizer 实现：

| 数据 | 实际组件 | 推荐代码位置 | 谁调用 |
| --- | --- | --- | --- |
| 用户文本 -> 文本 token ids | checkpoint 根目录的 HuggingFace `AutoTokenizer` | `prompt_builder.py`（参考 Qwen3-TTS 的 `Qwen3TTSPromptEmbedsBuilder`） | OpenAI TTS adapter 调用 builder；talker 只消费结果 |
| 参考 wav -> 16 路 reference codes | `audio_tokenizer/` 下的 `Qwen3TTSTokenizer.encode()` | `audio_tokenizer.py` 的 `BreezeReferenceAudioTokenizer` | `prompt_builder.py` 通过注入的 encoder 回调调用 |
| 生成 codes -> wav | Mimi decoder / Code2Wav | stage 1 的模型；stage 间传输在 `stage_input_processors/breeze_tts_2.py` | pipeline 的 stage 1 |

`StagePipelineConfig.owns_tokenizer=True` 表示 stage 0 使用自己的 tokenizer 生命周期和 tokenizer 路径；它不表示“文本 tokenizer 和音频 codec 必须写在同一个 Python 文件里”。在当前 Qwen3-TTS 实现中，文本 tokenizer 仍由 prompt builder 以 `AutoTokenizer.from_pretrained()` 延迟加载，这也是 Breeze 最接近的复用方式。

`prompt_builder.py` 不是 vLLM-Omni 的框架接口，但在 Breeze 适配中作为独立模块实现。之所以放在 `models/breeze_tts_2/`，是因为模板、特殊 token、segment mask、文本/音频 embedding 拼接规则都属于 Breeze 模型协议，而不是通用 stage payload 协议。

### `prompt_builder.py` 的职责

输入是单个请求的 `info_dict` 和 talker 提供的参考音频编码器；输出是 scheduler 可接受的 token prompt，以及由 talker 后续消费的运行时 metadata。当前实现只暴露一个主入口：

```python
class BreezeTTS2PromptBuilder:
    def build(self, request: Mapping[str, Any]) -> OmniTokensPrompt:
        """Return prompt_token_ids and additional_information."""
```

内部按以下顺序实现：

1. 根据 `template` / `instruction` / `ref_text` / `text` 生成 Breeze segment 序列；
2. 使用 checkpoint 根目录的 `AutoTokenizer` 编码所有 text segment；
3. 对 audio segment 调用注入的 `BreezeReferenceAudioTokenizer.encode()`，得到 `(T_ref, 16)` codes；
4. 按上游 Breeze `templates.py` 的流程，先对每个 text segment 加 special tokens 并 decode，再插入 audio placeholder，最后对完整渲染字符串做一次 `add_special_tokens=False` 的 tokenization；
5. 生成并校验 `text_ids_mask`、`text_ids_len`，保证 text/audio segment 的长度与最终 prompt 对齐；
6. 生成与 token 序列等长的 `attention_mask` 和 `text_ids_mask`；
7. 返回 `prompt_token_ids` 以及 `input_values`、`ref_code_len` 等运行时 metadata。文本/T5Gemma2 和 audio embedding 在第五章由 talker 完成。

建议返回值至少包含：

```text
prompt_token_ids  长度 S；提交 scheduler 的合法占位 token（当前为 pad_token_id）
prompt_ids        长度 S；Breeze 原始文本/audio token，存于 additional_information
attention_mask    (S,) bool
text_ids_mask     (S,) bool
text_ids_len      每个 text segment 的长度
input_values      (T_ref, 16) int16，或不存在
ref_code_len      T_ref，或不存在
```

它不负责：

- Qwen3 backbone forward、codebook-0 采样和 depth decoder；
- 持有跨 decode step 的 AR 状态；
- stage 0 -> stage 1 的 payload/chunk 传输；
- Mimi waveform decode。

### adapter 和 talker 的调用边界

serving adapter 在首次请求时创建并缓存 builder，把已加载的参考音频编码器注入；stage 0
talker 不加载 tokenizer，只消费 builder 产出的标准化 metadata：

```python
builder = BreezeTTS2PromptBuilder.from_pretrained(
    model_path,
    config,
    reference_audio_encoder=reference_audio_tokenizer,
)
prompt = builder.build(payload, template=template)
```

`BreezeTTS2Adapter.build()` 负责调用 builder；talker 的 `preprocess()` 只负责识别 prefill/decode，并把 metadata 交给第五章的 embedding 路径：

```python
prompt = await asyncio.to_thread(builder.build, payload, template=template)
return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="breeze_tts_2")
```

stage 0 的 prompt 具体构造过程为：

1. 调用 checkpoint 根目录的 `AutoTokenizer`，支持 `[S0]`、`<ins_bos>`、`<ins_eos>`；
2. reference 模式读取 wav，调用 `audio_tokenizer/` 下的 `Qwen3TTSTokenizer.encode()`；
3. 参考音频位置在 `prompt_ids` 中写入 `audio_token_id=262144`，末尾写 `audio_eos_token_id=262145`；
4. 生成：

   ```text
   prompt_token_ids  (S,)；scheduler 可接受的占位 token
   prompt_ids        (S,)；Breeze 原始 token（`additional_information`）
   attention_mask    (S,) bool
   text_ids_mask     (S,) bool
   text_ids_len      segment lengths
   input_values      (T_ref, 16) int16 或 None
   ```

5. 保证 `text_ids_len` 覆盖所有 `text_ids_mask=True` 的 token；
6. padding 时同步处理三个 mask/id 张量。

参考：

```text
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/breeze_infer/templates.py
/Users/liuzhiwei08/Desktop/work/Model/breeze-tts/breeze_infer/audio.py
```

`audio_tokenizer.py` 只做参考音频的 waveform -> `(T, 16)` code 转换，并复用上游
`Qwen3TTSTokenizer.encode()`；它不负责文本 tokenizer，也不负责生成音频的 decode。

`entrypoints/openai/tts_adapters/breeze_tts_2.py` 负责把 OpenAI speech request
转换成 builder 的 request 字典，并在 serving worker 内缓存文本 tokenizer、参考音频
tokenizer 和 builder。这样 tokenizer 不会在每次请求或每个 decode step 重载。

`stage_input_processors/breeze_tts_2.py` 只做跨 stage 的 payload 转换：

- `talker2codec()` 处理同进程同步输出，从 `(T, 16)` 转成 codebook-major `prompt_token_ids`；
- `talker2codec_full_payload()` 处理 connector 的最终 payload，从累计 codes 构造 Stage 1 输入，并返回 `OmniPayloadStruct`（connector 会以 `multimodal_output` 关键字调用）；
- `_FULL_PAYLOAD_REPLACE_KEYS = {"codes.audio"}` 让每步的“累计完整序列”覆盖旧值，避免通用 accumulator 重复拼接历史帧。

它不加载 `AutoTokenizer`，也不执行 wav 编码；async-chunk 仍留到后续章节。

### vLLM 特有输入处理

Breeze prompt id 最高到 262157，而 Qwen3 embedding 词表只有 151936：

- 不能让 Qwen3 默认 embedding 直接查完整 prompt；
- `prompt_ids` 在 additional_information 中保留 Breeze 的高位 text/audio id；第五章 talker 在进入 Qwen3 backbone 前负责把这些位置替换为 embedding；
- 提交给 scheduler 的 `prompt_token_ids` 使用等长、合法的 dummy/pad id，不能改变序列长度；
- 生成阶段的 `(B, 1, 16)` frame 通过 `embed_input_ids()` 变成下一步 embedding。

参考：

```text
vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py
vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py
```

建议 Breeze 直接复制这个边界：

- `prompt_builder.py` 只负责把文本 token、reference codes 和 speaker/instruction 信息拼成 prompt token，并做 mask/shape 校验；它可以持有一个懒加载的文本 `AutoTokenizer`；
- serving adapter 负责一次性加载并缓存文本 tokenizer、参考音频 tokenizer 和 prompt builder；talker 不重复加载这些 CPU 组件，只消费 `prompt_ids`、mask 和 `input_values`；
- `stage_input_processors/breeze_tts_2.py` 只负责 Omni payload 和同步 stage 间 codes 传输，不负责文本 tokenizer、wav 编码，也不负责把 Mimi decoder 混进 stage 0；
- 如果上游 `Qwen3TTSTokenizer` 无法直接复用，新增 `models/breeze_tts_2/audio_tokenizer.py`，类名应明确为 `BreezeReferenceAudioTokenizer`，只实现 wav -> `(T, 16)` codes；不要再使用含糊的 `tokenizer_breeze_tts_2.py`。

### 测试和验收

新增：

```text
tests/model_executor/models/breeze_tts_2/test_prompt_builder.py
tests/model_executor/models/breeze_tts_2/test_stage_input_processor.py
```

`test_prompt_builder.py` 测试 plain TTS、instruction、reference clone 和多个 text segment；检查 token 数、mask、`text_ids_len`、audio code shape 与上游 `templates.py` 一致。`test_stage_input_processor.py` 只测试同步 stage 0 -> stage 1 的 codes payload；chunk 行为在后续 streaming 章节实现。

## 5. 实现 text encoder 和 Breeze audio embedding

### 目标

让 stage 0 prefill embedding 与上游 `_merge_input_ids_with_input_values()` 一致。

### 修改文件

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_text.py
tests/model_executor/models/breeze_tts_2/test_embedding_alignment.py
```

### 要写什么

1. text encoder 先实现 eager 版本，并由 talker 注入 `prompt_builder.py`；
2. 复用 T5Gemma2 的 layer type、sliding/full attention、RoPE、RMSNorm 数学；
3. `text_encoder_proj` 将 1152 投影到 2048；
4. builder 只对 `text_ids_mask=True` 的独立 segment 调用 text encoder；
5. 实现 16 路 audio embedding：

   ```text
   audio_embed = sum(Embedding(code_i + i * 2051) for i in range(16))
   ```

6. builder 将 audio embedding 写回 `input_ids == audio_token_id` 的位置；
7. `audio_eos_token_id` 位置写入 16 路 `codebook_eos_token_id` embedding；
8. builder 输出 `(num_tokens, 2048)` 的 `inputs_embeds`，talker 只负责把它送入 backbone。

### 注意

- text encoder 只编码 prompt，不参与每个音频帧的 AR decode；
- 不同 segment 不能互相 attention；
- 第一版只支持 `text_encoder_proj_type="linear"`；
- LoRA、dim fusion、dual CFG 放到后面。

### 验收

从上游导出 text encoder 输出、projection 输出、merged `inputs_embeds` 和 audio embedding，用 BF16 宽松阈值 `atol=2e-2, rtol=2e-2` 对齐。至少一条 plain TTS 和一条 reference TTS 通过后再进入 backbone。

## 6. 接入 Qwen3 backbone 和 codebook-0 head

### 目标

先完成“输入 embedding -> Qwen3 hidden -> codebook-0 logits”。

### 修改文件

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
tests/model_executor/models/breeze_tts_2/test_backbone_alignment.py
```

### 要写什么

在 talker 中：

1. 用 `cfg.backbone_config` 构造 `vllm_config.with_hf_config(..., architectures=["Qwen3ForCausalLM"])`；
2. 使用 vLLM `Qwen3Model`，不要搬运上游 `BreezeDecoderLayer`；
3. 主 head 使用：

   ```python
   ParallelLMHead(cfg.vocab_size + 1, 2048, bias=False, prefix=...)
   ```

4. `forward()` 支持 `input_ids`、`positions`、`intermediate_tensors`、`inputs_embeds`；
5. `embed_input_ids()` 将生成的 16 路 frame 转成 audio embedding；
6. `compute_logits()` 输出 2052 类 logits。

上游主 head 是 `nn.Linear(config.hidden_size, config.vocab_size + 1)`，所以 checkpoint 的 `lm_head.weight` 应为 `(2052, 2048)`，不是 Qwen3 的 `(151936, 2048)`。

### 验收

固定同一份 merged embedding，比较上游和 vLLM 的最后 hidden、`lm_head` logits、greedy codebook-0 和 EOS 行。

## 7. 实现 depth decoder

### 目标

给定 backbone 最后 hidden 和 codebook-0，生成完整 16 路 code。

### 修改文件

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_depth.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
tests/model_executor/models/breeze_tts_2/test_depth_decoder_alignment.py
```

### 要写什么

实现 1024 hidden、12 层、8 Q heads、2 KV heads 的 decoder：

1. 位置 0 用 `backbone_last_hidden_state`，不能查普通 token embedding；
2. 位置 1..15 使用前一个 code 的 offset embedding；
3. 每个位置使用独立的 `codebooks_head[p]`，权重 `(15, 1024, 2051)`；
4. 每帧位置从 0 重新开始，不能跨帧复用 depth KV cache；
5. mask `[2048, 2050]`，只采样 `[0, 2047]`；
6. 第一版使用 eager、greedy、固定 15 次循环。

接口建议：

```python
def generate_frame(self, backbone_last_hidden_state, first_codebook):
    # return (B, 15)
    ...
```

talker 再执行 `torch.cat([first_codebook, depth_codes], dim=-1)`。

不能复用 MOSS local depth transformer，也不能把 `codebook_eos_token_id=0` 当作主模型 EOS。

### 验收

固定 `backbone_last_hidden_state` 和 `first_codebook`，比较上游和 vLLM 的 15 路 greedy code。

## 8. 把生成循环改造成 vLLM talker 状态机

### 目标

把上游 `GenerationMixin._sample()` 拆成 vLLM 能调度的职责。

### 修改文件

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
vllm_omni/model_executor/stage_input_processors/breeze_tts_2.py
tests/model_executor/models/breeze_tts_2/test_generation_state.py
```

### 要写什么

职责必须分开：

- `forward()`：只跑 Qwen3 backbone；
- `compute_logits()`：只算 codebook-0 logits 和 mask；
- `process_input()` 或等价 hook：处理 prefill 和下一帧 embedding；
- `make_omni_output()`：调用 depth decoder，输出每请求 `(T, 16)` codes；
- request state：保存 `step`、`finished`、`generated_frames`、`max_new_frames` 和累计 codes。

循环严格是：

```text
prefill prompt
  -> backbone hidden
  -> sample codebook-0
  -> depth decoder sample codebook-1..15
  -> emit one (1, 16) frame
  -> embed this frame
  -> next backbone decode
```

第一版只实现 `cfg_scale=1.0`。非音频阶段屏蔽音频控制 token，音频阶段屏蔽 `[2048, 2050]`，达到 `max_new_frames` 时强制主 EOS（第 2051 类）。

### 验收

单请求 greedy 生成有限帧数；每帧 shape `(16,)`；code 不越界；不会因为缺 EOS 无限循环；前几帧和上游一致。

## 9. 实现 stage 1 audio codec

### 目标

把 stage 0 的 `(T, 16)` codes 还原为 24 kHz waveform。上游 Breeze runtime 默认通过 bundled `audio_tokenizer/` 解码，Stage 1 应优先复用其中的 `Qwen3TTSTokenizer.decode()`；只有 checkpoint 没有该目录时，才使用 `codec_config.model_type="mimi"` 的兼容 fallback。

### 先做兼容性判断

优先检查现有 `qwen3_tts` code2wav 是否能加载 Breeze 的 Mimi/Qwen3-TTS tokenizer 权重。如果输入布局或权重不兼容，新增独立 codec 类；不要使用 MOSS codec 代替 Breeze codec。

### 修改/新增文件

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_codec.py  # 仅不复用现有 stage 时
vllm_omni/model_executor/stage_input_processors/breeze_tts_2.py
tests/model_executor/models/breeze_tts_2/test_codec.py
```

### 要写什么

```text
(T, 16)
  -> transpose -> (16, T)
  -> stage-1 payload
  -> Qwen3TTSTokenizer.decode()（优先）/ Mimi decode（fallback）
  -> mono waveform, 24 kHz
```

检查 codebook 数量、code 值 `<2048`、pad/EOS 截断、空 code 的 silence/sentinel 和输出采样率 24000。

### 验收

固定 codes 能被 stage 1 解码；输出非空、无 NaN、长度合理，且与上游 decode 结果一致。

## 10. 在模型目录创建 pipeline.py

### 目标

声明 Breeze 两阶段 pipeline。文件必须放在模型目录：

```text
vllm_omni/model_executor/models/breeze_tts_2/pipeline.py
```

参考：

```text
vllm_omni/model_executor/models/moss_tts/pipeline.py
vllm_omni/model_executor/models/qwen3_tts/pipeline.py
```

### 要写什么

第一版声明 stage 0 talker 和 stage 1 codec：

```python
BREEZE_TTS_2_PIPELINE = PipelineConfig(
    model_type="breeze",
    default_deploy_config_name="breeze_tts_2.yaml",
    model_arch="BreezeForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="breeze_tts_2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=(
                "vllm_omni.model_executor.stage_input_processors.breeze_tts_2.talker2codec_full_payload"
            ),
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="breeze_tts_2_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="BreezeTTS2MimiCodec",
            sync_process_input_func=(
                "vllm_omni.model_executor.stage_input_processors.breeze_tts_2.talker2codec"
            ),
            sampling_constraints={"detokenize": True},
        ),
    ),
)
```

`BreezeTTS2MimiCodec` 是当前 stage 1 的统一包装名；它优先调用 checkpoint 自带的 Qwen3-TTS tokenizer decoder，缺失时才实例化 Mimi。若未来替换为已有 code2wav stage，只需替换 pipeline 的 `model_arch` 和对应 codec wrapper，不要改变 stage 0 的 `(T, 16)` 接口。

### 关键规则

- pipeline 文件在 `models/breeze_tts_2/`；
- 全局注册在 `vllm_omni/config/pipeline_registry.py`；
- checkpoint 根配置是 `model_type="breeze"`，因此 pipeline 和 registry key 默认使用 `"breeze"`；
- 若提供 `"breeze_tts_2"` 别名，必须同时保留 `"breeze"`；
- stage 0 `model_arch` 必须和 `registry.py` 完全一致。

## 11. 注册 pipeline 和部署配置

### 修改文件

```text
vllm_omni/config/pipeline_registry.py
vllm_omni/deploy/breeze_tts_2.yaml
```

### 要写什么

在 `pipeline_registry.py` 增加：

```python
from vllm_omni.model_executor.models.breeze_tts_2.pipeline import (
    BREEZE_TTS_2_PIPELINE,
)

OMNI_PIPELINES["breeze"] = BREEZE_TTS_2_PIPELINE
```

第一版部署配置只打开 eager 和同步 full payload，配置足够的 `max_model_len`，关闭 async chunk、CUDA Graph、CFG 和多请求并发。

### 验收

pipeline factory 能从 checkpoint 的 `model_type="breeze"` 推导出 pipeline，并能构造两个 stage 的 engine args。

## 12. 实现权重加载和形状审计

### 修改文件

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_depth.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_text.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_codec.py
tests/model_executor/models/breeze_tts_2/test_weight_loading.py
```

### 要写什么

按此顺序实现：

1. `backbone_model.*`：去掉外层前缀，委托 `Qwen3Model.load_weights()`；
2. `lm_head.weight`：加载到 `(2052, 2048)` 的 Breeze head；
3. `embed_text_tokens.weight`：加载到 text embedding；
4. `backbone_model.embed_tokens.embed_audio_tokens.weight`：加载 audio embedding；
5. `depth_decoder.model.*`：加载 depth decoder；
6. `depth_decoder.codebooks_head.weight`：检查 `(15, 1024, 2051)`；
7. `text_encoder.*`、`text_encoder_proj.weight`：加载 text encoder；
8. codec 权重由 stage 1 独立加载。

每次加载都统计 `loaded`、`missing`、`unexpected` 和 `shape_mismatches`。任何模型参数缺失或 shape mismatch 都应失败。

候选 key 前缀以实际 safetensors 为准，预期包括：

```text
lm_head.weight
embed_text_tokens.weight
backbone_model.embed_tokens.embed_audio_tokens.weight
backbone_model.layers.*
backbone_model.norm.weight
depth_decoder.model.embed_tokens.weight
depth_decoder.model.inputs_embeds_projector.weight
depth_decoder.model.layers.*
depth_decoder.model.norm.weight
depth_decoder.codebooks_head.weight
text_encoder.*
text_encoder_proj.weight
```

### 验收

`test_weight_loading.py` 必须证明 stage 0 和 stage 1 参数完整加载，Qwen3 fused 参数没有漏载，depth head 和 text projection 不是随机初始化。

## 13. 完成第一条端到端同步链路

### 修改/新增文件

```text
tests/model_executor/models/breeze_tts_2/test_e2e_sync.py
```

### 实施步骤

1. 加载配置和 stage 0；
2. 创建 plain TTS 输入；
3. 运行 prefill；
4. 循环生成 codebook-0 和 depth codes；
5. stage 0 输出 `(T, 16)`；
6. `talker2codec()` 转成 stage 1 payload；
7. stage 1 解码 waveform；
8. 检查采样率、shape、NaN、长度和完成状态。

### 验收标准

一条 plain TTS 能出声音，`T` 不超过 `max_new_frames`，code 不越界，且长度、前几帧 code 和上游大致一致。

## 14. 增加 reference voice、instruction 和 CFG

### 修改文件

```text
vllm_omni/model_executor/stage_input_processors/breeze_tts_2.py
tests/model_executor/models/breeze_tts_2/test_reference_voice.py
tests/model_executor/models/breeze_tts_2/test_instruction.py
```

### 实施顺序

1. reference audio + exact transcript；
2. instruction + target text；
3. reference + instruction 的 voice direction；
4. 每种模式单独检查 segment 拼接、`text_ids_len` 和 audio code 对齐；
5. CFG 分支暂不进入第一版：adapter 对 `extra_params.guidance_scale`/
   `cfg_scale` 非 1.0 和 `negative_prompt` 直接返回明确错误，避免静默忽略用户条件；
   后续实现 CFG 时再增加独立的 negative/reference/instruction 分支。

当前同步实现等价于 `cfg_scale=1.0`，不会创建额外 companion request，也不会改变
vLLM scheduler 的请求状态机。

## 15. 扩展多请求、异步 chunk 和 streaming

### 修改文件

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
vllm_omni/model_executor/stage_input_processors/breeze_tts_2.py
vllm_omni/model_executor/models/breeze_tts_2/pipeline.py
tests/model_executor/models/breeze_tts_2/test_batching.py
tests/model_executor/models/breeze_tts_2/test_streaming.py
```

### 要写什么

- state 按 request id 保存，不能使用全局单请求变量；
- `make_omni_output()` 使用 `request_token_spans` 找每请求最后一个 hidden；
- 每请求独立累计 `(T, 16)`；
- chunk 只发送新增 tail，不能重复发送完整历史；
- EOS 时 flush 剩余帧并清理 state；
- full payload 和 async chunk 分开测试。

### 验收

两条不同长度请求并发时不串请求；一条先 EOS 不影响另一条；chunk 和 full payload 解码结果一致；异常后 state 清理。

## 16. 最后做性能优化

只有前面 correctness 通过后，才做：

1. Qwen3 KV cache 和 continuous batching；
2. text encoder batching；
3. depth decoder fused projection；
4. bundled Qwen3-TTS decoder 的 chunk decode（Mimi fallback 同样按 chunk 验证）；
5. torch.compile；
6. CUDA Graph；
7. FP8/量化。

不要直接搬上游 `models/cudagraph/*`，因为它们是固定 batch、固定 shape、单请求 runtime。

## 17. 最终文件清单

### 必须新增/修改

```text
vllm_omni/model_executor/models/breeze_tts_2/__init__.py
vllm_omni/model_executor/models/breeze_tts_2/configuration_breeze_tts_2.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_depth.py
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_text.py
vllm_omni/model_executor/models/breeze_tts_2/prompt_builder.py
vllm_omni/model_executor/models/breeze_tts_2/audio_tokenizer.py
vllm_omni/entrypoints/openai/tts_adapters/breeze_tts_2.py
vllm_omni/model_executor/models/breeze_tts_2/pipeline.py
vllm_omni/model_executor/models/registry.py
vllm_omni/config/pipeline_registry.py
vllm_omni/engine/arg_utils.py
vllm_omni/model_executor/stage_input_processors/breeze_tts_2.py
```

### codec 相关实现

```text
vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_codec.py
vllm_omni/deploy/breeze_tts_2.yaml
```

`audio_tokenizer.py` 和 serving adapter 是第四章的必需实现；只有 stage 1
codec wrapper 或部署配置需要按当前仓库拓扑调整时，才需要修改上面两个文件。

不需要新增 `text_tokenizer.py`：文本 tokenizer 使用 checkpoint 自带的 HuggingFace tokenizer，由 stage 0 的 prompt builder/talker 按 Qwen3-TTS 的方式加载。

## 18. 最短落地路线

严格按下面 10 个提交点推进：

1. 配置解析和 `get_text_config()`；
2. `BreezeForConditionalGeneration` 架构注册；
3. `prompt_builder.py` + 文本 tokenizer + plain prompt；
4. Breeze audio embedding + text encoder projection；
5. Qwen3 backbone hidden 对齐；
6. 2052 类 codebook-0 head 对齐；
7. 15 路 depth decoder 对齐；
8. 单请求 greedy 状态机；
9. bundled Qwen3-TTS tokenizer stage 1 同步 decode（无 bundled 目录时 fallback 到 Mimi）；
10. pipeline registry + 端到端回归测试。

做到第 10 步，才算完成“Breeze-TTS-2 在 vLLM-Omni 上可运行”的第一版。之后的 CFG、reference voice、并发、streaming 和 CUDA Graph 都是增量工作。
