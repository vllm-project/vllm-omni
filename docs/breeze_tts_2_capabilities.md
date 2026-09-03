# Breeze-TTS-2 当前能力与限制

本文是当前 vLLM-Omni 代码的能力基线，回答两个问题：现在已经接入了什么，以及调用 Breeze-TTS-2 时实际能传什么、得到什么。它描述的是当前实现，不等同于上游 Breeze 仓库的完整功能列表。

适配实施步骤见：[Breeze-TTS-2 适配手册](breeze_tts_2_adaptation.md)。本文对应当前工作区的实现状态；模型权重、音频 tokenizer 和运行环境仍需由部署方提供。

## 1. 范围与结论

当前实现是一条同步的两阶段 AR TTS 链路：

```text
OpenAI /v1/audio/speech
  -> BreezeTTS2Adapter
  -> HuggingFace text tokenizer + Breeze prompt builder
  -> Stage 0: T5Gemma2 + Qwen3 + depth decoder
  -> frame-major audio codes (T, 16)
  -> Stage 1: Qwen3-TTS audio tokenizer decoder (或 Mimi fallback)
  -> 24 kHz float32 waveform
  -> wav/pcm 等 HTTP 音频响应
```

当前可作为验证目标的是：单条请求、greedy/eager、完整 payload 传输、同步解码。代码已经包含批处理相关的数据结构，但多请求连续 batching、增量 chunk 和实时首包延迟尚未完成端到端验证。

## 2. 功能状态矩阵

| 能力 | 状态 | 当前行为 |
| --- | --- | --- |
| 普通文本合成 | 已实现 | `input` 文本进入 Breeze `tts_plain` prompt，生成 waveform |
| 文本 + 风格指令 | 已实现 | `instructions` 进入 `tts_instruction` prompt |
| 单参考音频克隆 | 已实现 | `ref_audio` 与 `ref_text` 同时提供，使用 `ref_clone_tata` |
| 参考音频 + 编辑/风格指令 | 已实现 | 三者同时提供，使用 `ref_edit_tata` |
| `voice`/speaker tag | 有条件支持 | 作为 `[S0]`、`[S1]` 等 Breeze speaker 前缀；不校验固定 speaker 清单 |
| 预编码 reference codes | 已实现（内部接口） | prompt builder 接受 `ref_audio_codes` 或 `codes["ref"]`，主要供测试/内部调用 |
| Stage 0 16 路 codec code 生成 | 已实现 | codebook-0 由主 head 采样，codebook-1..15 由 depth decoder 补齐 |
| Stage 1 code -> waveform | 已实现 | 优先使用 checkpoint 的 `audio_tokenizer/`，否则尝试 Mimi fallback |
| 非流式 `/v1/audio/speech` | 已实现 | 完整 waveform 解码后返回 |
| `wav`/`pcm`/其他通用格式编码 | 有条件支持 | 由通用 serving 层编码；Breeze 原生输出仍是 24 kHz waveform |
| `sample_rate=24000` | 已实现 | 适配器唯一声明的目标采样率 |
| 其他目标采样率 | 未支持 | 请求会被适配器拒绝，不做 Breeze 专属重采样 |
| HTTP/SSE 增量 streaming | 未实现 | 当前 pipeline `async_chunk: false`，Stage 1 依赖完整 codes |
| CFG / `negative_prompt` | 未实现 | `guidance_scale`/`cfg_scale` 只能为 `1.0`，否则报错 |
| `speed` 调整 | 未实现 | 仅允许 `speed=1.0` |
| `VoiceDesign` | 未实现 | `task_type=VoiceDesign` 明确报错 |
| 多参考音频/多说话人 | 未实现 | 只取单个 `ref_audio`；不支持 `ref_audio_2` |
| LoRA、量化、CUDA graph | 未实现/未验证 | 不属于当前 Breeze 适配的验证范围 |

“已实现”表示代码路径已存在；并不表示已经在所有 GPU、权重格式和并发配置下完成性能验收。

## 3. OpenAI TTS 输入

入口是 `POST /v1/audio/speech`。字段由通用 `OpenAICreateSpeechRequest` 接收，再由 `BreezeTTS2Adapter` 做模型专属校验和 prompt 构造。

| 字段 | 必填 | Breeze 当前含义与限制 |
| --- | --- | --- |
| `input` | 是 | 待合成文本；不能为空或全空白 |
| `voice`（或别名 `speaker`） | 否 | 默认 `S0`；任意字符串会规范成 Breeze speaker tag，当前不提供固定 speaker 列表 |
| `instructions` | 否 | 风格、情绪或表达指令；有值时选择 instruction/edit prompt |
| `ref_audio` | 否 | URL、`data:` 或 `file://` URI；必须与 `ref_text` 成对出现；列表当前只取第一项 |
| `ref_text` | 否 | 参考音频转写文本；必须与 `ref_audio` 成对出现，不能为空 |
| `task_type` | 否 | `Base` 需要 reference input；`VoiceDesign` 不支持；未指定时由是否有 reference/instruction 推导 prompt |
| `max_new_tokens` | 否 | 对 Breeze 按生成 frame 上限处理，并写入 `breeze_max_new_frames`；适配器范围为 1 到 4096 |
| `sample_rate` | 否 | 仅允许 `24000`；不填则使用 codec 返回的原生采样率（当前应为 24000） |
| `response_format` | 否 | 默认 `wav`。通用层声明支持 `wav`、`pcm`、`flac`、`mp3`、`opus`；格式转换依赖通用音频编码环境 |
| `speed` | 否 | 仅允许 `1.0`；Breeze 没有原生变速参数 |
| `extra_params.guidance_scale` / `cfg_scale` | 否 | 只能为数值 `1.0`；其他值被拒绝 |
| `extra_params.negative_prompt` | 否 | 不支持，出现即报错 |
| `language` | 否 | 请求 schema 接收，但当前 Breeze adapter 不把它注入 prompt，也不做语言白名单校验 |
| `seed` | 否 | 当前 adapter 不向 Breeze prompt 注入专用 seed 语义；不要把它当作已验证的确定性保证 |
| `ref_audio_2` | 否 | Breeze 单说话人链路不支持，应视为不可用 |
| `speaker_embedding`、`x_vector_only_mode` | 否 | 这些是其他 TTS 模型的字段，当前 Breeze 不消费 |

参考音频支持两种数据形态：

1. HTTP 层传 URI，serving 层先解析为 waveform 和采样率，再由 `BreezeReferenceAudioTokenizer` 调用 bundled `Qwen3TTSTokenizer.encode()`；
2. 内部 prompt builder 直接接收 `(waveform, sample_rate)` 或 `(T, 16)` 的预编码 codes。

## 4. 四种 Prompt 模式

| 模式 | 触发条件 | 组成 | 备注 |
| --- | --- | --- | --- |
| `tts_plain` | 只有 `input`（可带 `voice`） | speaker + text | 普通合成 |
| `tts_instruction` | `input` + `instructions`，无 reference | speaker + instruction BOS/EOS + text | instruction 是模型 prompt 的一部分 |
| `ref_clone_tata` | `ref_audio` + `ref_text` + `input`，无 instruction | speaker + ref text + audio placeholders + speaker + target text | 单参考音频克隆 |
| `ref_edit_tata` | reference 三件套再加 `instructions` | speaker + ref text + audio placeholders + speaker + instruction + target text | 参考声音上的编辑/风格控制 |

`prompt_builder.py` 会先用 checkpoint 根目录的 HuggingFace `AutoTokenizer` 编码文本 segment，再按上游 Breeze 规则渲染 audio placeholder 并对完整 prompt 做最终 tokenization。调度器看到的 `prompt_token_ids` 是合法 Qwen3 词表内的 dummy/pad id；真实的 Breeze 高位 text/audio id 保存在 `additional_information["prompt_ids"]`，由 Stage 0 talker 消费。这是为了避免把 Breeze 的 `262144+` 音频 token 直接交给 Qwen3 scheduler。

## 5. Stage 0 输入输出契约

### 输入

Stage 0 模型是 `BreezeTTS2TalkerForGeneration`，架构名为 `BreezeForConditionalGeneration`。它接收：

- scheduler 的占位 `prompt_token_ids`；
- `additional_information["prompt_ids"]`：真实 Breeze prompt ids；
- `text_ids_mask`、`text_ids_len`：文本 segment 的位置和长度；
- 可选 `input_values`：reference audio 的 `(T_ref, 16)` int16 codes；
- 可选 `breeze_max_new_frames`：生成帧上限。

模型内部依次使用 T5Gemma2 text encoder（1152 -> 2048 projection）、Qwen3 backbone、主 codebook-0 head 和 depth decoder。主 head 输出 2052 类，`2051` 是 Breeze 主 EOS；depth decoder 每个 head 输出 2051 类，只有 `[0, 2047]` 是合法 codec id。

### 输出

每个请求的 `multimodal_outputs["codes"]["audio"]` 是累计的 frame-major tensor：

```text
(T, 16), dtype=int16/long
```

其中 `T` 是已经生成的音频帧数，16 是 Breeze codebook 数。`[2048, 2050]` 等保留 codec id 会被 mask。达到 `max_new_frames` 时，talker 会强制结束并保留已经累计的帧。

## 6. Stage 1 输入输出契约

`talker2codec` 或 full-payload hook 将 Stage 0 的 `(T, 16)` 转置为 codebook-major `(16, T)` 并展平为长度 `16*T` 的序列。Stage 1 `BreezeTTS2MimiCodec` 再恢复布局并解码：

```text
codes.audio: (16*T,) codebook-major
  -> codes: (1, 16, T)
  -> waveform: 1-D CPU float32
  -> sr: int32 tensor, normally 24000
```

Stage 1 优先加载 `<checkpoint>/audio_tokenizer/` 中的 `Qwen3TTSTokenizer.decode()`。只有该目录不存在时，才尝试用 `codec_config` 初始化/加载 Mimi。Breeze 的 16 路 codebooks 与某些 Mimi 配置中的 32 quantizers 不是同一个接口，不能混用。

## 7. HTTP 输出

非流式请求在 Stage 1 完成后由通用 serving 层读取：

- `multimodal_outputs["model_outputs"]`：每请求一个 CPU waveform tensor；
- `multimodal_outputs["sr"]`：对应采样率。

随后按 `response_format` 编码并返回音频 bytes。默认是 `wav`。`pcm` 可用于原始 PCM；`flac`、`mp3`、`opus` 是否可用取决于通用编码依赖，不改变 Breeze 的原生 24 kHz 产物。

典型普通合成请求：

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "breeze-tts-2",
    "input": "你好，这是 Breeze-TTS-2。",
    "voice": "S0",
    "response_format": "wav",
    "sample_rate": 24000
  }' \
  --output output.wav
```

典型参考音频请求：

```json
{
  "model": "breeze-tts-2",
  "input": "这是要合成的新文本。",
  "ref_audio": "file:///tmp/reference.wav",
  "ref_text": "参考音频对应的转写文本",
  "instructions": "保持自然、平静的语气",
  "response_format": "wav"
}
```

## 8. 当前限制与部署注意事项

1. **同步 full-payload**：Stage 0 每一步输出累计 codes，full-payload 处理器采用 replace 语义，不能按普通增量 tensor 再 concat；Stage 1 必须等待完整序列。
2. **没有实时增量输出**：部署配置中的 `async_chunk: false` 是当前设计前提。HTTP `stream` 入口不应被宣传为 Breeze 的实时低延迟流式合成。
3. **CFG 被拒绝**：非 `1.0` 的 guidance scale 和 `negative_prompt` 会返回客户端错误，避免误以为已启用无分类器引导。
4. **reference 必须成对**：`ref_audio` 与 `ref_text` 缺一不可；当前只处理一个 reference。
5. **任务范围有限**：不支持 `VoiceDesign`、多说话人、第二参考音频、speed 调整以及 x-vector-only speaker embedding。
6. **采样率固定**：适配器只声明 24 kHz。请求任意其他 `sample_rate` 会在进入推理前失败。
7. **长度受两处约束**：prompt 长度受 stage 0 `max_model_len` 限制；生成长度受 `max_new_tokens`/`max_tokens` 和 `breeze_max_new_frames` 限制。当前部署默认 stage 0 `max_model_len=4096`、stage 0 `max_num_seqs=8`，stage 1 `max_num_seqs=1`、`max_model_len=65536`，需按显存和音频时长调整。
8. **权重与目录必须匹配**：缺少 text encoder、depth decoder、Breeze head 或 `audio_tokenizer/` 时，不能假设普通 Qwen3/Mimi 权重可以直接替代；加载器会对关键权重 shape 做校验。
9. **缓存与设备**：参考音频 tokenizer 在 serving worker 中复用；Stage 1 tokenizer/codec 在 worker 初始化后复用，最终 waveform 搬到 CPU 供序列化，GPU 显存仍需覆盖 codec 推理峰值。
10. **验证状态**：当前静态检查和单元测试覆盖 prompt、audio tokenizer 归一化、stage payload 和 pipeline 注册；完整 pytest 需要安装 PyTorch，并且仍需使用真实 Breeze checkpoint 做端到端音频回归。

## 9. 尚未实现的后续工作

建议按以下顺序推进：

1. 用真实 checkpoint 完成单请求 golden code/waveform 对齐；
2. 验证两条不同长度请求的 batching、EOS 和 state 隔离；
3. 将 Stage 0 累计 codes 改造为可安全传输的 async chunk，并补齐 Stage 1 增量 Code2Wav；
4. 再评估 HTTP SSE/WebSocket streaming、CFG、CUDA graph、量化和 LoRA；
5. 最后根据真实延迟/显存数据调整 `max_num_seqs`、`max_model_len` 和 codec 缓存策略。

## 10. 代码索引

| 文件 | 职责 |
| --- | --- |
| `vllm_omni/entrypoints/openai/tts_adapters/breeze_tts_2.py` | OpenAI 请求校验、四种模板选择、builder/参考音频 tokenizer 生命周期 |
| `vllm_omni/model_executor/models/breeze_tts_2/prompt_builder.py` | 文本 tokenizer 调用、Breeze prompt 模板、text mask 和 reference code metadata |
| `vllm_omni/model_executor/models/breeze_tts_2/audio_tokenizer.py` | waveform -> reference `(T,16)` codes 的包装，不负责文本或最终 waveform 解码 |
| `vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_talker.py` | Stage 0 文本/音频 embedding、Qwen3、codebook-0 和 depth decoder |
| `vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_text.py` | T5Gemma2 兼容 text encoder |
| `vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_depth.py` | codebook-1..15 depth decoder |
| `vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze_tts_2_codec.py` | Stage 1 codes -> waveform |
| `vllm_omni/model_executor/stage_input_processors/breeze_tts_2.py` | Stage 0 到 Stage 1 的布局转换和 full-payload replace |
| `vllm_omni/model_executor/models/breeze_tts_2/pipeline.py` | 两阶段 pipeline 声明 |
| `vllm_omni/deploy/breeze_tts_2.yaml` | 当前同步部署参数 |
| `tests/model_executor/models/breeze_tts_2/` | prompt、tokenizer、payload、pipeline 注册单元测试 |

