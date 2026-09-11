# Kimi-Audio online serving

Kimi-Audio accepts user/assistant text and audio messages and generates text,
audio, or both. The current deployment collects AR output before acoustic
decoding (`async_chunk: false`). The requests below use non-streaming responses.

Install vLLM-Omni with its `kimi-audio` extra, including the acoustic decoder
dependencies, and restart the server after installing this branch. The install
registers `kimi_audio_omni`: it selects the Omni renderer while reusing vLLM's
Kimi-Audio tokenizer.

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni \
  --stage-configs-path vllm_omni/deploy/kimi_audio.yaml \
  --trust-remote-code --port 8091
```

A local checkpoint directory can replace the model ID. The input encoder also
needs `THUDM/glm-4-voice-tokenizer`; loading resolves the pinned revision. To use
an existing local snapshot, add
`--additional-config '{"kimi_audio":{"glm_tokenizer_path":"/path/to/glm-tokenizer"}}'`.
The checkpoint's `whisper-large-v3`, `audio_detokenizer` and `vocoder` files must
also be present. GPU startup and memory sizing still require validation.

## Chat

```bash
curl http://localhost:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/Kimi-Audio-7B-Instruct",
    "messages": [{"role": "user", "content": "你好，请简短介绍一下自己。"}],
    "chat_template_kwargs": {"output_type": "both"},
    "modalities": ["text", "audio"],
    "audio": {"format": "wav"},
    "max_tokens": 512,
    "stream": false
  }'
```

`chat_template_kwargs.output_type` selects the model task: `text` generates
text, while `both` generates the text/audio pair. It defaults to `both` for chat.
Top-level `modalities` selects which stage outputs to return; it does not change
the model task. Set both fields explicitly:

| Model `output_type` | Response `modalities` | Behavior |
| --- | --- | --- |
| `text` | `["text"]` | Text generation; finish at the AR stage. |
| `both` | `["text"]` | Run dual-stream AR but return only text; skip the acoustic decoder. |
| `both` | `["audio"]` | Run dual-stream AR and acoustic decoding; return only audio. |
| `both` | `["text", "audio"]` | Return text and decoded audio. |

`output_type: "text"` with audio in `modalities` is incompatible. The renderer
does not receive top-level `modalities`; the existing AR-to-decoder boundary
rejects this combination **after AR completes** through Omni's existing
request-error path. This is not early HTTP input validation.
For text-only requests, set `output_type: "text"` and
`modalities: ["text"]` together. Do not put `modalities` inside
`chat_template_kwargs`; that experimental option has been replaced by
`output_type` and is rejected.

Omitting top-level `modalities` uses the deployed output modalities, which include
text and audio in the supplied two-stage configuration. The example uses
`max_tokens` because the current Omni stage-parameter override reads that field;
the Kimi renderer rejects `max_completion_tokens` to avoid silently using the
deployment's output-length default.
Following Omni's existing response format, text and audio appear in separate
`choices`; decode `message.audio.data` from base64 to obtain the audio file.

For audio input, replace `content` with OpenAI content parts, for example:

```json
[
  {"type": "text", "text": "请转写这段音频。"},
  {"type": "input_audio", "input_audio": {"data": "<base64 WAV bytes>", "format": "wav"}}
]
```

`audio_url` parts are also supported through vLLM's normal media access rules.
For transcription, combine these content parts with `output_type: "text"` and
`modalities: ["text"]`. Kimi requires mono audio; the renderer resamples it to
16 kHz and reuses the input builder. Whisper feature extraction runs in the
model's multimodal processor. Adjacent text parts within one message are joined
without adding whitespace; include any required spaces in the text itself.

Assistant audio history must provide one recording with its transcript as
content parts, for example this message within the conversation:

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "你好，我可以帮你。"},
    {"type": "input_audio", "input_audio": {"data": "<base64 WAV bytes>", "format": "wav"}}
  ]
}
```

Opaque `audio.id` references and cache-only audio UUIDs are not supported; send
the audio content again with its transcript. The normal media cache remains in
use.

The current chat contract covers one completion with `stream: false` and
ordinary user/assistant messages. System/tool messages, custom templates,
reasoning or structured-output modes, prompt truncation/padding and prompt
echo/offsets are outside this contract. Kimi uses its fixed audio preprocessing
configuration; `mm_processor_kwargs` and per-part `uuid` overrides are rejected.

## Speech

```bash
curl http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/Kimi-Audio-7B-Instruct",
    "input": "你好，欢迎使用 Kimi-Audio。",
    "voice": "default",
    "response_format": "wav",
    "max_new_tokens": 512
  }' --output speech.wav
```

This adapter asks the conversational model to read the input aloud; verbatim
reading is not enforced by the decoder. Voice selection, voice cloning and
streaming speech are not implemented. Standard speech formatting uses the
shared server. `seed` and `extra_params.kimi_audio` use the existing request
sampling mechanisms.

## Status

These examples describe the current input contract. Full server startup,
pretrained output quality and concurrency have not been validated. Existing
tests have not been updated for this contract. Cross-stage `async_chunk`
streaming remains unimplemented.
