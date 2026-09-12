# Kimi-Audio online serving

Kimi-Audio accepts user/assistant text and audio messages and generates text,
audio, or both. `kimi_audio.yaml` collects AR output before acoustic decoding;
`kimi_audio_async_chunk.yaml` transfers semantic codes while AR is generating.
Both use the same input path and acoustic networks.

For direct Python inference without an HTTP server, see the
[offline example](../../offline_inference/kimi_audio/README.md).

Both deployments use the official inference example's audio sampling settings:
`audio_temperature=0.8` and `audio_top_k=10`, with greedy text sampling.
The two streams have separate settings; setting the text `temperature` to zero
does not make audio sampling greedy.

Install vLLM-Omni with its `kimi-audio` extra, including the acoustic decoder
dependencies, and restart the server after installing this branch. The install
registers `kimi_audio_omni`: it selects the Omni renderer while reusing vLLM's
Kimi-Audio tokenizer.

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni \
  --deploy-config vllm_omni/deploy/kimi_audio.yaml \
  --trust-remote-code --port 8091
```

A local checkpoint directory can replace the model ID. The input encoder also
needs `THUDM/glm-4-voice-tokenizer`; loading resolves the pinned revision. To use
an existing local snapshot, add
`--additional-config '{"kimi_audio":{"glm_tokenizer_path":"/path/to/glm-tokenizer"}}'`.
The checkpoint's `whisper-large-v3`, `audio_detokenizer` and `vocoder` files must
also be present. Startup has been exercised on a single A800 80GB; memory
settings are not tuned for other devices or larger request capacities.

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
does not receive top-level `modalities`. In batch mode the AR-to-decoder boundary
rejects this combination after AR completes. In async-chunk mode the producer
records the conversion failure, but the framework does not propagate it as an
immediate HTTP error; it can result in empty audio. Joint HTTP validation and
streaming conversion-error propagation remain incomplete. Use the compatible
combinations above.
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

The current chat contract covers one completion and
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
reading is not enforced by the decoder. Voice selection and voice cloning
are not implemented. Standard speech formatting uses the
shared server. `seed` and `extra_params.kimi_audio` use the existing request
sampling mechanisms.

## Streaming

Start the server with the streaming deployment:

```bash
vllm serve moonshotai/Kimi-Audio-7B-Instruct --omni \
  --deploy-config vllm_omni/deploy/kimi_audio_async_chunk.yaml \
  --trust-remote-code --port 8091
```

For chat SSE, use the chat request above with `"stream": true` and
`"audio": {"format": "pcm16"}`. Text deltas and base64-encoded audio deltas
use Omni's existing chat response format. Concatenate decoded PCM bytes in
arrival order; they are 24 kHz, mono, signed 16-bit audio.

For raw streaming speech:

```bash
curl --no-buffer http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/Kimi-Audio-7B-Instruct",
    "input": "你好，欢迎使用 Kimi-Audio。",
    "response_format": "pcm",
    "stream": true,
    "stream_format": "audio",
    "max_new_tokens": 512
  }' --output speech.pcm
```

Omit `stream_format` to receive the shared speech SSE events instead of raw
bytes. Streaming speech requires the async-chunk deployment; non-streaming
responses can also use this deployment and are accumulated by Omni.

The producer preserves official 30-code blocks, holding one code beyond each
full block so the last block can be marked final. First decoding starts after
31 valid audio codes, or earlier if the request ends. The decoder retains its
12-token lookahead and waveform overlap per request and emits only new samples.
This preserves the acoustic block boundaries; it is not a tuned low-latency
configuration. Both stages default to one active request; the manual concurrency
run used `max_num_seqs=2` on both stages. An explicit request seed also seeds request-local
acoustic noise; it does not promise bitwise equality across GPU configurations.

## Status

Manual runs on an A800 80GB exercised pretrained-weight startup, offline text
and audio inputs, non-streaming chat and speech, chat and speech SSE, two
concurrent chat requests, and client cancellation followed by new requests.
For the short chat and speech SSE cases, concatenated PCM matched the saved
non-streaming reference. Audio review used saved WAV files. These observations
cover the recorded cases, not arbitrary inputs or concurrent workloads.

The instructed-reading examples exposed a missing opening greeting and, in a
longer case, repeated content. Both also occurred in independent official
reference runs under the compared conditions; their root causes remain
unresolved. One transcription omitted reference punctuation while retaining
the words. Do not interpret HTTP completion or an audio completion event as
proof of verbatim speech or natural AR termination.

`meta.finished` can produce a scalar-tensor concatenation warning in the shared
output accumulator, which falls back to keeping the latest value. No output
failure was observed from that warning in these runs; it remains unfixed.

The retained CPU tests are component reference checks.
`tests/e2e/online_serving/test_kimi_audio.py` defines single-request,
concurrent-request and SSE tests against one real server; this automated suite
was not run as part of the manual validation. Multi-GPU PP, raw-byte speech
streaming, real-time player buffering, graph capture, performance tuning and
fault-injection recovery remain outside that validation scope.

Decoder validation and acoustic computation share batch failure cleanup. If a
forward fails, it releases acoustic state for that batch's known request IDs,
including rows already advanced; requests outside the batch retain their state.
This local cleanup does not retry the request or replace framework error handling.

The connector owns transport timeouts and cancellation. Its sender-failure
channel currently logs rather than immediately failing the client request;
this implementation does not change that shared behavior.
